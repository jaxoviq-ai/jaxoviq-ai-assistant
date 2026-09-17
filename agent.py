import io
import json
import time
import os
import uuid
import html
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)


# ============================================================
# JAXOVIQ AI ASSISTANT
# Professional Multi-PDF RAG Assistant
# ============================================================

client = OpenAI()
TRACK_FILE = "usage_stats.json"

def load_stats():
    if os.path.exists(TRACK_FILE):
        try:
            with open(TRACK_FILE, "r") as f:
                return json.load(f)
        except:
            pass

    return {
        "app_opens": 0,
        "pdf_uploads": 0,
        "questions": 0,
        "external_sessions": 0,
        "admin_tests": 0,
    }
   
def save_stats(stats):
    with open(TRACK_FILE, "w") as f:
        json.dump(stats, f)

if "usage_counted" not in st.session_state:
    stats = load_stats()

    stats.setdefault("app_opens", 0)
    stats.setdefault("pdf_uploads", 0)
    stats.setdefault("questions", 0)
    stats.setdefault("external_sessions", 0)
    stats.setdefault("admin_tests", 0)

    stats["app_opens"] += 1
    stats["external_sessions"] += 1

    save_stats(stats)

    st.session_state.usage_counted = True
    st.session_state.visitor_id = uuid.uuid4().hex[:8]
    st.session_state.visitor_counted = True

MIN_SCORE = 0.30
MEDIUM_SCORE = 0.45
HIGH_SCORE = 0.60

TOP_K = 8
RERANK_TOP_N = 4

EMBEDDING_MODEL = "text-embedding-3-small"
AI_MODEL = "gpt-5.6-luna"


# ============================================================
# PERSISTENT KNOWLEDGE BASE PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "jaxoviq_data"

INDEX_FILE = DATA_DIR / "knowledge.index"
CHUNKS_FILE = DATA_DIR / "chunks.json"
META_FILE = DATA_DIR / "meta.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="JAXOVIQ AI Assistant",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# PROFESSIONAL UI CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.18);
    }

    .jaxoviq-hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 18px;
        margin-bottom: 1rem;
    }

    .jaxoviq-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0;
    }

    .jaxoviq-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-top: 0.35rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.22);
        padding: 0.9rem;
        border-radius: 14px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "index": None,
    "chunks": [],
    "messages": [],
    "pdf_names": [],
    "knowledge_base_ready": False,

    "pdf_summary": None,
    "summary_scope": None,

    "action_items": None,
    "action_scope": None,

    "document_insights": None,
    "insights_scope": None,

    "export_report": None,
    "export_scope": None,

    "pdf_report_bytes": None,
    "pdf_report_scope": None,

    "suggested_questions": [],
    "suggestions_scope": None,
    "pending_question": None,
    "admin_test_mode": False,
    "visitor_id": None,
    "visitor_counted": False,
    "persistent_loaded": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# OPENAI HELPERS
# ============================================================

def get_embedding(text):
    for attempt in range(1, 4):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )

            return response.data[0].embedding

        except Exception as e:
            if attempt < 3:
                time.sleep(2)

            else:
                st.error(f"Embedding error: {e}")
                return None


def ask_model(prompt):
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=AI_MODEL,
                reasoning_effort="none",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            if attempt < 3:
                time.sleep(2)

            else:
                st.error(f"AI error: {e}")
                return None


# ============================================================
# CONVERSATION-AWARE FOLLOW-UP
# ============================================================

def get_recent_conversation(max_messages=6):
    messages = st.session_state.messages[-max_messages:]

    if not messages:
        return ""

    lines = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "user":
            speaker = "USER"
        else:
            speaker = "JAXOVIQ"

        lines.append(
            f"{speaker}: {content}"
        )

    return "\n".join(lines)


def resolve_followup_question(question):
    conversation = get_recent_conversation()

    if not conversation:
        return question

    prompt = f"""
You are JAXOVIQ's conversation query resolver.

Your job is ONLY to rewrite the latest user question
into a clear standalone question for PDF search.

Use the previous conversation only to understand references
such as:

- it
- this
- that
- this budget
- that date
- he
- she
- they
- this goal
- this project
- this amount

Rules:

1. Do NOT answer the question.
2. Do NOT invent facts.
3. Keep the meaning exactly the same.
4. Use previous conversation only to resolve references.
5. If the latest question is already standalone, return it unchanged.
6. Return ONLY the rewritten question.
7. Do not add quotation marks.
8. Keep it short and natural.

PREVIOUS CONVERSATION:

{conversation}

LATEST USER QUESTION:

{question}

STANDALONE QUESTION:
"""

    try:
        resolved = ask_model(prompt)

        if not resolved:
            return question

        resolved = resolved.strip()

        if not resolved:
            return question

        return resolved

    except Exception:
        return question


# ============================================================
# PERSISTENT KNOWLEDGE BASE
# ============================================================

def save_persistent_knowledge_base(
    index,
    chunks,
    pdf_names,
):
    try:
        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            index,
            str(INDEX_FILE),
        )

        with open(
            CHUNKS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        with open(
            META_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "pdf_names": pdf_names,
                    "embedding_model": EMBEDDING_MODEL,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        return True

    except Exception as e:
        st.warning(
            f"Knowledge base worked, but could not save locally: {e}"
        )
        return False


def load_persistent_knowledge_base():
    if not (
        INDEX_FILE.exists()
        and CHUNKS_FILE.exists()
        and META_FILE.exists()
    ):
        return None

    try:
        index = faiss.read_index(
            str(INDEX_FILE)
        )

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(file)

        with open(
            META_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        pdf_names = metadata.get(
            "pdf_names",
            [],
        )

        saved_model = metadata.get(
            "embedding_model"
        )

        if saved_model != EMBEDDING_MODEL:
            return None

        if index.ntotal != len(chunks):
            return None

        return {
            "index": index,
            "chunks": chunks,
            "pdf_names": pdf_names,
        }

    except Exception:
        return None


def delete_persistent_knowledge_base():
    files = [
        INDEX_FILE,
        CHUNKS_FILE,
        META_FILE,
    ]

    for file_path in files:
        try:
            if file_path.exists():
                file_path.unlink()

        except Exception:
            pass


# ============================================================
# PDF PROCESSING
# ============================================================

def read_uploaded_pdf(uploaded_file):
    pages = []

    try:
        pdf_bytes = uploaded_file.getvalue()

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text()

            if text and text.strip():
                pages.append(
                    {
                        "text": text.strip(),
                        "page": page_number,
                        "file": uploaded_file.name,
                    }
                )

        return pages

    except Exception as e:
        st.error(
            f"Could not read {uploaded_file.name}: {e}"
        )

        return []


def chunk_pages(
    pages,
    chunk_size=1000,
    overlap=150,
):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "file": page["file"],
                        "page": page["page"],
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def build_faiss_index(uploaded_files):
    all_chunks = []

    for uploaded_file in uploaded_files:
        pages = read_uploaded_pdf(
            uploaded_file
        )

        all_chunks.extend(
            chunk_pages(pages)
        )

    if not all_chunks:
        return None, None

    embeddings = []

    progress = st.progress(0)
    status = st.empty()

    total = len(all_chunks)

    for i, chunk in enumerate(
        all_chunks,
        start=1,
    ):
        status.write(
            f"Creating embeddings... {i}/{total}"
        )

        embedding = get_embedding(
            chunk["text"]
        )

        if embedding is None:
            progress.empty()
            status.empty()

            return None, None

        embeddings.append(
            embedding
        )

        progress.progress(
            i / total
        )

    vectors = np.array(
        embeddings,
        dtype="float32",
    )

    faiss.normalize_L2(
        vectors
    )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(
        vectors
    )

    progress.empty()
    status.empty()

    return index, all_chunks


# ============================================================
# SCOPE + DOCUMENT STATS
# ============================================================

def get_scoped_chunks(selected_pdf):
    if selected_pdf == "All Documents":
        return st.session_state.chunks

    return [
        chunk
        for chunk in st.session_state.chunks
        if chunk["file"] == selected_pdf
    ]


def get_document_stats():
    stats = {}

    for chunk in st.session_state.chunks:
        file_name = chunk["file"]
        page_number = chunk["page"]

        if file_name not in stats:
            stats[file_name] = {
                "chunks": 0,
                "pages": set(),
            }

        stats[file_name]["chunks"] += 1

        stats[file_name]["pages"].add(
            page_number
        )

    final_stats = {}

    for file_name, data in stats.items():
        final_stats[file_name] = {
            "chunks": data["chunks"],
            "pages": len(
                data["pages"]
            ),
        }

    return final_stats


# ============================================================
# SMART SUGGESTED QUESTIONS
# ============================================================

def generate_suggested_questions(
    selected_pdf="All Documents",
):
    scoped_chunks = get_scoped_chunks(
        selected_pdf
    )

    if not scoped_chunks:
        return []

    context = "\n\n".join(
        f"""
DOCUMENT: {chunk["file"]}
PAGE: {chunk["page"]}

{chunk["text"]}
"""
        for chunk in scoped_chunks[:25]
    )

    prompt = f"""
You are JAXOVIQ, a professional PDF knowledge assistant.

Read the supplied document content and generate exactly
5 useful questions that a user would genuinely want to ask.

Rules:

1. Questions must be answerable from the supplied PDFs.
2. Do not invent information.
3. Keep each question short and clear.
4. Prefer useful questions about goals, dates, actions,
   people, numbers, risks, plans, or important facts.
5. Return ONLY valid JSON.

Required format:

{{
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?",
    "Question 4?",
    "Question 5?"
  ]
}}

DOCUMENT SCOPE:

{selected_pdf}

PDF CONTENT:

{context}
"""

    try:
        result = ask_model(prompt)

        if not result:
            return []

        cleaned = result.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace(
                "```json",
                "",
            )

            cleaned = cleaned.replace(
                "```",
                "",
            )

            cleaned = cleaned.strip()

        data = json.loads(
            cleaned
        )

        questions = data.get(
            "questions",
            [],
        )

        final_questions = []

        for question in questions:
            if (
                isinstance(question, str)
                and question.strip()
            ):
                final_questions.append(
                    question.strip()
                )

            if len(final_questions) >= 5:
                break

        return final_questions

    except Exception:
        return []


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    selected_pdf="All Documents",
):
    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return [], 0.0

    query_embedding = get_embedding(
        question
    )

    if query_embedding is None:
        return [], 0.0

    query_vector = np.array(
        [query_embedding],
        dtype="float32",
    )

    faiss.normalize_L2(
        query_vector
    )

    chunks = st.session_state.chunks
    index = st.session_state.index

    if selected_pdf == "All Documents":
        k = min(
            TOP_K,
            len(chunks),
        )

    else:
        k = len(chunks)

    scores, indices = index.search(
        query_vector,
        k,
    )

    candidates = []
    filtered_scores = []

    for score, chunk_index in zip(
        scores[0],
        indices[0],
    ):
        score = float(score)

        if chunk_index < 0:
            continue

        chunk = chunks[
            chunk_index
        ]

        if selected_pdf != "All Documents":
            if chunk["file"] != selected_pdf:
                continue

        filtered_scores.append(
            score
        )

        if score >= MIN_SCORE:
            candidates.append(
                {
                    "chunk": chunk,
                    "score": score,
                }
            )

    best_score = (
        max(filtered_scores)
        if filtered_scores
        else 0.0
    )

    return candidates, best_score


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence_label(score):
    if score >= HIGH_SCORE:
        return "🟢 High confidence"

    if score >= MEDIUM_SCORE:
        return "🟡 Medium confidence"

    if score >= MIN_SCORE:
        return "🟠 Low confidence"

    return "🔴 Not enough evidence"


# ============================================================
# RERANKING
# ============================================================

def rerank_chunks(
    question,
    candidates,
):
    if len(candidates) <= 1:
        return candidates

    candidate_text = ""

    for number, item in enumerate(
        candidates,
        start=1,
    ):
        chunk = item["chunk"]

        candidate_text += f"""
CANDIDATE {number}

FILE: {chunk["file"]}
PAGE: {chunk["page"]}

TEXT:
{chunk["text"]}

"""

    prompt = f"""
You are a PDF retrieval reranker.

Rank the candidate chunks by usefulness
for answering the user's question.

Do not answer the question.
Do not use outside knowledge.

Return ONLY valid JSON:

{{"ranking": [1, 2, 3]}}

Use at most {RERANK_TOP_N} candidates.

USER QUESTION:

{question}

CANDIDATES:

{candidate_text}
"""

    try:
        result = ask_model(
            prompt
        )

        if not result:
            return candidates[
                :RERANK_TOP_N
            ]

        cleaned = result.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace(
                "```json",
                "",
            )

            cleaned = cleaned.replace(
                "```",
                "",
            )

            cleaned = cleaned.strip()

        data = json.loads(
            cleaned
        )

        ranking = data.get(
            "ranking",
            [],
        )

        reranked = []

        for number in ranking:
            if not isinstance(
                number,
                int,
            ):
                continue

            position = number - 1

            if (
                0 <= position
                < len(candidates)
            ):
                item = candidates[
                    position
                ]

                if item not in reranked:
                    reranked.append(
                        item
                    )

            if (
                len(reranked)
                >= RERANK_TOP_N
            ):
                break

        if reranked:
            return reranked

    except Exception:
        pass

    return candidates[
        :RERANK_TOP_N
    ]


# ============================================================
# GROUNDED ANSWER
# ============================================================

def generate_answer(
    original_question,
    resolved_question,
    selected_chunks,
):
    context_parts = []

    for item in selected_chunks:
        chunk = item["chunk"]

        context_parts.append(
            f"""
SOURCE: {chunk["file"]}
PAGE: {chunk["page"]}

{chunk["text"]}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are JAXOVIQ, a professional PDF knowledge assistant.

Answer the user's ORIGINAL QUESTION using ONLY
the supplied PDF context.

The RESOLVED QUESTION exists only to clarify
what the user means in a follow-up conversation.

IMPORTANT:
Conversation context may clarify references,
but ALL factual information in your answer
must come from the PDF context below.

Rules:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent facts.
4. Use information from multiple PDFs if needed.
5. Keep the answer concise but complete.
6. If the answer is not clearly supported, say exactly:

"I could not find that information in the PDFs."

ORIGINAL QUESTION:

{original_question}

RESOLVED QUESTION:

{resolved_question}

PDF CONTEXT:

{context}
"""

    return ask_model(
        prompt
    )


# ============================================================
# SUMMARY
# ============================================================

def generate_pdf_summary(
    selected_pdf="All Documents",
):
    scoped_chunks = get_scoped_chunks(
        selected_pdf
    )

    if not scoped_chunks:
        return None

    combined_text = "\n\n".join(
        f"""
DOCUMENT: {chunk["file"]}
PAGE: {chunk["page"]}

{chunk["text"]}
"""
        for chunk in scoped_chunks[:30]
    )

    prompt = f"""
You are JAXOVIQ, a professional document assistant.

Create a professional summary for:

{selected_pdf}

Use ONLY the supplied PDF content.
Do not invent information.

Include important:

- people
- organizations
- dates
- targets
- budgets
- goals
- services
- facts
- actions

PDF CONTENT:

{combined_text}
"""

    return ask_model(
        prompt
    )


# ============================================================
# ACTION ITEMS
# ============================================================

def generate_action_items(
    selected_pdf="All Documents",
):
    scoped_chunks = get_scoped_chunks(
        selected_pdf
    )

    if not scoped_chunks:
        return None

    context = "\n\n".join(
        f"""
SOURCE PDF: {chunk["file"]}
PAGE: {chunk["page"]}

TEXT:
{chunk["text"]}
"""
        for chunk in scoped_chunks[:30]
    )

    prompt = f"""
You are JAXOVIQ, a professional business document assistant.

Extract genuine action items, tasks, next steps,
deadlines, targets, or required actions.

Use ONLY the PDF content.
Do not invent information.

For each item use:

### ✅ Task: [task]

- **Owner:** [owner or Not specified]
- **Deadline:** [deadline or Not specified]
- **Priority:** [High / Medium / Low / Not specified]
- **Source:** [PDF filename] — page [number]
- **Evidence:** [short supporting evidence]

DOCUMENT SCOPE:

{selected_pdf}

PDF CONTENT:

{context}
"""

    return ask_model(
        prompt
    )


# ============================================================
# INSIGHTS
# ============================================================

def generate_document_insights(
    selected_pdf="All Documents",
):
    scoped_chunks = get_scoped_chunks(
        selected_pdf
    )

    if not scoped_chunks:
        return None

    context = "\n\n".join(
        f"""
SOURCE PDF: {chunk["file"]}
PAGE: {chunk["page"]}

TEXT:
{chunk["text"]}
"""
        for chunk in scoped_chunks[:30]
    )

    prompt = f"""
You are JAXOVIQ, a professional document intelligence assistant.

Analyze ONLY the supplied PDF content.

Create exactly these sections:

# Document Insights

## Key Dates

## People & Organizations

## Goals & Targets

## Important Numbers

## Main Themes

## Risks / Missing Information

## Evidence Sources

Do not invent information.

DOCUMENT SCOPE:

{selected_pdf}

PDF CONTENT:

{context}
"""

    return ask_model(
        prompt
    )


# ============================================================
# EVIDENCE VIEWER
# ============================================================

def clean_evidence_text(
    text,
    max_length=500,
):
    text = " ".join(
        text.split()
    )

    if len(text) > max_length:
        return (
            text[:max_length].rstrip()
            + "..."
        )

    return text


def build_evidence_list(
    selected_chunks,
):
    evidence_list = []
    seen = set()

    for item in selected_chunks:
        chunk = item["chunk"]

        key = (
            chunk["file"],
            chunk["page"],
            chunk["text"],
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        evidence_list.append(
            {
                "file": chunk["file"],
                "page": chunk["page"],
                "score": float(
                    item["score"]
                ),
                "text": clean_evidence_text(
                    chunk["text"]
                ),
            }
        )

    return evidence_list


def display_evidence(
    evidence_list,
):
    if not evidence_list:
        return

    with st.expander(
        "🔎 View Evidence",
        expanded=False,
    ):
        for number, evidence in enumerate(
            evidence_list,
            start=1,
        ):
            st.markdown(
                f"### Evidence {number}"
            )

            c1, c2, c3 = st.columns(
                3
            )

            with c1:
                st.caption(
                    "PDF"
                )

                st.write(
                    evidence["file"]
                )

            with c2:
                st.caption(
                    "Page"
                )

                st.write(
                    evidence["page"]
                )

            with c3:
                st.caption(
                    "Match"
                )

                st.write(
                    f'{evidence["score"]:.3f}'
                )

            st.info(
                evidence["text"]
            )

            if number < len(
                evidence_list
            ):
                st.divider()


# ============================================================
# MARKDOWN EXPORT
# ============================================================

def build_export_report(
    selected_pdf,
    summary,
    actions,
    insights,
):
    parts = [
        "# JAXOVIQ AI REPORT",
        "",
        f"**Scope:** {selected_pdf}",
        "",
    ]

    if summary:
        parts += [
            "## Document Summary",
            "",
            summary,
            "",
        ]

    if actions:
        parts += [
            "## Action Items",
            "",
            actions,
            "",
        ]

    if insights:
        parts += [
            "## Document Insights",
            "",
            insights,
            "",
        ]

    parts += [
        "---",
        "",
        "Generated by JAXOVIQ AI Assistant",
    ]

    return "\n".join(
        parts
    )


# ============================================================
# PROFESSIONAL PDF EXPORT
# ============================================================

def clean_markdown_for_pdf(text):
    if not text:
        return ""

    text = text.replace(
        "**",
        "",
    )

    text = text.replace(
        "__",
        "",
    )

    text = text.replace(
        "`",
        "",
    )

    return text


def add_text_to_pdf_story(
    story,
    text,
    styles,
):
    if not text:
        return

    text = clean_markdown_for_pdf(
        text
    )

    lines = text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            story.append(
                Spacer(
                    1,
                    3 * mm,
                )
            )
            continue

        safe_line = html.escape(
            line
        )

        if line.startswith("### "):
            story.append(
                Paragraph(
                    html.escape(
                        line[4:]
                    ),
                    styles["JAXHeading3"],
                )
            )

        elif line.startswith("## "):
            story.append(
                Paragraph(
                    html.escape(
                        line[3:]
                    ),
                    styles["JAXHeading2"],
                )
            )

        elif line.startswith("# "):
            story.append(
                Paragraph(
                    html.escape(
                        line[2:]
                    ),
                    styles["JAXHeading1"],
                )
            )

        elif (
            line.startswith("- ")
            or line.startswith("* ")
        ):
            bullet_text = html.escape(
                line[2:]
            )

            story.append(
                Paragraph(
                    f"• {bullet_text}",
                    styles["JAXBullet"],
                )
            )

        else:
            story.append(
                Paragraph(
                    safe_line,
                    styles["JAXBody"],
                )
            )


def build_professional_pdf(
    selected_pdf,
    summary,
    actions,
    insights,
    pdf_names,
):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="JAXOVIQ AI Report",
        author="JAXOVIQ AI Assistant",
    )

    sample_styles = getSampleStyleSheet()

    styles = {}

    styles["JAXTitle"] = ParagraphStyle(
        "JAXTitle",
        parent=sample_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=8 * mm,
        textColor=colors.HexColor(
            "#1F2937"
        ),
    )

    styles["JAXSubtitle"] = ParagraphStyle(
        "JAXSubtitle",
        parent=sample_styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=6 * mm,
        textColor=colors.HexColor(
            "#4B5563"
        ),
    )

    styles["JAXHeading1"] = ParagraphStyle(
        "JAXHeading1",
        parent=sample_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceBefore=6 * mm,
        spaceAfter=4 * mm,
        textColor=colors.HexColor(
            "#111827"
        ),
    )

    styles["JAXHeading2"] = ParagraphStyle(
        "JAXHeading2",
        parent=sample_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        textColor=colors.HexColor(
            "#1F2937"
        ),
    )

    styles["JAXHeading3"] = ParagraphStyle(
        "JAXHeading3",
        parent=sample_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        textColor=colors.HexColor(
            "#374151"
        ),
    )

    styles["JAXBody"] = ParagraphStyle(
        "JAXBody",
        parent=sample_styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        spaceAfter=2.5 * mm,
        textColor=colors.HexColor(
            "#1F2937"
        ),
    )

    styles["JAXBullet"] = ParagraphStyle(
        "JAXBullet",
        parent=styles["JAXBody"],
        leftIndent=5 * mm,
        firstLineIndent=-3 * mm,
        spaceAfter=2 * mm,
    )

    story = []

    story.append(
        Paragraph(
            "JAXOVIQ AI REPORT",
            styles["JAXTitle"],
        )
    )

    story.append(
        Paragraph(
            "AI-powered document intelligence report",
            styles["JAXSubtitle"],
        )
    )

    info_data = [
        [
            "Report Scope",
            selected_pdf,
        ],
        [
            "Documents",
            str(
                len(pdf_names)
            ),
        ],
        [
            "Generated By",
            "JAXOVIQ AI Assistant",
        ],
    ]

    info_table = Table(
        info_data,
        colWidths=[
            42 * mm,
            120 * mm,
        ],
    )

    info_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F3F4F6"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#111827"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#D1D5DB"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(
        info_table
    )

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    if pdf_names:
        story.append(
            Paragraph(
                "Included Documents",
                styles["JAXHeading2"],
            )
        )

        for pdf_name in pdf_names:
            story.append(
                Paragraph(
                    f"• {html.escape(pdf_name)}",
                    styles["JAXBullet"],
                )
            )

    if summary:
        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Document Summary",
                styles["JAXHeading1"],
            )
        )

        add_text_to_pdf_story(
            story,
            summary,
            styles,
        )

    if actions:
        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Action Items",
                styles["JAXHeading1"],
            )
        )

        add_text_to_pdf_story(
            story,
            actions,
            styles,
        )

    if insights:
        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "Document Insights",
                styles["JAXHeading1"],
            )
        )

        add_text_to_pdf_story(
            story,
            insights,
            styles,
        )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    story.append(
        Paragraph(
            "Generated by JAXOVIQ AI Assistant",
            styles["JAXSubtitle"],
        )
    )

    document.build(
        story
    )

    pdf_bytes = buffer.getvalue()

    buffer.close()

    return pdf_bytes


# ============================================================
# AUTO-LOAD SAVED KNOWLEDGE BASE
# ============================================================

if (
    not st.session_state.persistent_loaded
    and not st.session_state.knowledge_base_ready
):
    saved_data = (
        load_persistent_knowledge_base()
    )

    if saved_data:
        st.session_state.index = (
            saved_data["index"]
        )

        st.session_state.chunks = (
            saved_data["chunks"]
        )

        st.session_state.pdf_names = (
            saved_data["pdf_names"]
        )

        st.session_state.knowledge_base_ready = True

    st.session_state.persistent_loaded = True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title(
        "⚡ JAXOVIQ"
    )

    st.caption(
        "AI Knowledge Assistant"
    )

    st.divider()
 with st.expander("🔐 Admin Usage Stats"):
        admin_password = st.text_input(
            "Admin password",
            type="password",
            key="stats_admin_password",
        )

        if admin_password == st.secrets.get("ADMIN_PASSWORD", "") and admin_password:
            stats = load_stats()

            admin_mode = st.checkbox(
                "🧪 Admin Test Mode",
                value=st.session_state.admin_test_mode,
            )

            if admin_mode and not st.session_state.admin_test_mode:
                stats.setdefault("external_sessions", 0)
                stats.setdefault("admin_tests", 0)

                if st.session_state.visitor_counted:
                    stats["external_sessions"] = max(
                        0,
                        stats["external_sessions"] - 1,
                    )
                    stats["admin_tests"] += 1
                    st.session_state.visitor_counted = False

                save_stats(stats)
                st.session_state.admin_test_mode = True

            elif not admin_mode:
                st.session_state.admin_test_mode = False

            st.write("Traffic Sources")
            st.write("Meta:", stats.get("meta_visits", 0))
            st.write("LinkedIn:", stats.get("linkedin_visits", 0))
            st.write("Instagram:", stats.get("instagram_visits", 0))
            st.metric("App Opens", stats.get("app_opens", 0))
            st.metric("PDF Uploads", stats.get("pdf_uploads", 0))
            st.metric("Questions", stats.get("questions", 0))

        elif admin_password:
            st.error("Wrong password")

    if st.session_state.knowledge_base_ready:
        st.success(
            "● Knowledge Base Active"
        )

    else:
        st.warning(
            "● Knowledge Base Not Built"
        )

    st.subheader(
        "📚 Documents"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
       current_files = [f.name for f in uploaded_files]

       if st.session_state.get("last_uploaded_files") != current_files:
        stats = load_stats()
        stats["pdf_uploads"] += len(uploaded_files)
        save_stats(stats)
        st.session_state.last_uploaded_files = current_files
    document_options = [
        "All Documents"
    ]

    if st.session_state.pdf_names:
        document_options += (
            st.session_state.pdf_names
        )

    elif uploaded_files:
        document_options += [
            file.name
            for file in uploaded_files
        ]

    selected_pdf = st.selectbox(
        "PDF Scope",
        document_options,
    )

    if st.button(
        "⚡ Build Knowledge Base",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning(
                "Please upload at least one PDF."
            )

        else:
            with st.spinner(
                "Building JAXOVIQ Knowledge Base..."
            ):
                index, chunks = (
                    build_faiss_index(
                        uploaded_files
                    )
                )

            if (
                index is not None
                and chunks is not None
            ):
                pdf_names = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.index = (
                    index
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.pdf_names = (
                    pdf_names
                )

                st.session_state.messages = []

                st.session_state.knowledge_base_ready = True

                st.session_state.pdf_summary = None
                st.session_state.summary_scope = None

                st.session_state.action_items = None
                st.session_state.action_scope = None

                st.session_state.document_insights = None
                st.session_state.insights_scope = None

                st.session_state.export_report = None
                st.session_state.export_scope = None

                st.session_state.pdf_report_bytes = None
                st.session_state.pdf_report_scope = None

                st.session_state.suggested_questions = []
                st.session_state.suggestions_scope = None
                st.session_state.pending_question = None

                save_persistent_knowledge_base(
                    index,
                    chunks,
                    pdf_names,
                )

                with st.spinner(
                    "Generating smart questions..."
                ):
                    st.session_state.suggested_questions = (
                        generate_suggested_questions(
                            "All Documents"
                        )
                    )

                    st.session_state.suggestions_scope = (
                        "All Documents"
                    )

                st.success(
                    "Knowledge base ready and saved."
                )

                st.rerun()

    if st.session_state.pdf_names:
        st.divider()

        st.caption(
            "Loaded PDFs"
        )

        for pdf_name in st.session_state.pdf_names:
            st.write(
                f"📄 {pdf_name}"
            )

        if (
            INDEX_FILE.exists()
            and CHUNKS_FILE.exists()
        ):
            st.caption(
                "💾 Saved locally"
            )

    st.divider()

    col_a, col_b = st.columns(
        2
    )

    with col_a:
        if st.button(
            "🗑 Clear Chat",
            use_container_width=True,
        ):
            st.session_state.messages = []

            st.rerun()

    with col_b:
        if st.button(
            "🔄 Reset All",
            use_container_width=True,
        ):
            delete_persistent_knowledge_base()

            for key, value in defaults.items():
                st.session_state[key] = value

            st.session_state.persistent_loaded = True

            st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="jaxoviq-hero">'
    '<div class="jaxoviq-title">⚡ JAXOVIQ AI Assistant</div>'
    '<div class="jaxoviq-subtitle">Search, analyze and understand your documents with grounded AI.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# TOP DASHBOARD
# ============================================================

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)

with metric1:
    st.metric(
        "PDFs",
        len(
            st.session_state.pdf_names
        ),
    )

with metric2:
    st.metric(
        "Knowledge Chunks",
        len(
            st.session_state.chunks
        ),
    )

with metric3:
    st.metric(
        "Chat Messages",
        len(
            st.session_state.messages
        ),
    )

with metric4:
    status_text = (
        "Active"
        if st.session_state.knowledge_base_ready
        else "Offline"
    )

    st.metric(
        "Knowledge Base",
        status_text,
    )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if st.session_state.knowledge_base_ready:
    st.success(
        f"✅ Knowledge Base Active · Scope: {selected_pdf}"
    )

    if (
        INDEX_FILE.exists()
        and CHUNKS_FILE.exists()
    ):
        st.caption(
            "💾 Persistent Knowledge Base: Saved"
        )

else:
    st.info(
        "Upload PDFs from the sidebar and click "
        "**Build Knowledge Base**."
    )


# ============================================================
# DOCUMENT STATISTICS
# ============================================================

if st.session_state.knowledge_base_ready:

    stats = get_document_stats()

    with st.expander(
        "📊 Document Statistics",
        expanded=True,
    ):
        st.caption(
            f"Current scope: {selected_pdf}"
        )

        for pdf_name in st.session_state.pdf_names:

            data = stats.get(
                pdf_name,
                {
                    "pages": 0,
                    "chunks": 0,
                },
            )

            st.markdown(
                f"### 📄 {pdf_name}"
            )

            stat1, stat2, stat3 = st.columns(
                3
            )

            with stat1:
                st.metric(
                    "Pages Found",
                    data["pages"],
                )

            with stat2:
                st.metric(
                    "Chunks",
                    data["chunks"],
                )

            with stat3:
                if selected_pdf == pdf_name:
                    scope_status = "Selected"

                elif selected_pdf == "All Documents":
                    scope_status = "Included"

                else:
                    scope_status = "Not Selected"

                st.metric(
                    "Scope",
                    scope_status,
                )

            st.divider()


# ============================================================
# SMART SUGGESTED QUESTIONS
# ============================================================

if st.session_state.knowledge_base_ready:

    st.subheader(
        "💡 Suggested Questions"
    )

    st.caption(
        "Click a question and JAXOVIQ will search your PDFs automatically."
    )

    if (
        st.session_state.suggestions_scope
        != selected_pdf
    ):
        if st.button(
            "✨ Generate Questions for Current Scope",
            use_container_width=True,
        ):
            with st.spinner(
                "Generating smart questions..."
            ):
                st.session_state.suggested_questions = (
                    generate_suggested_questions(
                        selected_pdf
                    )
                )

                st.session_state.suggestions_scope = (
                    selected_pdf
                )

            st.rerun()

    if st.session_state.suggested_questions:

        for index, suggested_question in enumerate(
            st.session_state.suggested_questions,
            start=1,
        ):
            safe_scope = selected_pdf.replace(
                " ",
                "_",
            )

            if st.button(
                f"💬 {suggested_question}",
                key=f"suggested_question_{safe_scope}_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = (
                    suggested_question
                )

                st.rerun()

    else:
        st.info(
            "No suggested questions available yet."
        )

    st.divider()


# ============================================================
# DOCUMENT TOOLS
# ============================================================

if st.session_state.knowledge_base_ready:

    st.subheader(
        "🧰 Document Tools"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:
        summarize_clicked = st.button(
            "📄 Summarize",
            use_container_width=True,
        )

    with col2:
        actions_clicked = st.button(
            "✅ Action Items",
            use_container_width=True,
        )

    with col3:
        insights_clicked = st.button(
            "📊 Insights",
            use_container_width=True,
        )

    with col4:
        export_clicked = st.button(
            "📥 Export Report",
            use_container_width=True,
        )

    if summarize_clicked:
        with st.spinner(
            "Creating document summary..."
        ):
            st.session_state.pdf_summary = (
                generate_pdf_summary(
                    selected_pdf
                )
            )

            st.session_state.summary_scope = (
                selected_pdf
            )

    if actions_clicked:
        with st.spinner(
            "Finding action items..."
        ):
            st.session_state.action_items = (
                generate_action_items(
                    selected_pdf
                )
            )

            st.session_state.action_scope = (
                selected_pdf
            )

    if insights_clicked:
        with st.spinner(
            "Analyzing document insights..."
        ):
            st.session_state.document_insights = (
                generate_document_insights(
                    selected_pdf
                )
            )

            st.session_state.insights_scope = (
                selected_pdf
            )

    if export_clicked:
        with st.spinner(
            "Preparing professional JAXOVIQ report..."
        ):

            summary = (
                st.session_state.pdf_summary
                if (
                    st.session_state.summary_scope
                    == selected_pdf
                )
                else None
            )

            if not summary:
                summary = generate_pdf_summary(
                    selected_pdf
                )

                st.session_state.pdf_summary = (
                    summary
                )

                st.session_state.summary_scope = (
                    selected_pdf
                )

            actions = (
                st.session_state.action_items
                if (
                    st.session_state.action_scope
                    == selected_pdf
                )
                else None
            )

            if not actions:
                actions = generate_action_items(
                    selected_pdf
                )

                st.session_state.action_items = (
                    actions
                )

                st.session_state.action_scope = (
                    selected_pdf
                )

            insights = (
                st.session_state.document_insights
                if (
                    st.session_state.insights_scope
                    == selected_pdf
                )
                else None
            )

            if not insights:
                insights = generate_document_insights(
                    selected_pdf
                )

                st.session_state.document_insights = (
                    insights
                )

                st.session_state.insights_scope = (
                    selected_pdf
                )

            st.session_state.export_report = (
                build_export_report(
                    selected_pdf,
                    summary,
                    actions,
                    insights,
                )
            )

            st.session_state.export_scope = (
                selected_pdf
            )

            try:
                st.session_state.pdf_report_bytes = (
                    build_professional_pdf(
                        selected_pdf,
                        summary,
                        actions,
                        insights,
                        st.session_state.pdf_names,
                    )
                )

                st.session_state.pdf_report_scope = (
                    selected_pdf
                )

            except Exception as e:
                st.session_state.pdf_report_bytes = None
                st.session_state.pdf_report_scope = None

                st.error(
                    f"Could not create PDF report: {e}"
                )

    if (
        st.session_state.pdf_summary
        and st.session_state.summary_scope
        == selected_pdf
    ):
        with st.expander(
            "📄 Document Summary",
            expanded=True,
        ):
            st.markdown(
                st.session_state.pdf_summary
            )

    if (
        st.session_state.action_items
        and st.session_state.action_scope
        == selected_pdf
    ):
        with st.expander(
            "✅ Action Items",
            expanded=True,
        ):
            st.markdown(
                st.session_state.action_items
            )

    if (
        st.session_state.document_insights
        and st.session_state.insights_scope
        == selected_pdf
    ):
        with st.expander(
            "📊 Document Insights",
            expanded=True,
        ):
            st.markdown(
                st.session_state.document_insights
            )

    if (
        st.session_state.export_report
        and st.session_state.export_scope
        == selected_pdf
    ):
        st.success(
            "📥 JAXOVIQ report is ready."
        )

        download_col1, download_col2 = st.columns(
            2
        )

        with download_col1:
            st.download_button(
                label="⬇️ Download Markdown Report",
                data=st.session_state.export_report,
                file_name="JAXOVIQ_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with download_col2:
            if (
                st.session_state.pdf_report_bytes
                and st.session_state.pdf_report_scope
                == selected_pdf
            ):
                st.download_button(
                    label="📄 Download Professional PDF",
                    data=st.session_state.pdf_report_bytes,
                    file_name="JAXOVIQ_AI_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.divider()


# ============================================================
# CHAT
# ============================================================

st.subheader(
    "💬 Ask JAXOVIQ"
)


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message.get("sources"):
            with st.expander(
                "📚 Sources"
            ):
                for source in message["sources"]:
                    st.write(
                        source
                    )

        if message.get("evidence"):
            display_evidence(
                message["evidence"]
            )

        if message.get("confidence"):
            st.caption(
                f'Confidence: {message["confidence"]}'
            )

        if message.get("score") is not None:
            st.caption(
                f'Similarity score: '
                f'{message["score"]:.3f}'
            )

# ---------------- FEEDBACK ----------------
with st.expander("💬 Give Feedback"):
    feedback_rating = st.radio(
        "Was JAXOVIQ helpful?",
        ["👍 Yes", "😐 Partly", "👎 No"],
        horizontal=True
    )

    feedback_comment = st.text_area(
        "Tell us what we can improve:",
        placeholder="Write your feedback here..."
    )

    if st.button("Submit Feedback"):
        from datetime import datetime

        with open("feedback.txt", "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now()} | {feedback_rating} | {feedback_comment}\n"
            )

        st.success("Thank you! Your feedback has been submitted. 🙏")



typed_question = st.chat_input(
    "Ask a question about your PDFs..."
)

question = None

if st.session_state.pending_question:
    question = st.session_state.pending_question

    st.session_state.pending_question = None

elif typed_question:
    question = typed_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    stats = load_stats()
    stats["questions"] += 1
    save_stats(stats)

    if not st.session_state.knowledge_base_ready:
        st.warning(
            "Please build the knowledge base first."
        )
    else:
        with st.spinner(
            "Understanding your question..."
        ):
            resolved_question = (
                resolve_followup_question(
                    question
                )
            )
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):
            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching your documents..."
            ):

                candidates, best_score = (
                    retrieve_chunks(
                        resolved_question,
                        selected_pdf,
                    )
                )

                confidence = (
                    get_confidence_label(
                        best_score
                    )
                )

                if (
                    not candidates
                    or best_score < MIN_SCORE
                ):

                    answer = (
                        "I could not find that "
                        "information in the PDFs."
                    )

                    st.markdown(
                        answer
                    )

                    st.caption(
                        f"Confidence: {confidence}"
                    )

                    st.caption(
                        f"Similarity score: "
                        f"{best_score:.3f}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                            "evidence": [],
                            "score": best_score,
                            "confidence": confidence,
                        }
                    )

                else:

                    selected = rerank_chunks(
                        resolved_question,
                        candidates,
                    )

                    answer = generate_answer(
                        question,
                        resolved_question,
                        selected,
                    )

                    if not answer:
                        answer = (
                            "I could not generate an answer."
                        )

                    st.markdown(
                        answer
                    )

                    source_list = []
                    seen_sources = set()

                    for item in selected:

                        chunk = item["chunk"]

                        source_key = (
                            chunk["file"],
                            chunk["page"],
                        )

                        if source_key in seen_sources:
                            continue

                        seen_sources.add(
                            source_key
                        )

                        source_list.append(
                            f'📄 {chunk["file"]} '
                            f'— page {chunk["page"]}'
                        )

                    if source_list:
                        with st.expander(
                            "📚 Sources"
                        ):
                            for source in source_list:
                                st.write(
                                    source
                                )

                    evidence_list = (
                        build_evidence_list(
                            selected
                        )
                    )

                    display_evidence(
                        evidence_list
                    )

                    st.caption(
                        f"Confidence: {confidence}"
                    )

                    st.caption(
                        f"Similarity score: "
                        f"{best_score:.3f}"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_list,
                            "evidence": evidence_list,
                            "score": best_score,
                            "confidence": confidence,
                        }
                    )
