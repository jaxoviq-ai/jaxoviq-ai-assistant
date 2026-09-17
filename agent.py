import io
import json
import time
import html
import hashlib
import os
from datetime import datetime, timezone
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
# Client Demo Edition + Voice Input
# ============================================================

client = OpenAI()

MIN_SCORE = 0.30
MEDIUM_SCORE = 0.45
HIGH_SCORE = 0.60

TOP_K = 8
RERANK_TOP_N = 4

# Keep only evidence close enough to the strongest retrieved result.
# This reduces weak / unrelated source clutter in client demos.
SOURCE_SCORE_GAP = 0.12

EMBEDDING_MODEL = "text-embedding-3-small"
AI_MODEL = "gpt-5.6-luna"
VOICE_MODEL = "gpt-4o-mini-transcribe"


# ============================================================
# PERSISTENT KNOWLEDGE BASE PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "jaxoviq_data"

INDEX_FILE = DATA_DIR / "knowledge.index"
CHUNKS_FILE = DATA_DIR / "chunks.json"
META_FILE = DATA_DIR / "meta.json"
ANALYTICS_FILE = DATA_DIR / "analytics.json"
FEEDBACK_FILE = DATA_DIR / "feedback.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="JAXOVIQ AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.18);
    }

    .jaxoviq-hero {
        padding: 1.6rem 1.7rem;
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 20px;
        margin-bottom: 1rem;
        background: linear-gradient(
            135deg,
            rgba(127, 90, 240, 0.12),
            rgba(0, 180, 216, 0.08)
        );
    }

    .jaxoviq-eyebrow {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 0.35rem;
    }

    .jaxoviq-title {
        font-size: 2.45rem;
        font-weight: 850;
        margin: 0;
        line-height: 1.05;
    }

    .jaxoviq-subtitle {
        font-size: 1.03rem;
        opacity: 0.82;
        margin-top: 0.55rem;
        max-width: 900px;
    }

    .jaxoviq-usecase {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        min-height: 120px;
    }

    .jaxoviq-usecase-title {
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .jaxoviq-usecase-text {
        opacity: 0.78;
        font-size: 0.93rem;
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

    .demo-note {
        padding: 0.8rem 1rem;
        border-left: 4px solid rgba(127, 90, 240, 0.75);
        background: rgba(127, 90, 240, 0.07);
        border-radius: 8px;
        margin: 0.65rem 0 1rem 0;
        font-size: 0.93rem;
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

    # Voice input
    "last_voice_hash": None,
    "last_voice_text": None,
    "last_submitted_voice_hash": None,

    # Business analysis tools
    "comparison_result": None,
    "comparison_pair": None,
    "risk_check_result": None,
    "risk_check_scope": None,
    "deadline_result": None,
    "deadline_scope": None,

    # Executive intelligence
    "executive_brief": None,
    "executive_brief_scope": None,
    "document_audit": None,
    "document_audit_scope": None,

    # Extreme intelligence
    "conflict_scan": None,
    "conflict_scan_scope": None,
    "obligation_register": None,
    "obligation_scope": None,
    "decision_memo": None,
    "decision_scope": None,
    "full_scan_last_scope": None,

    # Admin / analytics
    "admin_authenticated": False,
    "admin_test_mode": False,
    "analytics_session_tracked": False,

    "persistent_loaded": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PRODUCT ANALYTICS + FEEDBACK
# ============================================================

def _analytics_defaults():
    return {
        "app_opens": 0,
        "pdf_uploads": 0,
        "questions": 0,
        "analysis_runs": 0,
        "reports_exported": 0,
        "feedback_count": 0,
        "traffic_sources": {
            "Meta": 0,
            "Instagram": 0,
            "LinkedIn": 0,
            "Google": 0,
            "Direct": 0,
            "Other": 0,
        },
        "last_updated_utc": None,
    }


def load_analytics():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    defaults_data = _analytics_defaults()

    if not ANALYTICS_FILE.exists():
        return defaults_data

    try:
        saved = json.loads(ANALYTICS_FILE.read_text(encoding="utf-8"))
        for key, value in defaults_data.items():
            if key not in saved:
                saved[key] = value
        if not isinstance(saved.get("traffic_sources"), dict):
            saved["traffic_sources"] = defaults_data["traffic_sources"]
        for source, count in defaults_data["traffic_sources"].items():
            saved["traffic_sources"].setdefault(source, count)
        return saved
    except Exception:
        return defaults_data


def save_analytics(data):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        ANALYTICS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def normalize_traffic_source(raw_source):
    value = (raw_source or "").strip().lower()
    if not value:
        return "Direct"
    if "instagram" in value or value in {"ig", "insta"}:
        return "Instagram"
    if "linkedin" in value:
        return "LinkedIn"
    if "meta" in value or "facebook" in value or value == "fb":
        return "Meta"
    if "google" in value:
        return "Google"
    return "Other"


def current_traffic_source():
    try:
        raw = st.query_params.get("utm_source") or st.query_params.get("source")
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        return normalize_traffic_source(raw)
    except Exception:
        return "Direct"


def track_event(event_name, amount=1, traffic_source=None):
    if st.session_state.get("admin_test_mode", False):
        return

    data = load_analytics()
    if event_name in data and isinstance(data[event_name], int):
        data[event_name] += int(amount)

    if traffic_source:
        source = normalize_traffic_source(traffic_source)
        data["traffic_sources"][source] = (
            data["traffic_sources"].get(source, 0) + int(amount)
        )

    save_analytics(data)


def save_feedback(rating, message):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    if FEEDBACK_FILE.exists():
        try:
            entries = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            entries = []

    entries.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "message": message.strip(),
        }
    )
    FEEDBACK_FILE.write_text(
        json.dumps(entries[-500:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    track_event("feedback_count")


# Count one open per browser session, not every Streamlit rerun.
if not st.session_state.analytics_session_tracked:
    source = current_traffic_source()
    track_event("app_opens", traffic_source=source)
    st.session_state.analytics_session_tracked = True


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
# VOICE INPUT / SPEECH-TO-TEXT
# ============================================================

def transcribe_voice(audio_file):
    """Convert Streamlit microphone audio into text for JAXOVIQ Q&A."""
    if audio_file is None:
        return None

    try:
        audio_bytes = audio_file.getvalue()
        if not audio_bytes:
            return None

        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = "jaxoviq_voice.wav"

        transcript = client.audio.transcriptions.create(
            model=VOICE_MODEL,
            file=audio_buffer,
        )

        text = getattr(transcript, "text", "")
        return text.strip() if text else None

    except Exception as e:
        st.error(f"Voice transcription error: {e}")
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

        speaker = "USER" if role == "user" else "JAXOVIQ"
        lines.append(f"{speaker}: {content}")

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

def save_persistent_knowledge_base(index, chunks, pdf_names):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            index,
            str(INDEX_FILE),
        )

        with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        with open(META_FILE, "w", encoding="utf-8") as file:
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
        index = faiss.read_index(str(INDEX_FILE))

        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        with open(META_FILE, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        pdf_names = metadata.get("pdf_names", [])
        saved_model = metadata.get("embedding_model")

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
    for file_path in [INDEX_FILE, CHUNKS_FILE, META_FILE]:
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
        reader = PdfReader(io.BytesIO(pdf_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
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
        st.error(f"Could not read {uploaded_file.name}: {e}")
        return []


def chunk_pages(pages, chunk_size=1000, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

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
        pages = read_uploaded_pdf(uploaded_file)
        all_chunks.extend(chunk_pages(pages))

    if not all_chunks:
        return None, None

    embeddings = []

    progress = st.progress(0)
    status = st.empty()

    total = len(all_chunks)

    for i, chunk in enumerate(all_chunks, start=1):
        status.write(f"Creating embeddings... {i}/{total}")

        embedding = get_embedding(chunk["text"])

        if embedding is None:
            progress.empty()
            status.empty()
            return None, None

        embeddings.append(embedding)
        progress.progress(i / total)

    vectors = np.array(
        embeddings,
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

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
        stats[file_name]["pages"].add(page_number)

    final_stats = {}

    for file_name, data in stats.items():
        final_stats[file_name] = {
            "chunks": data["chunks"],
            "pages": len(data["pages"]),
        }

    return final_stats


# ============================================================
# SMART SUGGESTED QUESTIONS
# ============================================================

def generate_suggested_questions(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)

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
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        questions = data.get("questions", [])

        final_questions = []

        for question in questions:
            if isinstance(question, str) and question.strip():
                final_questions.append(question.strip())

            if len(final_questions) >= 5:
                break

        return final_questions

    except Exception:
        return []


# ============================================================
# SMART RETRIEVAL QUERY EXPANSION
# ============================================================

def expand_retrieval_query(question):
    """Add common business-document synonyms without changing user intent.

    This improves semantic retrieval when a user says, for example,
    "holidays" while the PDF uses "vacation days" or "annual leave".
    """
    if not question:
        return question

    q = question.strip()
    q_lower = q.lower()

    synonym_groups = [
        (["holiday", "holidays"], "vacation days annual leave paid leave leave entitlement"),
        (["vacation", "annual leave"], "holidays paid leave leave entitlement"),
        (["sick", "illness", "medical leave"], "sick leave sickness absence medical certificate"),
        (["salary", "wage", "pay"], "salary compensation wages remuneration payment"),
        (["notice", "termination", "quit", "resign"], "notice period termination resignation employment end"),
        (["renew", "renewal", "expire", "expiry"], "renewal expiry expiration end date extension"),
        (["remote", "work from home", "home office"], "remote work work from home home office hybrid work"),
        (["expense", "reimbursement", "claim"], "expense reimbursement claim receipt approval business expense"),
        (["hours", "working time", "schedule"], "working hours schedule core hours weekly hours"),
        (["probation", "trial period"], "probation probationary period trial period"),
        (["benefit", "benefits"], "employee benefits entitlement allowance perks"),
        (["deadline", "due date"], "deadline due date submission date time limit notice period"),
    ]

    extras = []
    for triggers, expansion in synonym_groups:
        if any(trigger in q_lower for trigger in triggers):
            extras.append(expansion)

    if extras:
        return q + "\nRelevant equivalent terms: " + " ; ".join(dict.fromkeys(extras))

    return q



# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(question, selected_pdf="All Documents"):
    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return [], 0.0

    retrieval_query = expand_retrieval_query(question)
    query_embedding = get_embedding(retrieval_query)

    if query_embedding is None:
        return [], 0.0

    query_vector = np.array(
        [query_embedding],
        dtype="float32",
    )

    faiss.normalize_L2(query_vector)

    chunks = st.session_state.chunks
    index = st.session_state.index

    if selected_pdf == "All Documents":
        k = min(TOP_K, len(chunks))
    else:
        # Search enough rows to find chunks from the selected file.
        k = len(chunks)

    scores, indices = index.search(
        query_vector,
        k,
    )

    candidates = []
    filtered_scores = []

    for score, chunk_index in zip(scores[0], indices[0]):
        score = float(score)

        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        if selected_pdf != "All Documents":
            if chunk["file"] != selected_pdf:
                continue

        filtered_scores.append(score)

        if score >= MIN_SCORE:
            candidates.append(
                {
                    "chunk": chunk,
                    "score": score,
                }
            )

    best_score = max(filtered_scores) if filtered_scores else 0.0

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

def rerank_chunks(question, candidates):
    if len(candidates) <= 1:
        return candidates

    candidate_text = ""

    for number, item in enumerate(candidates, start=1):
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
        result = ask_model(prompt)

        if not result:
            return candidates[:RERANK_TOP_N]

        cleaned = result.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "")
            cleaned = cleaned.replace("```", "")
            cleaned = cleaned.strip()

        data = json.loads(cleaned)
        ranking = data.get("ranking", [])

        reranked = []

        for number in ranking:
            if not isinstance(number, int):
                continue

            position = number - 1

            if 0 <= position < len(candidates):
                item = candidates[position]

                if item not in reranked:
                    reranked.append(item)

            if len(reranked) >= RERANK_TOP_N:
                break

        if reranked:
            return reranked

    except Exception:
        pass

    return candidates[:RERANK_TOP_N]


def filter_selected_evidence(selected, best_score):
    """
    Keep the strongest evidence and remove weak unrelated chunks.
    This makes client-facing sources cleaner without changing retrieval.
    """
    if not selected:
        return []

    threshold = max(MIN_SCORE, best_score - SOURCE_SCORE_GAP)

    filtered = [
        item
        for item in selected
        if float(item["score"]) >= threshold
    ]

    if filtered:
        return filtered

    return selected[:1]


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

    context = "\n\n".join(context_parts)

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
6. Prefer the most directly relevant source first.
7. If the answer is not clearly supported, say exactly:

"I could not find that information in the PDFs."

ORIGINAL QUESTION:

{original_question}

RESOLVED QUESTION:

{resolved_question}

PDF CONTEXT:

{context}
"""

    return ask_model(prompt)


# ============================================================
# SUMMARY
# ============================================================

def generate_pdf_summary(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)

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

    return ask_model(prompt)


# ============================================================
# ACTION ITEMS
# ============================================================

def generate_action_items(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)

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

    return ask_model(prompt)


# ============================================================
# INSIGHTS
# ============================================================

def generate_document_insights(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)

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

    return ask_model(prompt)


# ============================================================
# BUSINESS ANALYSIS TOOLS
# ============================================================

def _document_context_for_analysis(pdf_name, max_chunks=35):
    """Build page-labelled context for one PDF from the existing KB."""
    chunks = [
        chunk
        for chunk in st.session_state.chunks
        if chunk["file"] == pdf_name
    ]

    if not chunks:
        return ""

    return "\n\n".join(
        f"DOCUMENT: {chunk['file']}\nPAGE: {chunk['page']}\n\n{chunk['text']}"
        for chunk in chunks[:max_chunks]
    )


def compare_documents(pdf_a, pdf_b):
    if not pdf_a or not pdf_b or pdf_a == pdf_b:
        return None

    context_a = _document_context_for_analysis(pdf_a)
    context_b = _document_context_for_analysis(pdf_b)

    if not context_a or not context_b:
        return None

    prompt = f"""
You are JAXOVIQ, a professional business document comparison assistant.

Compare DOCUMENT A and DOCUMENT B using ONLY the supplied content.
Do not use outside knowledge and do not invent facts.

Create these sections:

# Document Comparison

## Executive Summary
Give a short factual overview of the most important differences.

## Added or New Items
List material items that appear in B but not A.

## Removed Items
List material items that appear in A but not B.

## Changed Terms
For every meaningful change, show:
- Topic
- Document A value/wording
- Document B value/wording
- Why the change matters operationally
- Evidence with filename and page number

## Dates, Numbers and Obligations Changed
Highlight changed dates, amounts, notice periods, targets, benefits,
responsibilities, deadlines or other measurable terms.

## Unclear or Unverified Differences
State anything that cannot be compared confidently from the supplied text.

Rules:
1. Be precise and neutral.
2. If something is absent from the supplied text, say "Not found in supplied text".
3. Never claim a legal conclusion.
4. Cite filename and page for important findings.

DOCUMENT A: {pdf_a}

{context_a}

DOCUMENT B: {pdf_b}

{context_b}
"""

    return ask_model(prompt)


def run_risk_check(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)

    if not scoped_chunks:
        return None

    context = "\n\n".join(
        f"SOURCE PDF: {chunk['file']}\nPAGE: {chunk['page']}\n\n{chunk['text']}"
        for chunk in scoped_chunks[:40]
    )

    prompt = f"""
You are JAXOVIQ, a business document review assistant.

Review ONLY the supplied document text for practical risks, ambiguities,
missing information and obligations. This is a document review, NOT legal advice.
Do not invent requirements that are not stated in the documents.

Create these sections:

# Contract / Policy Check

## High Attention Items
Material clauses or wording that may need human review.

## Ambiguous or Unclear Wording
Terms that are vague, incomplete, internally inconsistent or difficult to apply.

## Important Obligations
Who must do what, and when.

## Missing or Not Found
Only list common document information when its absence is genuinely relevant,
and label it clearly as "Not found in supplied text" rather than saying it is legally required.

## Dates and Renewal / Notice Terms
Extract relevant effective dates, expiry dates, notice periods, renewals and deadlines.

## Recommended Human Review
Identify points that a manager, HR professional, accountant or qualified legal professional
may want to verify, without giving legal advice.

For each important finding include:
- Finding
- Attention level: High / Medium / Low
- Source: filename and page
- Short evidence

DOCUMENT SCOPE: {selected_pdf}

PDF CONTENT:

{context}
"""

    return ask_model(prompt)


def extract_deadlines_and_key_terms(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)

    if not scoped_chunks:
        return None

    context = "\n\n".join(
        f"SOURCE PDF: {chunk['file']}\nPAGE: {chunk['page']}\n\n{chunk['text']}"
        for chunk in scoped_chunks[:40]
    )

    prompt = f"""
You are JAXOVIQ, a business document extraction assistant.

Use ONLY the supplied PDF content. Do not calculate or invent dates or terms
that are not supported by the text.

Create these sections:

# Deadlines & Key Terms

## Critical Dates
Use a Markdown table with columns:
| Date / Period | Event or Obligation | Responsible Party | Source |

## Notice, Renewal and Expiry
Use a Markdown table with columns:
| Term | Value | Source |

## Money and Payment Terms
Use a Markdown table with columns:
| Item | Amount / Timing | Source |

## Employment / Policy Terms
Include leave, probation, working time, benefits, approvals or other policy terms when present.
Use a Markdown table:
| Topic | Key Term | Source |

## Other Important Numbers
Use a Markdown table:
| Item | Value | Source |

## Missing Dates or Owners
List actions or obligations that appear to lack a clear date or responsible person.

Always include filename and page number in Source.
If a section has no supported information, write "No supported information found."

DOCUMENT SCOPE: {selected_pdf}

PDF CONTENT:

{context}
"""

    return ask_model(prompt)


# ============================================================
# EVIDENCE VIEWER
# ============================================================

# ============================================================
# EXECUTIVE INTELLIGENCE
# ============================================================

def generate_executive_brief(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)
    if not scoped_chunks:
        return "No document content is available for this scope."

    context = "\n\n".join(
        f'DOCUMENT: {c["file"]}\nPAGE: {c["page"]}\n{c["text"]}'
        for c in scoped_chunks[:45]
    )

    prompt = f"""
You are JAXOVIQ Executive Intelligence.
Create a decision-ready executive brief using ONLY the supplied PDF content.
Never invent facts. Every important fact must include its source in the form
[Document name, p.X].

Return these sections:

## Executive Snapshot
5 concise bullets covering the most decision-relevant facts.

## Key Numbers & Commitments
Important amounts, dates, targets, notice periods, limits or obligations.

## Risks / Watch Items
Only evidence-based concerns, ambiguity, missing clarity or operational risk.
If none are supported, say so.

## Immediate Actions
Prioritized actions clearly supported by the documents.

## Questions Leadership Should Ask
3-5 high-value follow-up questions that the documents leave open.

SCOPE: {selected_pdf}

PDF CONTENT:
{context}
"""
    return ask_model(prompt) or "Could not generate the executive brief."


def run_document_audit(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)
    if not scoped_chunks:
        return "No document content is available for this scope."

    context = "\n\n".join(
        f'DOCUMENT: {c["file"]}\nPAGE: {c["page"]}\n{c["text"]}'
        for c in scoped_chunks[:45]
    )

    prompt = f"""
You are JAXOVIQ Document Audit AI.
Audit the supplied business documents using ONLY their content.
Do not give legal advice and do not invent missing facts.
Reference evidence with [Document name, p.X].

Produce:

## Document Health
- Purpose and apparent audience
- Version/effective-date clarity
- Ownership/responsibility clarity
- Internal consistency observations

## Missing or Unclear Information
List only items that are genuinely absent, ambiguous, inconsistent or hard to act on.

## Operational Risk Flags
Rate each supported issue as High / Medium / Low and explain why in one sentence.

## Improvement Opportunities
Specific edits or additions that would make the document easier for staff or managers to use.

## Audit Summary
A short non-legal summary of the most important findings.

SCOPE: {selected_pdf}

PDF CONTENT:
{context}
"""
    return ask_model(prompt) or "Could not complete the document audit."


def generate_obligation_register(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)
    if not scoped_chunks:
        return "No document content is available for this scope."

    context = "\n\n".join(
        f'DOCUMENT: {c["file"]}\nPAGE: {c["page"]}\n{c["text"]}'
        for c in scoped_chunks[:50]
    )

    prompt = f"""
You are JAXOVIQ Obligation Intelligence.
Use ONLY the supplied PDF content. Do not invent duties, deadlines or owners.
Extract every material obligation, required action, approval, restriction or recurring duty.

Return:

# Obligation Register

Use a Markdown table:
| Priority | Owner / Responsible Party | Obligation / Required Action | Trigger / Deadline | Consequence or Dependency Stated in Document | Source |

Rules:
- Priority can be High / Medium / Low based only on operational importance evident in the text.
- If owner, trigger, deadline or consequence is not stated, write "Not specified".
- Source must include filename and page.
- Do not give legal advice.

## Unassigned / Unclear Obligations
List obligations whose owner, deadline or process is unclear.

SCOPE: {selected_pdf}

PDF CONTENT:
{context}
"""
    return ask_model(prompt) or "Could not generate the obligation register."


def generate_decision_memo(selected_pdf="All Documents"):
    scoped_chunks = get_scoped_chunks(selected_pdf)
    if not scoped_chunks:
        return "No document content is available for this scope."

    context = "\n\n".join(
        f'DOCUMENT: {c["file"]}\nPAGE: {c["page"]}\n{c["text"]}'
        for c in scoped_chunks[:50]
    )

    prompt = f"""
You are JAXOVIQ Decision Intelligence.
Create a management-ready memo using ONLY the supplied PDF content.
Never invent facts. Cite important evidence as [Document name, p.X].

Return exactly:

# Decision Memo

## Situation
What the documents establish in 3-5 concise bullets.

## Decisions / Approvals Needed
Only decisions that are genuinely suggested by gaps, obligations, deadlines or choices in the supplied documents.
If none are evident, say "No explicit decision requirement found."

## Evidence That Matters
A concise table:
| Evidence | Why It Matters | Source |

## Risks of Inaction
Only evidence-based operational consequences or unresolved issues. Do not invent consequences.

## Recommended Next Actions
Prioritized practical actions grounded in the documents. This is operational guidance, not legal advice.

## Open Questions
Questions the documents do not resolve.

SCOPE: {selected_pdf}

PDF CONTENT:
{context}
"""
    return ask_model(prompt) or "Could not generate the decision memo."


def scan_cross_document_conflicts():
    if len(st.session_state.pdf_names) < 2:
        return "Upload at least 2 PDFs to run a cross-document conflict scan."

    context_parts = []
    for pdf_name in st.session_state.pdf_names[:8]:
        pdf_context = _document_context_for_analysis(pdf_name, max_chunks=22)
        if pdf_context:
            context_parts.append(pdf_context)

    if len(context_parts) < 2:
        return "Not enough readable document content is available for a conflict scan."

    context = "\n\n===== NEXT DOCUMENT =====\n\n".join(context_parts)

    prompt = f"""
You are JAXOVIQ Cross-Document Conflict Intelligence.
Analyze ONLY the supplied PDF content and detect contradictions, mismatched terms,
duplicate rules with different values, inconsistent dates, conflicting responsibilities,
or policies that could create operational confusion.

Do NOT treat different topics as conflicts. Do NOT invent conflicts.

Return:

# Cross-Document Conflict Scan

## Confirmed Conflicts
Use a Markdown table:
| Topic | Document / Page A | Statement A | Document / Page B | Statement B | Operational Impact |

Only include a row when the supplied text clearly conflicts.
If none are found, write "No confirmed conflicts found in supplied text."

## Potential Inconsistencies to Verify
Use this for wording that may conflict but cannot be confirmed from the supplied text.

## Duplicate / Overlapping Rules
List important topics covered by more than one document, even when consistent.

## Highest-Priority Review Points
3-5 evidence-based items for a human reviewer.

PDF CONTENT:
{context}
"""
    return ask_model(prompt) or "Could not complete the cross-document conflict scan."


def clean_evidence_text(text, max_length=500):
    text = " ".join(text.split())

    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."

    return text


def build_evidence_list(selected_chunks):
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

        seen.add(key)

        evidence_list.append(
            {
                "file": chunk["file"],
                "page": chunk["page"],
                "score": float(item["score"]),
                "text": clean_evidence_text(chunk["text"]),
            }
        )

    return evidence_list


def display_evidence(evidence_list):
    if not evidence_list:
        return

    with st.expander(
        "🔎 View Evidence",
        expanded=False,
    ):
        for number, evidence in enumerate(evidence_list, start=1):
            st.markdown(f"### Evidence {number}")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption("PDF")
                st.write(evidence["file"])

            with c2:
                st.caption("Page")
                st.write(evidence["page"])

            with c3:
                st.caption("Match")
                st.write(f'{evidence["score"]:.3f}')

            st.info(evidence["text"])

            if number < len(evidence_list):
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

    return "\n".join(parts)


# ============================================================
# PROFESSIONAL PDF EXPORT
# ============================================================

def clean_markdown_for_pdf(text):
    if not text:
        return ""

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    return text


def add_text_to_pdf_story(story, text, styles):
    if not text:
        return

    text = clean_markdown_for_pdf(text)
    lines = text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 3 * mm))
            continue

        safe_line = html.escape(line)

        if line.startswith("### "):
            story.append(
                Paragraph(
                    html.escape(line[4:]),
                    styles["JAXHeading3"],
                )
            )

        elif line.startswith("## "):
            story.append(
                Paragraph(
                    html.escape(line[3:]),
                    styles["JAXHeading2"],
                )
            )

        elif line.startswith("# "):
            story.append(
                Paragraph(
                    html.escape(line[2:]),
                    styles["JAXHeading1"],
                )
            )

        elif line.startswith("- ") or line.startswith("* "):
            bullet_text = html.escape(line[2:])
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
        textColor=colors.HexColor("#1F2937"),
    )

    styles["JAXSubtitle"] = ParagraphStyle(
        "JAXSubtitle",
        parent=sample_styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=6 * mm,
        textColor=colors.HexColor("#4B5563"),
    )

    styles["JAXHeading1"] = ParagraphStyle(
        "JAXHeading1",
        parent=sample_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceBefore=6 * mm,
        spaceAfter=4 * mm,
        textColor=colors.HexColor("#111827"),
    )

    styles["JAXHeading2"] = ParagraphStyle(
        "JAXHeading2",
        parent=sample_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        textColor=colors.HexColor("#1F2937"),
    )

    styles["JAXHeading3"] = ParagraphStyle(
        "JAXHeading3",
        parent=sample_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
        textColor=colors.HexColor("#374151"),
    )

    styles["JAXBody"] = ParagraphStyle(
        "JAXBody",
        parent=sample_styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        spaceAfter=2.5 * mm,
        textColor=colors.HexColor("#1F2937"),
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
        ["Report Scope", selected_pdf],
        ["Documents", str(len(pdf_names))],
        ["Generated By", "JAXOVIQ AI Assistant"],
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
                    colors.HexColor("#F3F4F6"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#111827"),
                ),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D1D5DB"),
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(info_table)
    story.append(Spacer(1, 8 * mm))

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
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Document Summary",
                styles["JAXHeading1"],
            )
        )
        add_text_to_pdf_story(story, summary, styles)

    if actions:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Action Items",
                styles["JAXHeading1"],
            )
        )
        add_text_to_pdf_story(story, actions, styles)

    if insights:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Document Insights",
                styles["JAXHeading1"],
            )
        )
        add_text_to_pdf_story(story, insights, styles)

    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "Generated by JAXOVIQ AI Assistant",
            styles["JAXSubtitle"],
        )
    )

    document.build(story)

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
    saved_data = load_persistent_knowledge_base()

    if saved_data:
        st.session_state.index = saved_data["index"]
        st.session_state.chunks = saved_data["chunks"]
        st.session_state.pdf_names = saved_data["pdf_names"]
        st.session_state.knowledge_base_ready = True

    st.session_state.persistent_loaded = True


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚡ JAXOVIQ")

    st.caption("AI Knowledge Assistant")

    st.divider()

    if st.session_state.knowledge_base_ready:
        st.success("● Knowledge Base Active")
    else:
        st.warning("● Knowledge Base Not Built")

    st.subheader("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    document_options = ["All Documents"]

    if st.session_state.pdf_names:
        document_options += st.session_state.pdf_names

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
            st.warning("Please upload at least one PDF.")

        else:
            with st.spinner(
                "Building JAXOVIQ Knowledge Base..."
            ):
                index, chunks = build_faiss_index(uploaded_files)

            if index is not None and chunks is not None:
                pdf_names = [
                    file.name
                    for file in uploaded_files
                ]

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.pdf_names = pdf_names
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

                st.session_state.comparison_result = None
                st.session_state.comparison_pair = None
                st.session_state.risk_check_result = None
                st.session_state.risk_check_scope = None
                st.session_state.deadline_result = None
                st.session_state.deadline_scope = None
                st.session_state.executive_brief = None
                st.session_state.executive_brief_scope = None
                st.session_state.document_audit = None
                st.session_state.document_audit_scope = None
                st.session_state.conflict_scan = None
                st.session_state.conflict_scan_scope = None
                st.session_state.obligation_register = None
                st.session_state.obligation_scope = None
                st.session_state.decision_memo = None
                st.session_state.decision_scope = None
                st.session_state.full_scan_last_scope = None

                track_event("pdf_uploads", amount=len(pdf_names))

                save_persistent_knowledge_base(
                    index,
                    chunks,
                    pdf_names,
                )

                with st.spinner("Generating smart questions..."):
                    st.session_state.suggested_questions = (
                        generate_suggested_questions("All Documents")
                    )
                    st.session_state.suggestions_scope = "All Documents"

                st.success("Knowledge base ready and saved.")
                st.rerun()

    if st.session_state.pdf_names:
        st.divider()
        st.caption("Loaded PDFs")

        for pdf_name in st.session_state.pdf_names:
            st.write(f"📄 {pdf_name}")

        if INDEX_FILE.exists() and CHUNKS_FILE.exists():
            st.caption("💾 Saved locally")

    st.divider()

    with st.expander("🔐 Admin Usage Stats", expanded=False):
        admin_password = st.text_input(
            "Admin password",
            type="password",
            key="admin_password_input",
        )

        # Set ADMIN_PASSWORD in Streamlit Secrets for production.
        configured_password = ""
        try:
            configured_password = st.secrets.get("ADMIN_PASSWORD", "")
        except Exception:
            configured_password = ""

        if configured_password:
            if st.button("Unlock Admin", use_container_width=True):
                st.session_state.admin_authenticated = (
                    admin_password == configured_password
                )
                if not st.session_state.admin_authenticated:
                    st.error("Incorrect admin password.")
        else:
            st.caption(
                "Admin dashboard is ready. Add ADMIN_PASSWORD in Streamlit Secrets to unlock it."
            )

        if st.session_state.admin_authenticated:
            st.session_state.admin_test_mode = st.checkbox(
                "🧪 Admin Test Mode",
                value=st.session_state.admin_test_mode,
                help="When enabled, your own testing does not increase analytics counters.",
            )

            usage = load_analytics()
            st.caption("Traffic Sources")
            for source in ["Meta", "LinkedIn", "Instagram", "Google", "Direct", "Other"]:
                st.write(f"{source}: **{usage['traffic_sources'].get(source, 0)}**")

            a1, a2 = st.columns(2)
            with a1:
                st.metric("App Opens", usage.get("app_opens", 0))
                st.metric("Questions", usage.get("questions", 0))
                st.metric("Reports", usage.get("reports_exported", 0))
            with a2:
                st.metric("PDF Uploads", usage.get("pdf_uploads", 0))
                st.metric("Analysis Runs", usage.get("analysis_runs", 0))
                st.metric("Feedback", usage.get("feedback_count", 0))

    st.divider()

    col_a, col_b = st.columns(2)

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
# CLIENT-DEMO HERO
# ============================================================

st.markdown(
    """
    <div class="jaxoviq-hero">
        <div class="jaxoviq-eyebrow">AI Document Intelligence</div>
        <div class="jaxoviq-title">⚡ JAXOVIQ AI Assistant</div>
        <div class="jaxoviq-subtitle">
            Turn business PDFs into a searchable knowledge base.
            Ask questions, verify answers with page-level evidence,
            compare documents, detect review points, extract deadlines,
            create executive briefs and export decision-ready reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

use1, use2, use3, use4 = st.columns(4)

with use1:
    st.markdown(
        """
        <div class="jaxoviq-usecase">
            <div class="jaxoviq-usecase-title">🍽️ Restaurants</div>
            <div class="jaxoviq-usecase-text">
                SOPs, menus, supplier documents, staff manuals and policies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with use2:
    st.markdown(
        """
        <div class="jaxoviq-usecase">
            <div class="jaxoviq-usecase-title">👥 HR & Operations</div>
            <div class="jaxoviq-usecase-text">
                Employee handbooks, procedures, onboarding and internal knowledge.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with use3:
    st.markdown(
        """
        <div class="jaxoviq-usecase">
            <div class="jaxoviq-usecase-title">📑 Contracts & Policies</div>
            <div class="jaxoviq-usecase-text">
                Find clauses, dates, obligations, risks and supporting evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with use4:
    st.markdown(
        """
        <div class="jaxoviq-usecase">
            <div class="jaxoviq-usecase-title">📊 Business Reports</div>
            <div class="jaxoviq-usecase-text">
                Summaries, action items, insights and downloadable AI reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="demo-note">
        <strong>Grounded answers:</strong>
        JAXOVIQ answers from uploaded PDFs only and shows source files,
        page numbers, evidence and confidence scores.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP DASHBOARD
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "PDFs",
        len(st.session_state.pdf_names),
    )

with metric2:
    st.metric(
        "Knowledge Chunks",
        len(st.session_state.chunks),
    )

with metric3:
    st.metric(
        "Chat Messages",
        len(st.session_state.messages),
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

    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        st.caption("💾 Persistent Knowledge Base: Saved")

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
        expanded=False,
    ):
        st.caption(f"Current scope: {selected_pdf}")

        for pdf_name in st.session_state.pdf_names:
            data = stats.get(
                pdf_name,
                {
                    "pages": 0,
                    "chunks": 0,
                },
            )

            st.markdown(f"### 📄 {pdf_name}")

            stat1, stat2, stat3 = st.columns(3)

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
    st.subheader("💡 Suggested Questions")

    st.caption(
        "Click a question and JAXOVIQ will search your PDFs automatically."
    )

    if st.session_state.suggestions_scope != selected_pdf:
        if st.button(
            "✨ Generate Questions for Current Scope",
            use_container_width=True,
        ):
            with st.spinner("Generating smart questions..."):
                st.session_state.suggested_questions = (
                    generate_suggested_questions(selected_pdf)
                )

                st.session_state.suggestions_scope = selected_pdf

            st.rerun()

    if st.session_state.suggested_questions:
        for index, suggested_question in enumerate(
            st.session_state.suggested_questions,
            start=1,
        ):
            safe_scope = selected_pdf.replace(" ", "_")

            if st.button(
                f"💬 {suggested_question}",
                key=f"suggested_question_{safe_scope}_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = suggested_question
                st.rerun()

    else:
        st.info("No suggested questions available yet.")

    st.divider()


# ============================================================
# JAXOVIQ INTELLIGENCE COMMAND CENTER
# ============================================================

if st.session_state.knowledge_base_ready:
    st.subheader("⚡ JAXOVIQ Intelligence Command Center")
    st.caption(
        "Executive intelligence, risk review, obligation extraction, conflict detection, "
        "decision support and evidence-grounded document comparison."
    )

    # One-click scan for maximum demo impact.
    full_scan_clicked = st.button(
        "🚀 Run Full Intelligence Scan",
        use_container_width=True,
        type="primary",
        help="Runs the main intelligence modules for the current scope. Multiple AI analyses may take a little time.",
    )

    if full_scan_clicked:
        with st.status("Running JAXOVIQ full intelligence scan...", expanded=True) as scan_status:
            st.write("🧠 Building Executive Brief...")
            st.session_state.executive_brief = generate_executive_brief(selected_pdf)
            st.session_state.executive_brief_scope = selected_pdf

            st.write("🛡️ Reviewing risks and policies...")
            st.session_state.risk_check_result = run_risk_check(selected_pdf)
            st.session_state.risk_check_scope = selected_pdf

            st.write("📅 Extracting deadlines and key terms...")
            st.session_state.deadline_result = extract_deadlines_and_key_terms(selected_pdf)
            st.session_state.deadline_scope = selected_pdf

            st.write("🔎 Auditing document quality...")
            st.session_state.document_audit = run_document_audit(selected_pdf)
            st.session_state.document_audit_scope = selected_pdf

            st.write("📋 Building obligation register...")
            st.session_state.obligation_register = generate_obligation_register(selected_pdf)
            st.session_state.obligation_scope = selected_pdf

            st.write("🧭 Building decision memo...")
            st.session_state.decision_memo = generate_decision_memo(selected_pdf)
            st.session_state.decision_scope = selected_pdf

            if len(st.session_state.pdf_names) >= 2:
                st.write("⚔️ Scanning cross-document conflicts...")
                st.session_state.conflict_scan = scan_cross_document_conflicts()
                st.session_state.conflict_scan_scope = "All Documents"

            st.session_state.full_scan_last_scope = selected_pdf
            track_event("analysis_runs")
            scan_status.update(label="✅ Full intelligence scan complete", state="complete", expanded=False)

    row1 = st.columns(4)
    row2 = st.columns(3)

    with row1[0]:
        executive_clicked = st.button(
            "🧠 Executive Brief",
            use_container_width=True,
            help="Leadership-ready snapshot with evidence, numbers, risks and actions.",
        )
    with row1[1]:
        risk_clicked = st.button(
            "🛡️ Contract / Policy Check",
            use_container_width=True,
            help="Flags practical risks, ambiguity, obligations and review points. Not legal advice.",
        )
    with row1[2]:
        deadline_clicked = st.button(
            "📅 Deadlines & Key Terms",
            use_container_width=True,
            help="Extracts dates, notice periods, renewals, money terms, limits and other key facts.",
        )
    with row1[3]:
        audit_clicked = st.button(
            "🔎 Document Audit",
            use_container_width=True,
            help="Checks clarity, missing information, operational risk flags and improvement opportunities.",
        )

    with row2[0]:
        obligations_clicked = st.button(
            "📋 Obligation Register",
            use_container_width=True,
            help="Turns document duties, approvals, restrictions and deadlines into an actionable register.",
        )
    with row2[1]:
        decision_clicked = st.button(
            "🧭 Decision Memo",
            use_container_width=True,
            help="Creates a management-ready memo with evidence, open questions and grounded next actions.",
        )
    with row2[2]:
        conflict_clicked = st.button(
            "⚔️ Conflict Scanner",
            use_container_width=True,
            disabled=len(st.session_state.pdf_names) < 2,
            help="Checks multiple PDFs for contradictory dates, rules, amounts, responsibilities and policy terms.",
        )

    if executive_clicked:
        with st.spinner("Building executive intelligence brief..."):
            st.session_state.executive_brief = generate_executive_brief(selected_pdf)
            st.session_state.executive_brief_scope = selected_pdf
            track_event("analysis_runs")

    if risk_clicked:
        with st.spinner("Reviewing contract / policy terms..."):
            st.session_state.risk_check_result = run_risk_check(selected_pdf)
            st.session_state.risk_check_scope = selected_pdf
            track_event("analysis_runs")

    if deadline_clicked:
        with st.spinner("Extracting deadlines and key terms..."):
            st.session_state.deadline_result = extract_deadlines_and_key_terms(selected_pdf)
            st.session_state.deadline_scope = selected_pdf
            track_event("analysis_runs")

    if audit_clicked:
        with st.spinner("Auditing document quality and operational clarity..."):
            st.session_state.document_audit = run_document_audit(selected_pdf)
            st.session_state.document_audit_scope = selected_pdf
            track_event("analysis_runs")

    if obligations_clicked:
        with st.spinner("Building obligation register..."):
            st.session_state.obligation_register = generate_obligation_register(selected_pdf)
            st.session_state.obligation_scope = selected_pdf
            track_event("analysis_runs")

    if decision_clicked:
        with st.spinner("Building management decision memo..."):
            st.session_state.decision_memo = generate_decision_memo(selected_pdf)
            st.session_state.decision_scope = selected_pdf
            track_event("analysis_runs")

    if conflict_clicked:
        with st.spinner("Scanning documents for conflicts and inconsistencies..."):
            st.session_state.conflict_scan = scan_cross_document_conflicts()
            st.session_state.conflict_scan_scope = "All Documents"
            track_event("analysis_runs")

    # ---------- Robust side-by-side comparison ----------
    if len(st.session_state.pdf_names) >= 2:
        st.markdown("### 🔄 Compare Two Documents")
        compare_col1, compare_col2, compare_col3 = st.columns([2, 2, 1])

        with compare_col1:
            compare_a = st.selectbox(
                "Document A",
                st.session_state.pdf_names,
                index=0,
                key="compare_document_a",
            )

        with compare_col2:
            default_b_index = 1 if len(st.session_state.pdf_names) > 1 else 0
            compare_b = st.selectbox(
                "Document B",
                st.session_state.pdf_names,
                index=default_b_index,
                key="compare_document_b",
            )

        with compare_col3:
            st.write("")
            st.write("")
            compare_clicked = st.button(
                "⚡ Compare",
                use_container_width=True,
                type="primary",
                key="extreme_compare_button",
            )

        if compare_clicked:
            if compare_a == compare_b:
                st.warning("Choose two different documents to compare.")
            else:
                with st.spinner(f"Comparing {compare_a} with {compare_b}..."):
                    comparison = compare_documents(compare_a, compare_b)

                if comparison and comparison.strip():
                    st.session_state.comparison_result = comparison
                    st.session_state.comparison_pair = (compare_a, compare_b)
                    track_event("analysis_runs")
                    st.success("✅ Comparison complete — result shown directly below.")
                else:
                    st.session_state.comparison_result = None
                    st.session_state.comparison_pair = None
                    st.error(
                        "Comparison could not be generated. Both PDFs were loaded, but the AI returned no result. "
                        "Try again once; if it repeats, rebuild the knowledge base."
                    )

        # Keep comparison output right under the Compare controls so it is impossible to miss.
        if st.session_state.comparison_result and st.session_state.comparison_pair:
            pair = st.session_state.comparison_pair
            st.markdown(f"### 🔄 Comparison Result: {pair[0]} ↔ {pair[1]}")
            st.markdown(st.session_state.comparison_result)
    else:
        st.info("Upload at least 2 PDFs to unlock Document Comparison and Conflict Scanner.")

    # ---------- Intelligence outputs ----------
    if (
        st.session_state.executive_brief
        and st.session_state.executive_brief_scope == selected_pdf
    ):
        with st.expander("🧠 Executive Brief", expanded=True):
            st.markdown(st.session_state.executive_brief)

    if (
        st.session_state.decision_memo
        and st.session_state.decision_scope == selected_pdf
    ):
        with st.expander("🧭 Decision Memo", expanded=True):
            st.markdown(st.session_state.decision_memo)

    if (
        st.session_state.obligation_register
        and st.session_state.obligation_scope == selected_pdf
    ):
        with st.expander("📋 Obligation Register", expanded=True):
            st.markdown(st.session_state.obligation_register)

    if (
        st.session_state.document_audit
        and st.session_state.document_audit_scope == selected_pdf
    ):
        with st.expander("🔎 Document Audit", expanded=True):
            st.info("Operational document review only — not legal advice.")
            st.markdown(st.session_state.document_audit)

    if (
        st.session_state.risk_check_result
        and st.session_state.risk_check_scope == selected_pdf
    ):
        with st.expander("🛡️ Contract / Policy Check", expanded=True):
            st.warning(
                "AI-assisted document review only — not legal advice. "
                "Important clauses should be verified by a qualified professional."
            )
            st.markdown(st.session_state.risk_check_result)

    if (
        st.session_state.deadline_result
        and st.session_state.deadline_scope == selected_pdf
    ):
        with st.expander("📅 Deadlines & Key Terms", expanded=True):
            st.markdown(st.session_state.deadline_result)

    if st.session_state.conflict_scan:
        with st.expander("⚔️ Cross-Document Conflict Scan", expanded=True):
            st.markdown(st.session_state.conflict_scan)

    st.divider()


# ============================================================
# DOCUMENT TOOLS
# ============================================================

if st.session_state.knowledge_base_ready:
    st.subheader("🧰 Document Tools")

    col1, col2, col3, col4 = st.columns(4)

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
        with st.spinner("Creating document summary..."):
            st.session_state.pdf_summary = (
                generate_pdf_summary(selected_pdf)
            )
            st.session_state.summary_scope = selected_pdf

    if actions_clicked:
        with st.spinner("Finding action items..."):
            st.session_state.action_items = (
                generate_action_items(selected_pdf)
            )
            st.session_state.action_scope = selected_pdf

    if insights_clicked:
        with st.spinner("Analyzing document insights..."):
            st.session_state.document_insights = (
                generate_document_insights(selected_pdf)
            )
            st.session_state.insights_scope = selected_pdf

    if export_clicked:
        track_event("reports_exported")
        with st.spinner(
            "Preparing professional JAXOVIQ report..."
        ):
            summary = (
                st.session_state.pdf_summary
                if st.session_state.summary_scope == selected_pdf
                else None
            )

            if not summary:
                summary = generate_pdf_summary(selected_pdf)
                st.session_state.pdf_summary = summary
                st.session_state.summary_scope = selected_pdf

            actions = (
                st.session_state.action_items
                if st.session_state.action_scope == selected_pdf
                else None
            )

            if not actions:
                actions = generate_action_items(selected_pdf)
                st.session_state.action_items = actions
                st.session_state.action_scope = selected_pdf

            insights = (
                st.session_state.document_insights
                if st.session_state.insights_scope == selected_pdf
                else None
            )

            if not insights:
                insights = generate_document_insights(selected_pdf)
                st.session_state.document_insights = insights
                st.session_state.insights_scope = selected_pdf

            st.session_state.export_report = (
                build_export_report(
                    selected_pdf,
                    summary,
                    actions,
                    insights,
                )
            )

            st.session_state.export_scope = selected_pdf

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

                st.session_state.pdf_report_scope = selected_pdf

            except Exception as e:
                st.session_state.pdf_report_bytes = None
                st.session_state.pdf_report_scope = None

                st.error(f"Could not create PDF report: {e}")

    if (
        st.session_state.pdf_summary
        and st.session_state.summary_scope == selected_pdf
    ):
        with st.expander(
            "📄 Document Summary",
            expanded=True,
        ):
            st.markdown(st.session_state.pdf_summary)

    if (
        st.session_state.action_items
        and st.session_state.action_scope == selected_pdf
    ):
        with st.expander(
            "✅ Action Items",
            expanded=True,
        ):
            st.markdown(st.session_state.action_items)

    if (
        st.session_state.document_insights
        and st.session_state.insights_scope == selected_pdf
    ):
        with st.expander(
            "📊 Document Insights",
            expanded=True,
        ):
            st.markdown(st.session_state.document_insights)

    if (
        st.session_state.export_report
        and st.session_state.export_scope == selected_pdf
    ):
        st.success("📥 JAXOVIQ report is ready.")

        download_col1, download_col2 = st.columns(2)

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
                and st.session_state.pdf_report_scope == selected_pdf
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

st.subheader("💬 Ask JAXOVIQ")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(source)

        if message.get("evidence"):
            display_evidence(message["evidence"])

        if message.get("confidence"):
            st.caption(
                f'Confidence: {message["confidence"]}'
            )

        if message.get("score") is not None:
            st.caption(
                f'Similarity score: '
                f'{message["score"]:.3f}'
            )


# Voice input: record a question, transcribe it, then send it through
# the exact same grounded RAG pipeline as typed questions.
voice_question = None

with st.expander("🎙️ Ask by Voice", expanded=False):
    st.caption("Record your question. JAXOVIQ converts it to text and searches only your uploaded documents.")
    voice_audio = st.audio_input(
        "Record a voice question",
        key="jaxoviq_voice_audio",
    )

    if voice_audio is not None:
        voice_bytes = voice_audio.getvalue()
        voice_hash = hashlib.sha256(voice_bytes).hexdigest()

        if voice_hash != st.session_state.last_voice_hash:
            with st.spinner("Transcribing your voice question..."):
                transcribed_text = transcribe_voice(voice_audio)

            st.session_state.last_voice_hash = voice_hash
            st.session_state.last_voice_text = transcribed_text

        if st.session_state.last_voice_text:
            st.success(f'Heard: “{st.session_state.last_voice_text}”')
            st.caption("Check the transcription, then press the button once to ask JAXOVIQ.")
            if st.button(
                "🎙️ Ask This Voice Question",
                type="primary",
                use_container_width=True,
                key="ask_voice_question_button",
            ):
                if voice_hash != st.session_state.last_submitted_voice_hash:
                    voice_question = st.session_state.last_voice_text
                    st.session_state.last_submitted_voice_hash = voice_hash
                else:
                    st.info("This recording was already submitted. Record a new question to ask again.")

typed_question = st.chat_input(
    "Ask a question about your PDFs..."
)

question = None

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

elif voice_question:
    question = voice_question

elif typed_question:
    question = typed_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    if not st.session_state.knowledge_base_ready:
        st.warning("Please build the knowledge base first.")

    else:
        track_event("questions")
        with st.spinner("Understanding your question..."):
            resolved_question = (
                resolve_followup_question(question)
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                candidates, best_score = (
                    retrieve_chunks(
                        resolved_question,
                        selected_pdf,
                    )
                )

                confidence = get_confidence_label(best_score)

                if not candidates or best_score < MIN_SCORE:
                    answer = (
                        "I could not find that "
                        "information in the PDFs."
                    )

                    st.markdown(answer)
                    st.caption(f"Confidence: {confidence}")
                    st.caption(
                        f"Similarity score: {best_score:.3f}"
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

                    # Clean weak unrelated evidence from final client-facing output.
                    selected = filter_selected_evidence(
                        selected,
                        best_score,
                    )

                    answer = generate_answer(
                        question,
                        resolved_question,
                        selected,
                    )

                    if not answer:
                        answer = "I could not generate an answer."

                    st.markdown(answer)

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

                        seen_sources.add(source_key)

                        source_list.append(
                            f'📄 {chunk["file"]} '
                            f'— page {chunk["page"]}'
                        )

                    if source_list:
                        with st.expander("📚 Sources"):
                            for source in source_list:
                                st.write(source)

                    evidence_list = build_evidence_list(selected)
                    display_evidence(evidence_list)

                    st.caption(f"Confidence: {confidence}")
                    st.caption(
                        f"Similarity score: {best_score:.3f}"
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


# ============================================================
# USER FEEDBACK
# ============================================================

with st.expander("💬 Give Feedback", expanded=False):
    st.caption("Help improve JAXOVIQ. Feedback is saved privately with the app data.")
    feedback_rating = st.select_slider(
        "How useful was JAXOVIQ?",
        options=[1, 2, 3, 4, 5],
        value=5,
        key="feedback_rating",
    )
    feedback_message = st.text_area(
        "What worked well or what should improve?",
        key="feedback_message",
        placeholder="Example: The answer was useful, but I want a clearer comparison table...",
    )
    if st.button("Send Feedback", use_container_width=True, key="send_feedback_button"):
        if feedback_message.strip():
            save_feedback(feedback_rating, feedback_message)
            st.success("Thank you — feedback saved.")
        else:
            st.warning("Please write a short feedback message first.")
