import hmac
import io
import os
import re
import threading
import uuid
from datetime import datetime, timezone

import requests
import streamlit as st
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Must be the first Streamlit call
st.set_page_config(page_title="Project Rini v1.0", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAIN_COLLECTION = "rini_growth_marketing"      # trusted knowledge base (retrieved from)
PENDING_COLLECTION = "rini_pending_review"     # user corrections awaiting human review (never retrieved)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ANSWER_MODEL = "openai/gpt-oss-120b"
HISTORY_TURNS = 3
MAX_UPLOAD_MB = 15


def get_secret(name: str) -> str:
    """Streamlit secrets first, environment variable as fallback (handy for local runs)."""
    try:
        return str(st.secrets[name])
    except Exception:
        return os.environ.get(name, "")


QDRANT_ENDPOINT = get_secret("QDRANT_API_ENDPOINT")
QDRANT_KEY = get_secret("QDRANT_API_KEY")
GROQ_KEY = get_secret("GROQ_API_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")      # optional: enables the admin panel
LOG_WEBHOOK_URL = get_secret("LOG_WEBHOOK_URL")    # optional: enables logging to Google Sheets
LOG_TOKEN = get_secret("LOG_TOKEN")

if not (QDRANT_ENDPOINT and QDRANT_KEY and GROQ_KEY):
    st.error(
        "🔒 Missing credentials. Set QDRANT_API_ENDPOINT, QDRANT_API_KEY and "
        "GROQ_API_KEY in the app's Secrets (or as environment variables locally)."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def load_qdrant(endpoint: str, api_key: str) -> QdrantClient:
    client = QdrantClient(url=endpoint, api_key=api_key, timeout=30)
    for name in (MAIN_COLLECTION, PENDING_COLLECTION):
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
            )
    try:
        client.create_payload_index(MAIN_COLLECTION, "source_doc", models.PayloadSchemaType.KEYWORD)
    except Exception:
        pass  # already exists
    return client


try:
    qdrant = load_qdrant(QDRANT_ENDPOINT, QDRANT_KEY)
except Exception as exc:
    st.error(f"Could not connect to Qdrant: {exc}")
    st.stop()

embedder = load_embedder()
groq_client = Groq(api_key=GROQ_KEY)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "chat_history": [],
    "last_query": None,
    "last_response": None,
    "retrieved": [],
    "retrieval_error": None,
    "show_correction": False,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# These are deliberately NOT in `defaults`, so "Clear conversation" doesn't reset them
st.session_state.setdefault("session_id", uuid.uuid4().hex[:10])  # random, not linked to a person
st.session_state.setdefault("is_admin", False)
st.session_state.setdefault("admin_attempts", 0)


# ---------------------------------------------------------------------------
# Privacy-safe logging to Google Sheets (via an Apps Script web app)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def _mask_phone(match: re.Match) -> str:
    return "[phone]" if sum(ch.isdigit() for ch in match.group()) >= 9 else match.group()


def redact(text: str) -> str:
    """Strip e-mail addresses and phone-number-like digit runs before anything is logged."""
    return PHONE_RE.sub(_mask_phone, EMAIL_RE.sub("[email]", text or ""))


def log_event(event: str, **fields) -> None:
    """Fire-and-forget: never slows down or breaks the app if logging is down."""
    if not LOG_WEBHOOK_URL:
        return
    payload = {
        "token": LOG_TOKEN,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": st.session_state.session_id,
        "event": event,
    }
    for name, value in fields.items():
        payload[name] = redact(value)[:3000] if isinstance(value, str) else value

    def _send() -> None:
        try:
            requests.post(LOG_WEBHOOK_URL, json=payload, timeout=10)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


# ---------------------------------------------------------------------------
# Chunking (paragraph-aware, keeps the nearest markdown heading with each chunk)
# ---------------------------------------------------------------------------
def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_document(text: str, max_chars: int = 1000, overlap: int = 150) -> list[tuple[str, str]]:
    sections, heading, paras = [], "", []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            first, _, rest = block.partition("\n")
            if paras:
                sections.append((heading, paras))
                paras = []
            heading = first.lstrip("#").strip()
            block = rest.strip()
            if not block:
                continue
        paras.extend(_split_long(block, max_chars))
    if paras:
        sections.append((heading, paras))

    chunks = []
    for heading, paras in sections:
        cur = ""
        for para in paras:
            if cur and len(cur) + 2 + len(para) > max_chars:
                chunks.append((heading, cur))
                tail = cur[-overlap:] if overlap else ""
                tail = tail[tail.find(" ") + 1 :] if " " in tail else ""
                cur = f"{tail}\n\n{para}" if tail and len(tail) + 2 + len(para) <= max_chars else para
            else:
                cur = f"{cur}\n\n{para}" if cur else para
        if cur:
            chunks.append((heading, cur))
    return chunks


# ---------------------------------------------------------------------------
# Knowledge-base management (admin only)
# ---------------------------------------------------------------------------
def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # un-wrap hard line breaks from PDF layout
    return data.decode("utf-8", errors="ignore")


def ingest_document(name: str, text: str) -> int:
    """Chunk, embed and store one document. Re-uploading the same filename replaces the old version."""
    chunks = chunk_document(text)
    if not chunks:
        return 0
    texts = [f"{h}\n{c}" if h else c for h, c in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).tolist()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}#{i}")),
            vector=vec,
            payload={"source_doc": name, "chunk_index": i, "heading": h, "text_content": t, "uploaded_at": now},
        )
        for i, ((h, _), t, vec) in enumerate(zip(chunks, texts, vectors))
    ]
    qdrant.delete(
        collection_name=MAIN_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="source_doc", match=models.MatchValue(value=name))])
        ),
        wait=True,
    )
    for start in range(0, len(points), 64):
        qdrant.upsert(collection_name=MAIN_COLLECTION, points=points[start : start + 64], wait=True)
    return len(points)


def list_pending(limit: int = 20):
    points, _ = qdrant.scroll(PENDING_COLLECTION, limit=limit, with_payload=True, with_vectors=True)
    return points


def approve_pending(point) -> None:
    payload = point.payload or {}
    qdrant.upsert(
        MAIN_COLLECTION,
        points=[
            models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload={
                    "source_doc": "reviewed_corrections",
                    "text_content": payload.get("text_content", ""),
                    "source_url": payload.get("source_url", ""),
                    "status": "approved",
                },
            )
        ],
        wait=True,
    )
    qdrant.delete(PENDING_COLLECTION, points_selector=models.PointIdsList(points=[point.id]), wait=True)


def reject_pending(point_id) -> None:
    qdrant.delete(PENDING_COLLECTION, points_selector=models.PointIdsList(points=[point_id]), wait=True)


def render_admin_panel() -> None:
    st.markdown("**Upload documents** (.md, .txt, .pdf)")
    files = st.file_uploader(
        "Documents", type=["md", "txt", "pdf"], accept_multiple_files=True, label_visibility="collapsed", key="admin_files"
    )
    if files and st.button("Add to knowledge base", type="primary", key="admin_ingest"):
        bar = st.progress(0.0)
        for i, uploaded in enumerate(files, 1):
            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"{uploaded.name}: larger than {MAX_UPLOAD_MB} MB, skipped.")
            else:
                try:
                    text = extract_text(uploaded.name, uploaded.getvalue())
                    if not text.strip():
                        st.warning(f"{uploaded.name}: no text found (scanned PDF?). Skipped.")
                    else:
                        st.success(f"{uploaded.name}: {ingest_document(uploaded.name, text)} passages stored.")
                except Exception as exc:
                    st.error(f"{uploaded.name}: failed ({exc})")
            bar.progress(i / len(files))
        st.caption("Reload the page to refresh the passage count.")

    st.divider()
    st.markdown("**Corrections waiting for review**")
    try:
        pending = list_pending()
    except Exception as exc:
        st.error(f"Could not load corrections: {exc}")
        pending = []
    if not pending:
        st.caption("None waiting.")
    for point in pending:
        payload = point.payload or {}
        st.markdown(f"Q: {payload.get('original_query') or '(unknown)'}")
        st.text(payload.get("text_content", ""))
        st.caption(f"Source: {payload.get('source_url') or 'none given'}")
        ok_col, no_col = st.columns(2)
        if ok_col.button("Approve", key=f"ok_{point.id}"):
            approve_pending(point)
            st.rerun()
        if no_col.button("Reject", key=f"no_{point.id}"):
            reject_pending(point.id)
            st.rerun()
        st.divider()
    if st.button("Lock admin panel", key="admin_lock"):
        st.session_state.is_admin = False
        st.rerun()


# ---------------------------------------------------------------------------
# Core chat functions
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int, threshold: float):
    """Dense retrieval from the trusted collection, dropping weak matches."""
    vector = embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    result = qdrant.query_points(
        collection_name=MAIN_COLLECTION,
        query=vector,
        limit=top_k,
        score_threshold=threshold,
        with_payload=True,
    )
    return result.points


def build_system_prompt(passages: list[dict]) -> str:
    if passages:
        context = "\n\n".join(
            f"[{i}] (source: {p['source']})\n{p['text']}" for i, p in enumerate(passages, 1)
        )
        rules = (
            "Answer using the numbered context passages below and cite them inline like [1]. "
            "If the passages do not fully answer the question, say which part is not covered, and "
            "label anything you add from general knowledge as 'general knowledge (unverified)'. "
            "Never invent API fields, endpoints, parameters or figures."
        )
    else:
        context = "(no relevant passages found in the knowledge base)"
        rules = (
            "No sourced passages matched this question. Say so plainly in your first sentence, then "
            "give a best-effort answer clearly labelled 'general knowledge (unverified)'. "
            "Never invent API fields, endpoints, parameters or figures."
        )
    return (
        "You are Rini, a precise digital-marketing analytics assistant. "
        f"{rules}\n\nCONTEXT:\n{context}"
    )


def stream_answer(messages: list[dict]):
    """Yield text deltas from Groq so st.write_stream gets plain strings."""
    stream = groq_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def submit_correction(correction: str, source_url: str) -> None:
    """Corrections go to a quarantine collection, NOT the live knowledge base."""
    vector = embedder.encode(correction, normalize_embeddings=True).tolist()
    qdrant.upsert(
        collection_name=PENDING_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source_doc": "user_correction",
                    "text_content": correction,
                    "source_url": source_url,
                    "original_query": st.session_state.last_query,
                    "flagged_answer": (st.session_state.last_response or "")[:2000],
                    "status": "pending",
                },
            )
        ],
        wait=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Retrieval settings")
    top_k = st.slider("Passages to retrieve", 1, 8, 4)
    threshold = st.slider(
        "Minimum similarity",
        0.0,
        0.9,
        0.50,
        0.05,
        help="Passages scoring below this are discarded. Tune it using the scores shown in the telemetry panel.",
    )
    try:
        kb_size = qdrant.count(collection_name=MAIN_COLLECTION, exact=True).count
        st.metric("Knowledge base passages", kb_size)
        if kb_size < 20:
            st.warning("Knowledge base is almost empty, so most answers will be ungrounded. Upload documents in the Admin panel.")
    except Exception as exc:
        st.error(f"Could not read knowledge base size: {exc}")
    if st.button("Clear conversation", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

    if ADMIN_PASSWORD:
        with st.expander("🔐 Admin", expanded=st.session_state.is_admin):
            if st.session_state.is_admin:
                render_admin_panel()
            else:
                password = st.text_input("Admin password", type="password", key="admin_pw")
                if st.button("Unlock", key="admin_unlock"):
                    if st.session_state.admin_attempts >= 5:
                        st.error("Too many attempts. Reload the page to try again.")
                    elif hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode()):
                        st.session_state.is_admin = True
                        st.rerun()
                    else:
                        st.session_state.admin_attempts += 1
                        st.error("Wrong password.")

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🤖 Project Rini v1.0")
st.caption("Decoupled Non-Parametric Cognitive Architecture — Growth Marketing Expert Core")
if LOG_WEBHOOK_URL:
    st.caption(
        "🔒 Questions and answers are logged without names or IP addresses to improve Rini. "
        "Please don't enter personal or confidential information."
    )
st.markdown("---")

# chat_input at top level (pinned to the bottom) works on every Streamlit version;
# older versions reject it inside st.columns.
user_query = st.chat_input("Enter complex performance marketing query...")

left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("💬 Active Interaction Buffer")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_query:
        recent = st.session_state.chat_history[-HISTORY_TURNS * 2 :]
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.last_query = user_query
        st.session_state.last_response = None
        st.session_state.show_correction = False
        with st.chat_message("user"):
            st.write(user_query)

        # 1. Retrieval
        with st.spinner("Searching knowledge base..."):
            try:
                points = retrieve(user_query, top_k, threshold)
                st.session_state.retrieval_error = None
            except Exception as exc:
                points = []
                st.session_state.retrieval_error = str(exc)
        passages = [
            {
                "id": str(p.id),
                "source": (p.payload or {}).get("source_doc", "unknown"),
                "text": (p.payload or {}).get("text_content", ""),
                "score": p.score,
            }
            for p in points
        ]
        st.session_state.retrieved = passages

        # 2. Generation (previous turns included so follow-up questions work)
        messages = (
            [{"role": "system", "content": build_system_prompt(passages)}]
            + recent
            + [{"role": "user", "content": user_query}]
        )
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream_answer(messages))
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_response = answer
                log_event(
                    "question",
                    question=user_query,
                    answer=answer,
                    passages_used=len(passages),
                    top_score=round(passages[0]["score"], 3) if passages else "",
                    grounded="yes" if passages else "no",
                )
            except Exception as exc:
                st.error(f"Inference error: {exc}")

    # 3. Feedback
    if st.session_state.last_response and st.session_state.last_query:
        st.markdown("##### Was this response accurate to production specs?")
        up_col, down_col = st.columns(2)
        if up_col.button("👍 Good", use_container_width=True):
            st.toast("Thanks for the feedback!", icon="👍")
            log_event(
                "feedback",
                question=st.session_state.last_query,
                answer=st.session_state.last_response,
                feedback="thumbs_up",
            )
        if down_col.button("👎 Wrong / Poor", use_container_width=True):
            st.session_state.show_correction = True
            log_event(
                "feedback",
                question=st.session_state.last_query,
                answer=st.session_state.last_response,
                feedback="thumbs_down",
            )

    if st.session_state.show_correction:
        with st.form("correction_form", clear_on_submit=True):
            correction = st.text_area("What is the correct information?")
            source_url = st.text_input("Source URL (strongly recommended)")
            submitted = st.form_submit_button("Submit for review")
        if submitted:
            if correction.strip():
                try:
                    submit_correction(correction.strip(), source_url.strip())
                    log_event("correction", question=st.session_state.last_query)
                    st.session_state.show_correction = False
                    st.success("Submitted. It will be added to the knowledge base only after human review.")
                except Exception as exc:
                    st.error(f"Could not save correction: {exc}")
            else:
                st.warning("Please enter the correct information first.")

with right_col:
    st.subheader("📊 Cognitive Telemetry Monitor")

    if st.session_state.retrieval_error:
        st.error(f"Retrieval failed, so the last answer was NOT grounded: {st.session_state.retrieval_error}")
    elif st.session_state.last_query and not st.session_state.retrieved:
        st.warning("No passages above the similarity threshold. The last answer was NOT grounded in the knowledge base.")

    for i, passage in enumerate(st.session_state.retrieved, 1):
        with st.expander(f"[{i}] {passage['source']} — score {passage['score']:.3f}"):
            st.caption(f"Node ID: {passage['id']}")
            st.text(passage["text"])
