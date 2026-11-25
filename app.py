import os
import uuid

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


def get_secret(name: str) -> str:
    """Streamlit secrets first, environment variable as fallback (handy for local runs)."""
    try:
        return str(st.secrets[name])
    except Exception:
        return os.environ.get(name, "")


QDRANT_ENDPOINT = get_secret("QDRANT_API_ENDPOINT")
QDRANT_KEY = get_secret("QDRANT_API_KEY")
GROQ_KEY = get_secret("GROQ_API_KEY")

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
    return client


try:
    qdrant = load_qdrant(QDRANT_ENDPOINT, QDRANT_KEY)
except Exception as exc:
    st.error(f"Could not connect to Qdrant: {exc}")
    st.stop()

embedder = load_embedder()
groq_client = Groq(api_key=GROQ_KEY)


# ---------------------------------------------------------------------------
# Core functions
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
            st.warning("Knowledge base is almost empty, so most answers will be ungrounded. Run seed_knowledge_base.py.")
    except Exception as exc:
        st.error(f"Could not read knowledge base size: {exc}")
    if st.button("Clear conversation", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🤖 Project Rini v1.0")
st.caption("Decoupled Non-Parametric Cognitive Architecture — Growth Marketing Expert Core")
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
            except Exception as exc:
                st.error(f"Inference error: {exc}")

    # 3. Feedback
    if st.session_state.last_response and st.session_state.last_query:
        st.markdown("##### Was this response accurate to production specs?")
        up_col, down_col = st.columns(2)
        if up_col.button("👍 Good", use_container_width=True):
            st.toast("Thanks for the feedback!", icon="👍")
        if down_col.button("👎 Wrong / Poor", use_container_width=True):
            st.session_state.show_correction = True

    if st.session_state.show_correction:
        with st.form("correction_form", clear_on_submit=True):
            correction = st.text_area("What is the correct information?")
            source_url = st.text_input("Source URL (strongly recommended)")
            submitted = st.form_submit_button("Submit for review")
        if submitted:
            if correction.strip():
                try:
                    submit_correction(correction.strip(), source_url.strip())
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
