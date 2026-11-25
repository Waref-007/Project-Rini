import streamlit as st
import uuid
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from groq import Groq
from sentence_transformers import SentenceTransformer

# Enforce clean page layouts mapping to frontier platform architectures
st.set_page_config(page_title="Project Rini v1.0", page_icon="🤖", layout="wide")

@st.cache_resource
def bootstrap_local_embedding_engine():
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

@st.cache_resource
def bootstrap_cloud_infrastructure(_endpoint, _api_key):
    return QdrantClient(url=_endpoint, api_key=_api_key)

# Bind security contexts cleanly out-of-band from the cached runtime secrets vault
qdrant_endpoint = st.secrets.get("QDRANT_API_ENDPOINT", "")
qdrant_key = st.secrets.get("QDRANT_API_KEY", "")
groq_key = st.secrets.get("GROQ_API_KEY", "")

# App initialization safety handshake check
if not qdrant_endpoint or not qdrant_key or not groq_key:
    st.error("🔒 Critical Error: Missing system environment credentials inside Secrets management.")
    st.stop()

# Instantiate core engine layers
embedder = bootstrap_local_embedding_engine()
qdrant_client = bootstrap_cloud_infrastructure(qdrant_endpoint, qdrant_key)
groq_client = Groq(api_key=groq_key)

st.title("🤖 Project Rini v1.0")
st.caption("Decoupled Non-Parametric Cognitive Architecture — Growth Marketing Expert Core")
st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = ""

left_col, right_col = st.columns()

with left_col:
    st.subheader("💬 Active Interaction Buffer")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    user_query = st.chat_input("Enter complex performance marketing query...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
            
        # 1. Cloud Search Subspace Exploration via dense vectors
        query_vector = embedder.encode(f"Represent this sentence for searching relevant passages: {user_query}").tolist()
        
        try:
            search_results = qdrant_client.query_points(
                collection_name="rini_growth_marketing",
                query_vector=query_vector,
                limit=2
            ).points
            
            context_fragments = [hit.payload.get("text_content", "") for hit in search_results if hit.payload]
            st.session_state.retrieved_context = "\n\n".join(context_fragments) if context_fragments else "No relevant non-parametric vectors located."
        except Exception as e:
            st.session_state.retrieved_context = f"Database read exception: {str(e)}"
            
        # 2. Cloud Inference Pipeline Generation
        system_instruction = (
            "You are Rini, an elite expert marketing cognitive agent. "
            "Synthesize a deep, precise, and highly technical answer based strictly on the provided factual context blocks.\n\n"
            "FOUNDATIONAL DOMAIN CONTEXT:\n"
        ) + st.session_state.retrieved_context
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.1
                )
                assistant_response = completion.choices.message.content
                response_placeholder.write(assistant_response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                st.session_state.last_response = assistant_response
            except Exception as e:
                st.error(f"Inference crash: {str(e)}")

    # 3. Interactive Alignment Feedback Loops
    if st.session_state.last_response:
        st.markdown("##### Was this response accurate to production specs?")
        f_left, f_right = st.columns()
        
        if f_left.button("👍 Good"):
            st.toast("✅ Alignment feedback logged!", icon="🔥")
            
        if f_right.button("👎 Hallucination / Poor"):
            st.toast("⚠️ Negative alignment logged. Running reflection...", icon="🧠")
            dialogue_snippet = f"USER: {st.session_state.chat_history[-2]['content']}\nASSISTANT: {st.session_state.last_response}"
            reflection_prompt = (
                "Analyze the following conversation failure. Extract a corrected, explicit, clean factual assertion "
                "statement summarizing the true platform mechanic to patch the knowledge base.\n"
                "Output ONLY a valid JSON payload matching this format: {\"assertions\": [\"Corrected Fact\"]}\n\n"
                f"TRANSCRIPT:\n{dialogue_snippet}"
            )
            try:
                raw_ref = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    temperature=0.1
                ).choices.message.content
                s_idx = raw_ref.find('{')
                e_idx = raw_ref.rfind('}') + 1
                parsed_facts = json.loads(raw_ref[s_idx:e_idx]).get("assertions", [])
                for fact in parsed_facts:
                    f_vector = embedder.encode(fact).tolist()
                    qdrant_client.upsert(
                        collection_name="rini_growth_marketing",
                        points=[models.PointStruct(id=int(uuid.uuid4().int & 0xFFFFFFFF), vector=f_vector, payload={"source_doc": "user_critique_alignment.md", "text_content": fact})]
                    )
                st.success("🧠 Non-Parametric Memory Space Self-Evolved!")
            except Exception as ex:
                st.sidebar.error(f"Reflection loop error: {str(ex)}")

with right_col:
    st.subheader("📊 Cognitive Telemetry Monitor")
    st.text_area("Retrieval Logs Context", value=st.session_state.retrieved_context, height=400, disabled=True, label_visibility="collapsed")
