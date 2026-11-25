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
    # Cache local vector embedding extraction weights safely on CPU
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

@st.cache_resource
def bootstrap_cloud_infrastructure(_endpoint, _api_key):
    # Establish connection persistence to cloud vector cluster
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

# Session state initialization hooks
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = ""

# Fixed empty columns crash by specifying width allocation (2 columns)
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("💬 Active Interaction Buffer")
    
    # Render static log history entries cleanly
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    user_query = st.chat_input("Enter complex performance marketing query...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.last_query = user_query
        with st.chat_message("user"):
            st.write(user_query)
            
        # 1. Cloud Search Subspace Exploration via dense vectors
        with st.spinner("Traversing non-parametric vector space..."):
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
            
        # 2. Cloud Inference Pipeline Generation with Streaming Capabilities
        system_instruction = (
            "You are Rini, an elite expert marketing cognitive agent. "
            "Synthesize a deep, precise, and highly technical answer based strictly on the provided factual context blocks.\n\n"
            "FOUNDATIONAL DOMAIN CONTEXT:\n"
        ) + st.session_state.retrieved_context
        
        with st.chat_message("assistant"):
            try:
                # Optimized pipeline to fetch from standard high-throughput production models
                completion_stream = groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.1,
                    stream=True
                )
                # Stream token production chunks smoothly for superior human-centric UI latency response
                assistant_response = st.write_stream(completion_stream)
                
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                st.session_state.last_response = assistant_response
                st.rerun() # Refresh layout context trees immediately
            except Exception as e:
                st.error(f"Inference crash: {str(e)}")

    # 3. Interactive Alignment Feedback Loops
    if st.session_state.last_response and st.session_state.last_query:
        st.markdown("##### Was this response accurate to production specs?")
        # Fixed layout nesting alignment problem
        f_left, f_right = st.columns(2)
        
        if f_left.button("👍 Good", use_container_width=True):
            st.toast("✅ Alignment feedback logged!", icon="🔥")
            
        if f_right.button("👎 Hallucination / Poor", use_container_width=True):
            st.toast("⚠️ Negative alignment logged. Running reflection...", icon="🧠")
            
            # Isolated message retrieval structures cleanly from dynamic shifting history indices
            dialogue_snippet = f"USER: {st.session_state.last_query}\nASSISTANT: {st.session_state.last_response}"
            reflection_prompt = (
                "Analyze the following conversation failure. Extract a corrected, explicit, clean factual assertion "
                "statement summarizing the true platform mechanic to patch the knowledge base.\n"
                "Output ONLY a valid JSON payload matching this format: {\"assertions\": [\"Corrected Fact\"]}\n\n"
                f"TRANSCRIPT:\n{dialogue_snippet}"
            )
            try:
                # Swapped to faster analytical instance configuration profiles for backend data transformation checks
                raw_ref = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                ).choices.message.content
                
                parsed_facts = json.loads(raw_ref).get("assertions", [])
                for fact in parsed_facts:
                    f_vector = embedder.encode(fact).tolist()
                    qdrant_client.upsert(
                        collection_name="rini_growth_marketing",
                        points=[models.PointStruct(
                            id=int(uuid.uuid4().int & 0xFFFFFFFF), 
                            vector=f_vector, 
                            payload={"source_doc": "user_critique_alignment.md", "text_content": fact}
                        )]
                    )
                st.success("🧠 Non-Parametric Memory Space Self-Evolved!")
            except Exception as ex:
                st.sidebar.error(f"Reflection loop error: {str(ex)}")

with right_col:
    st.subheader("📊 Cognitive Telemetry Monitor")
    st.text_area(
        "Retrieval Logs Context", 
        value=st.session_state.retrieved_context, 
        height=500, 
        disabled=True, 
        label_visibility="collapsed"
    )
