import streamlit as st

from ingestion.pdf_parser import extract_pdf_text
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_documents, embed_query

from retrieval.vectore_store import (
    add_chunks,
    get_document_count,
    get_all_chunks,
    clear_all_chunks
)

from retrieval.bm25_retriever import build_bm25_index, clear_bm25_index
from retrieval.hybrid_retriever import retrieve

from generation.prompt_builder import build_prompt
from generation.generator import generate_answer

# ── Bootstrap BM25 from persisted ChromaDB on every startup ──────────────────
all_chunks = get_all_chunks()
if all_chunks:
    build_bm25_index(all_chunks)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #0b0f1a;
    border-right: 1px solid #1a2035;
}

.stat-box {
    background: #0f1729;
    border: 1px solid #1a2a4a;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
}
.stat-box .val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4d9fff;
}
.stat-box .lbl {
    font-size: 0.7rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.1rem;
}

div[data-testid="stChatInput"] {
    border-top: 1px solid #1a2035;
}

div[data-testid="stChatMessage"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_tokens" not in st.session_state:
    st.session_state.last_tokens = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 MedRAG")
    st.caption("Medical Literature Assistant")
    st.divider()

    document_count = get_document_count()
    st.markdown(f"""
    <div class="stat-box">
        <div class="val">{document_count}</div>
        <div class="lbl">Stored Chunks</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Last response details
    if st.session_state.last_sources:
        st.markdown("**Sources Used**")
        for s in st.session_state.last_sources:
            st.caption(f"📄 {s['source']} — Page {s['page']}")
        st.divider()

    if st.session_state.last_tokens:
        st.markdown("**Last Response Tokens**")
        t = st.session_state.last_tokens
        col1, col2 = st.columns(2)
        col1.metric("Prompt", t["prompt"])
        col2.metric("Completion", t["completion"])
        st.caption(f"Total: {t['total']}")
        st.divider()

    # Clear database button
    if document_count > 0:
        st.markdown("**Database**")
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            clear_all_chunks()
            clear_bm25_index()
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.session_state.last_tokens = None
            st.success("Database cleared.")
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 🩺 MedRAG")
st.caption("Ask questions across your medical documents")
st.divider()

# Render chat history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"], avatar="🩺" if entry["role"] == "assistant" else "👤"):
        if entry["role"] == "user":
            st.write(entry["content"])
        else:
            data = entry["content"]
            st.write(data["answer"])

            confidence = data["confidence"]
            if confidence >= 70:
                st.caption(f"🟢 Confidence: {confidence}%")
            elif confidence >= 40:
                st.caption(f"🟡 Confidence: {confidence}%")
            else:
                st.caption(f"🔴 Confidence: {confidence}%")

            if data["contradictions"]:
                notice = "These sources may discuss this topic differently — worth comparing directly.\n\n"
                for c in data["contradictions"]:
                    notice += f"• {c['source_a']} (p.{c['page_a']}) ↔ {c['source_b']} (p.{c['page_b']})\n"
                notice += "\nBased on passage similarity — check original sources to confirm."
                st.info(notice)

# ── Bottom input row ──────────────────────────────────────────────────────────
upload_col, input_col = st.columns([1, 6])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        if st.button("➕ Process", use_container_width=True):
            try:
                with st.spinner("Processing..."):
                    pages = extract_pdf_text(uploaded_file)
                    chunks = chunk_pages(pages)
                    embedded_chunks = embed_documents(chunks)
                    add_chunks(embedded_chunks)
                    all_chunks = get_all_chunks()
                    build_bm25_index(all_chunks)
                st.success("Done.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))

with input_col:
    query = st.chat_input("Ask a medical question...")

# ── Handle query ──────────────────────────────────────────────────────────────
if query:
    if get_document_count() == 0:
        st.warning("Upload and process a document first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        try:
            with st.spinner("Searching..."):
                query_embedding = embed_query(query)
                retrieval_output = retrieve(query, query_embedding)
                results = retrieval_output["results"]
                confidence = retrieval_output["confidence"]
                contradictions = retrieval_output["contradictions"]

            if not results:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": {
                        "answer": "No relevant information was found in your documents.",
                        "confidence": 0.0,
                        "contradictions": [],
                    }
                })
                st.session_state.last_sources = []
                st.session_state.last_tokens = None
            else:
                prompt = build_prompt(query, results)
                generation_output = generate_answer(prompt)

                sources = [{"source": r["source"], "page": r["page"]} for r in results]
                tokens = {
                    "prompt": generation_output["prompt_tokens"],
                    "completion": generation_output["completion_tokens"],
                    "total": generation_output["total_tokens"]
                }

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": {
                        "answer": generation_output["answer"],
                        "confidence": confidence,
                        "contradictions": contradictions,
                    }
                })

                st.session_state.last_sources = sources
                st.session_state.last_tokens = tokens

        except ValueError as error:
            st.error(str(error))

        st.rerun()