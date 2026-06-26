import streamlit as st

from ingestion.pdf_parser import extract_pdf_text
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_documents, embed_query

from retrieval.vectore_store import (
    add_chunks,
    get_document_count,
    get_all_chunks
)

from retrieval.bm25_retriever import build_bm25_index
from retrieval.hybrid_retriever import retrieve

from generation.prompt_builder import build_prompt
from generation.generator import generate_answer


all_chunks = get_all_chunks()

if all_chunks:
    build_bm25_index(all_chunks)


st.set_page_config(
    page_title="MedRAG",
    layout="wide"
)

st.title("MedRAG - Medical Literature Assistant")
st.caption("Medical Literature Assistant")


uploaded_file = st.file_uploader(
    "Upload a medical PDF",
    type=["pdf"]
)


if uploaded_file and st.button("Process Document"):
    try:
        with st.spinner("Processing document..."):
            pages = extract_pdf_text(uploaded_file)
            chunks = chunk_pages(pages)
            embedded_chunks = embed_documents(chunks)

            add_chunks(embedded_chunks)

            all_chunks = get_all_chunks()
            build_bm25_index(all_chunks)

        st.success("Document processed successfully.")

    except ValueError as error:
        st.error(str(error))


document_count = get_document_count()

st.sidebar.header("Database")
st.sidebar.write(f"Stored Chunks: {document_count}")


st.divider()


query = st.text_input("Ask a medical question")


if st.button("Generate Answer"):

    if document_count == 0:
        st.warning(
            "Please upload and process a document before asking a question."
        )

    elif not query.strip():
        st.error("Please enter a question.")

    else:
        try:
            with st.spinner("Searching documents..."):
                query_embedding = embed_query(query)

                retrieval_output = retrieve(
                    query,
                    query_embedding
                )

                results = retrieval_output["results"]
                confidence = retrieval_output["confidence"]
                contradictions = retrieval_output["contradictions"]

            if not results:
                st.info("No relevant information was found.")

            else:
                prompt = build_prompt(
                    query,
                    results
                )

                generation_output = generate_answer(prompt)

                st.subheader("Answer")
                st.write(generation_output["answer"])

                st.metric(
                    "Confidence Score",
                    f"{confidence}%"
                )


                if contradictions:
                    warning_text = (
                        "These sources may discuss this topic differently — "
                        "it is worth comparing them directly.\n\n"
                    )

                    for item in contradictions:
                        warning_text += (
                            f"• {item['source_a']} "
                            f"(Page {item['page_a']}) ↔ "
                            f"{item['source_b']} "
                            f"(Page {item['page_b']})\n"
                        )

                    warning_text += (
                        "\nThis is based on how differently these passages "
                        "discuss your query — check the original sources "
                        "to confirm."
                    )

                    st.info(warning_text)


                with st.expander("Sources Used"):
                    for result in results:
                        st.write(
                            f"{result['source']} "
                            f"(Page {result['page']})"
                        )


                with st.expander("Token Usage"):
                    st.write(
                        f"Prompt Tokens: {generation_output['prompt_tokens']}"
                    )
                    st.write(
                        f"Completion Tokens: {generation_output['completion_tokens']}"
                    )
                    st.write(
                        f"Total Tokens: {generation_output['total_tokens']}"
                    )

        except ValueError as error:
            st.error(str(error))