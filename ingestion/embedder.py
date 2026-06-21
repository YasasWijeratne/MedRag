from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def embed_documents(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = chunk.copy()

        embedded_chunk["embedding"] = embedding.tolist()

        embedded_chunks.append(embedded_chunk)

    return embedded_chunks


def embed_query(query):
    query_embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True
    )

    return query_embedding.tolist()