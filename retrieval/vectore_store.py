import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "medical_documents"

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def add_chunks(chunks):
    if not chunks:
        return

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        ids.append(
            f"{chunk['source']}_{chunk['page']}_{chunk['chunk_id']}"
        )

        documents.append(chunk["text"])

        embeddings.append(chunk["embedding"])

        metadatas.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def similarity_search(query_embedding, top_k=5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    matches = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        matches.append(
            {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "score": 1 - distance
            }
        )

    return matches


def get_document_count():
    return collection.count()

def clear_all_chunks():
    collection.delete(where={"source": {"$ne": ""}})

def get_all_chunks():
    results = collection.get(
        include=["documents", "metadatas"]
    )

    chunks = []

    documents = results["documents"]
    metadatas = results["metadatas"]

    for document, metadata in zip(
        documents,
        metadatas
    ):
        chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"]
            }
        )

    return chunks