import re

from rank_bm25 import BM25Okapi


bm25_index = None
chunk_store = []


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks):
    global bm25_index
    global chunk_store

    if not chunks:
        bm25_index = None
        chunk_store = []
        return

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25_index = BM25Okapi(tokenized_chunks)

    chunk_store = chunks


def bm25_search(query, top_k=5):
    if bm25_index is None:
        return []

    tokenized_query = tokenize(query)

    scores = bm25_index.get_scores(tokenized_query)

    ranked_results = sorted(
        zip(chunk_store, scores),
        key=lambda item: item[1],
        reverse=True
    )

    matches = []

    for chunk, score in ranked_results[:top_k]:
        matches.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "score": float(score)
            }
        )

    return matches