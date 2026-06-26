from sentence_transformers import CrossEncoder

from retrieval.bm25_retriever import bm25_search
from retrieval.vectore_store import similarity_search

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)
CONTRADICTION_THRESHOLD = 0.3

def merge_results(vector_results, bm25_results):
    merged = {}

    for result in vector_results + bm25_results:
        key = (
            result["source"],
            result["page"],
            result["chunk_id"]
        )

        if key not in merged:
            merged[key] = result

    return list(merged.values())


def rerank_results(query, results):
    if not results:
        return []

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        item = result.copy()

        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked


def calculate_confidence(results):
    if not results:
        return 0.0

    best_score = results[0]["rerank_score"]

    confidence = 1 / (1 + pow(2.71828, -best_score))

    return round(confidence * 100, 2)


def detect_contradictions(results):
    if len(results) < 2:
        return []

    contradictions = []

    try:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                if results[i]["source"] == results[j]["source"]:
                    continue

                similarity_score = float(
                    reranker.predict(
                        [
                            (
                                results[i]["text"],
                                results[j]["text"]
                            )
                        ]
                    )[0]
                )

                if similarity_score < CONTRADICTION_THRESHOLD:
                    contradictions.append(
                        {
                            "source_a": results[i]["source"],
                            "source_b": results[j]["source"],
                            "page_a": results[i]["page"],
                            "page_b": results[j]["page"],
                            "similarity_score": similarity_score
                        }
                    )

    except Exception as error:
        print(
            f"Contradiction detection error"
        )
        return []

    return contradictions


def retrieve(query, query_embedding, top_k=3):
    vector_results = similarity_search(
        query_embedding,
        top_k=10
    )

    bm25_results = bm25_search(
        query,
        top_k=10
    )

    merged_results = merge_results(
        vector_results,
        bm25_results
    )

    reranked_results = rerank_results(
        query,
        merged_results
    )

    confidence = calculate_confidence(
        reranked_results
    )

    contradictions = detect_contradictions(
        reranked_results[:top_k]
    )

    return {
        "results": reranked_results[:top_k],
        "confidence": confidence,
        "contradictions": contradictions
    }