"""Comprehensive Information Retrieval (IR) metrics for semantic and lexical search.

Calculates:
  - Precision@K
  - Recall@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG@K)
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Callable


def _is_relevant(result: dict[str, Any], query: dict[str, Any]) -> bool:
    """Check if a retrieved result matches any labeled relevant document or chunk ID."""
    relevant_documents = set(query.get("relevant_document_ids", []))
    relevant_chunks = set(query.get("relevant_chunk_ids", []))
    return (
        result.get("document_id") in relevant_documents
        or result.get("chunk_id") in relevant_chunks
    )


def precision_at_k(results: list[dict[str, Any]], query: dict[str, Any], k: int) -> float:
    """Compute Precision@K: fraction of top-K results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not results:
        return 0.0
    top_k = results[:k]
    relevant_count = sum(1 for item in top_k if _is_relevant(item, query))
    return float(relevant_count / k)


def hit_at_k(results: list[dict[str, Any]], query: dict[str, Any], k: int) -> float:
    """Binary hit indicator: 1.0 if at least one relevant passage is in top-K, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive.")
    return float(any(_is_relevant(item, query) for item in results[:k]))


def recall_at_k(results: list[dict[str, Any]], query: dict[str, Any], k: int) -> float:
    """Compute Recall@K: fraction of all relevant documents that appear in the top-K.

    If target relevant documents are specified, computes unique relevant documents retrieved / total relevant documents.
    If total relevant documents is 1 or unspecified, this equals Hit@K.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    target_docs = set(query.get("relevant_document_ids", []))
    if not target_docs:
        # Fall back to checking relevant chunks or hit
        target_chunks = set(query.get("relevant_chunk_ids", []))
        if target_chunks:
            retrieved_chunks = {item.get("chunk_id") for item in results[:k] if item.get("chunk_id") in target_chunks}
            return float(len(retrieved_chunks) / len(target_chunks))
        return hit_at_k(results, query, k)

    retrieved_relevant_docs = {
        item.get("document_id") for item in results[:k] if item.get("document_id") in target_docs
    }
    return float(len(retrieved_relevant_docs) / len(target_docs))


def reciprocal_rank(results: list[dict[str, Any]], query: dict[str, Any]) -> float:
    """Compute Reciprocal Rank: 1 / rank of the first relevant result (0.0 if none)."""
    for rank, item in enumerate(results, start=1):
        if _is_relevant(item, query):
            return 1.0 / rank
    return 0.0


def dcg_at_k(results: list[dict[str, Any]], query: dict[str, Any], k: int) -> float:
    """Compute Discounted Cumulative Gain at K with binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive.")
    dcg = 0.0
    for i, item in enumerate(results[:k], start=1):
        rel = 1.0 if _is_relevant(item, query) else 0.0
        dcg += rel / math.log2(i + 1)
    return dcg


def idcg_at_k(query: dict[str, Any], k: int) -> float:
    """Compute Ideal Discounted Cumulative Gain at K."""
    if k <= 0:
        raise ValueError("k must be positive.")
    n_relevant = len(query.get("relevant_document_ids", [])) or len(query.get("relevant_chunk_ids", [])) or 1
    ideal_hits = min(k, n_relevant)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))


def ndcg_at_k(results: list[dict[str, Any]], query: dict[str, Any], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K)."""
    idcg = idcg_at_k(query, k)
    if idcg <= 0.0:
        return 0.0
    return float(min(1.0, dcg_at_k(results, query, k) / idcg))


def evaluate_retriever(
    queries: list[dict[str, Any]],
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
    method_name: str = "retriever",
) -> dict[str, Any]:
    """Evaluate a search function across queries and standard IR metrics."""
    if not queries:
        return {"query_count": 0, "mrr": 0.0}

    max_k = max(ks)
    per_query: list[dict[str, Any]] = []

    for q in queries:
        q_text = str(q.get("query", ""))
        results = search_fn(q_text, max_k)
        rr = reciprocal_rank(results, q)

        row: dict[str, Any] = {
            "query": q_text,
            "reciprocal_rank": rr,
            "top_document_ids": [item.get("document_id") for item in results],
        }

        for k in ks:
            row[f"precision_at_{k}"] = precision_at_k(results, q, k)
            row[f"recall_at_{k}"] = recall_at_k(results, q, k)
            row[f"hit_at_{k}"] = hit_at_k(results, q, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(results, q, k)

        per_query.append(row)

    summary: dict[str, Any] = {
        "method": method_name,
        "query_count": len(queries),
        "mrr": mean(item["reciprocal_rank"] for item in per_query),
    }

    for k in ks:
        summary[f"precision_at_{k}"] = mean(item[f"precision_at_{k}"] for item in per_query)
        summary[f"recall_at_{k}"] = mean(item[f"recall_at_{k}"] for item in per_query)
        summary[f"hit_at_{k}"] = mean(item[f"hit_at_{k}"] for item in per_query)
        summary[f"ndcg_at_{k}"] = mean(item[f"ndcg_at_{k}"] for item in per_query)

    summary["per_query"] = per_query
    return summary
