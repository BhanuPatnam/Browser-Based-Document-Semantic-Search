"""Hybrid retrieval combining BM25 lexical ranking and Sentence-BERT dense cosine scores."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .lexical_retrieval import BM25Retriever
from .semantic_search import cosine_similarity_matrix


def normalize_scores(scores: np.ndarray | list[float], eps: float = 1e-9) -> np.ndarray:
    """Perform min-max normalization on an array of scores into [0, 1]."""
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    s_min = float(np.min(arr))
    s_max = float(np.max(arr))
    if s_max - s_min < eps:
        # If all scores are identical, return zeros (if 0) or uniform 1.0
        return np.zeros_like(arr) if s_max == 0 else np.ones_like(arr)
    return (arr - s_min) / (s_max - s_min)


class HybridRetriever:
    """Hybrid ranker fusing lexical BM25 and dense Sentence-BERT vectors."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        dense_matrix: np.ndarray,
        bm25_retriever: BM25Retriever | None = None,
        alpha: float = 0.5,
    ) -> None:
        if not chunks:
            raise ValueError("chunks list cannot be empty.")
        if dense_matrix.ndim != 2 or len(chunks) != len(dense_matrix):
            raise ValueError("chunks count and dense_matrix rows must match.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        self.chunks = list(chunks)
        self.dense_matrix = np.asarray(dense_matrix, dtype=np.float32)
        self.alpha = float(alpha)

        if bm25_retriever is not None:
            self.bm25_retriever = bm25_retriever
        else:
            self.bm25_retriever = BM25Retriever().fit(self.chunks)

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        alpha: float | None = None,
        category: str | None = None,
        document_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search combining normalized BM25 and dense cosine similarity scores."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        active_alpha = self.alpha if alpha is None else float(alpha)
        if not (0.0 <= active_alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {active_alpha}")

        # 1. Lexical BM25 scores
        bm25_raw = self.bm25_retriever.score_all(query)
        bm25_norm = normalize_scores(bm25_raw)

        # 2. Dense Cosine similarity scores
        dense_raw = cosine_similarity_matrix(query_vector, self.dense_matrix)
        dense_norm = normalize_scores(dense_raw)

        # 3. Convex linear combination
        hybrid_scores = (active_alpha * bm25_norm) + ((1.0 - active_alpha) * dense_norm)

        candidates: list[tuple[int, float]] = []
        for index, (chunk, score) in enumerate(zip(self.chunks, hybrid_scores, strict=True)):
            if category and chunk.get("project_category") != category:
                continue
            if document_type and chunk.get("document_type") != document_type:
                continue
            candidates.append((index, float(score)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            {
                **self.chunks[index],
                "rank": rank,
                "similarity_score": score,
                "hybrid_score": score,
                "bm25_score_raw": float(bm25_raw[index]),
                "bm25_score_norm": float(bm25_norm[index]),
                "dense_score_raw": float(dense_raw[index]),
                "dense_score_norm": float(dense_norm[index]),
                "retrieval_method": "hybrid",
            }
            for rank, (index, score) in enumerate(candidates[:top_k], start=1)
        ]
