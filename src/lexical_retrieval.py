"""Lexical retrieval baselines: TF-IDF with Cosine Similarity and BM25Okapi."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_@.][A-Za-z0-9]+)*")


def tokenize_technical(text: str) -> list[str]:
    """Tokenize text preserving technical terms, model names, hyphenations, and metrics.

    Examples preserved:
      'all-MiniLM-L6-v2' -> 'all-minilm-l6-v2'
      'Recall@1' -> 'recall@1'
      'U-Net' -> 'u-net'
      'Sentence-BERT' -> 'sentence-bert'
    """
    if not isinstance(text, str):
        return []
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if len(t) > 1 or t.isalnum()]


class TFIDFRetriever:
    """TF-IDF vector space retriever with Cosine Similarity."""

    def __init__(self, max_features: int | None = 5000) -> None:
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_technical,
            token_pattern=None,
            lowercase=True,
            max_features=max_features,
            norm="l2",
        )
        self.chunks: list[dict[str, Any]] = []
        self.tfidf_matrix: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, chunks: list[dict[str, Any]]) -> TFIDFRetriever:
        if not chunks:
            raise ValueError("Cannot fit TFIDFRetriever with an empty chunk list.")
        self.chunks = list(chunks)
        corpus_texts = [str(chunk.get("text", "")) for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts).toarray().astype(np.float32)
        self._is_fitted = True
        return self

    def score_all(self, query: str) -> np.ndarray:
        """Return cosine similarity scores against all chunks."""
        if not self._is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("TFIDFRetriever is not fitted. Call fit() first.")
        if not query or not query.strip():
            return np.zeros(len(self.chunks), dtype=np.float32)

        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32).reshape(-1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(self.chunks), dtype=np.float32)

        doc_norms = np.linalg.norm(self.tfidf_matrix, axis=1)
        denominator = np.maximum(query_norm * doc_norms, 1e-12)
        return (self.tfidf_matrix @ query_vec) / denominator

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        document_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not self._is_fitted:
            raise RuntimeError("TFIDFRetriever is not fitted. Call fit() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query or not query.strip():
            return []

        scores = self.score_all(query)
        candidates: list[tuple[int, float]] = []
        for index, (chunk, score) in enumerate(zip(self.chunks, scores, strict=True)):
            if score <= min_score:
                continue
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
                "retrieval_method": "tfidf",
            }
            for rank, (index, score) in enumerate(candidates[:top_k], start=1)
        ]


class BM25Retriever:
    """BM25 (BM25Okapi) lexical ranking model."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[dict[str, Any]] = []
        self.bm25: BM25Okapi | None = None
        self._is_fitted = False

    def fit(self, chunks: list[dict[str, Any]]) -> BM25Retriever:
        if not chunks:
            raise ValueError("Cannot fit BM25Retriever with an empty chunk list.")
        self.chunks = list(chunks)
        tokenized_corpus = [tokenize_technical(str(c.get("text", ""))) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._is_fitted = True
        return self

    def score_all(self, query: str) -> np.ndarray:
        """Return raw BM25 score array for all indexed chunks."""
        if not self._is_fitted or self.bm25 is None:
            raise RuntimeError("BM25Retriever is not fitted. Call fit() first.")
        query_tokens = tokenize_technical(query)
        if not query_tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        document_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        if not self._is_fitted or self.bm25 is None:
            raise RuntimeError("BM25Retriever is not fitted. Call fit() first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not query or not query.strip():
            return []

        scores = self.score_all(query)
        candidates: list[tuple[int, float]] = []
        for index, (chunk, score) in enumerate(zip(self.chunks, scores, strict=True)):
            if score <= min_score:
                continue
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
                "retrieval_method": "bm25",
            }
            for rank, (index, score) in enumerate(candidates[:top_k], start=1)
        ]
