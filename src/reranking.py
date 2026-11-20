"""Two-stage retrieval: Cross-Encoder neural reranker for dense candidate shortlists."""

from __future__ import annotations

from typing import Any

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Second-stage neural reranker scoring query-document pairs simultaneously."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load CrossEncoder model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank stage-1 candidates using Cross-Encoder joint self-attention."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not candidates:
            return []
        if not query or not query.strip():
            return candidates[:top_k]

        pairs = [(query, str(cand.get("text", ""))) for cand in candidates]
        scores = self.model.predict(pairs)

        # Build reranked entries preserving stage-1 metadata
        reranked_items: list[tuple[float, dict[str, Any]]] = []
        for cand, score in zip(candidates, scores, strict=True):
            dense_score = cand.get("similarity_score", cand.get("dense_score_raw", 0.0))
            item = dict(cand)
            item["initial_rank"] = cand.get("rank", 1)
            item["dense_score"] = float(dense_score)
            item["rerank_score"] = float(score)
            item["retrieval_method"] = "cross_encoder_rerank"
            reranked_items.append((float(score), item))

        reranked_items.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, (_score, item) in enumerate(reranked_items[:top_k], start=1):
            item["rank"] = rank
            # For consistent display, set similarity_score to rerank_score
            item["similarity_score"] = item["rerank_score"]
            results.append(item)

        return results
