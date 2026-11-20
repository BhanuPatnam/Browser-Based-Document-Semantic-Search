"""Cosine-similarity semantic search for offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def cosine_similarity_matrix(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    docs = np.asarray(matrix, dtype=np.float32)
    if docs.ndim != 2:
        raise ValueError("matrix must be a two-dimensional array")
    if docs.shape[1] != query.size:
        raise ValueError(
            f"Vector dimension mismatch: query has {query.size}, index has {docs.shape[1]}"
        )
    if not np.isfinite(query).all() or not np.isfinite(docs).all():
        raise ValueError("query vector and index matrix must contain only finite values")
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    return (docs @ query) / denominator


def load_index(chunks_path: Path, embeddings_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    payload = json.loads(embeddings_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not all(isinstance(chunk, dict) for chunk in chunks):
        raise ValueError("Chunk file must contain a list of chunk objects")
    if not chunks:
        raise ValueError("Chunk file contains no chunks")
    if not isinstance(payload, dict):
        raise ValueError("Embedding file must contain an object")
    if payload.get("status") != "ready" or not payload.get("embeddings"):
        raise ValueError("Embedding file is not ready. Run scripts/generate_embeddings.py first.")

    embeddings = payload["embeddings"]
    if not isinstance(embeddings, list):
        raise ValueError("Embedding payload must contain a list of embeddings")
    try:
        vector_by_id = {item["chunk_id"]: item["vector"] for item in embeddings}
    except (KeyError, TypeError) as exc:
        raise ValueError("Each embedding must contain chunk_id and vector fields") from exc
    try:
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    except KeyError as exc:
        raise ValueError("Each chunk must contain a chunk_id field") from exc
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in vector_by_id]
    if missing:
        raise ValueError(f"Missing embeddings for {len(missing)} chunks")
    try:
        matrix = np.asarray([vector_by_id[chunk_id] for chunk_id in chunk_ids], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding vectors must be numeric and have a consistent dimension") from exc
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Embedding vectors must be non-empty and have a consistent dimension")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must contain only finite values")
    model_name = payload.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Embedding payload must include a model_name")
    return chunks, matrix, model_name


def search_with_vector(
    query_vector: np.ndarray,
    chunks: list[dict[str, Any]],
    matrix: np.ndarray,
    top_k: int = 5,
    category: str | None = None,
    document_type: str | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(chunks) != len(matrix):
        raise ValueError("chunks and matrix must have the same number of rows")
    scores = cosine_similarity_matrix(query_vector, matrix)
    candidates: list[tuple[int, float]] = []
    for index, (chunk, score) in enumerate(zip(chunks, scores, strict=True)):
        if category and chunk.get("project_category") != category:
            continue
        if document_type and chunk.get("document_type") != document_type:
            continue
        candidates.append((index, float(score)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        {**chunks[index], "rank": rank, "similarity_score": score}
        for rank, (index, score) in enumerate(candidates[:top_k], start=1)
    ]
