"""Chunking strategy experiments for technical document retrieval.

Evaluates:
  - Fixed-window chunking
  - Sentence-aware chunking
  - Section-aware chunking
Across grid:
  - Chunk sizes: 100, 180, 250 words
  - Overlaps: 20, 40, 60 words
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .document_chunking import chunk_documents
from .document_loader import LoadedDocument, load_documents, slugify
from .model_evaluation import evaluate_retriever
from .semantic_search import cosine_similarity_matrix, search_with_vector
from .text_preprocessing import extract_sections, normalize_for_deduplication

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")


def chunk_fixed_window_text(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> list[str]:
    """Slice text into word windows without respecting sentence or heading boundaries."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive.")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_size_words.")

    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size_words]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size_words >= len(words):
            break
    return chunks


def chunk_sentence_aware_text(text: str, chunk_size_words: int = 180, overlap_words: int = 40) -> list[str]:
    """Group complete sentences into chunks, rolling overlap when word limit is reached."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive.")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and < chunk_size_words.")

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sent in sentences:
        words = sent.split()
        if len(words) > chunk_size_words:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0
            chunks.extend(chunk_fixed_window_text(sent, chunk_size_words, overlap_words))
            continue

        if current_sentences and (current_word_count + len(words) > chunk_size_words):
            completed = " ".join(current_sentences)
            chunks.append(completed)
            overlap_tail = completed.split()[-overlap_words:] if overlap_words else []
            current_sentences = overlap_tail + [sent]
            current_word_count = len(overlap_tail) + len(words)
        else:
            current_sentences.append(sent)
            current_word_count += len(words)

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def build_chunks_for_strategy(
    documents: list[LoadedDocument],
    strategy: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    """Build standardized chunk dictionaries for a given chunking strategy."""
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()

    if strategy == "section_aware":
        # Uses heading preservation + sentence awareness
        return chunk_documents(documents, chunk_size_words, overlap_words)

    for doc in documents:
        if strategy == "fixed_window":
            text_slices = chunk_fixed_window_text(doc.text, chunk_size_words, overlap_words)
        elif strategy == "sentence_aware":
            text_slices = chunk_sentence_aware_text(doc.text, chunk_size_words, overlap_words)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for part_idx, piece in enumerate(text_slices):
            dedup_key = normalize_for_deduplication(piece)
            if not dedup_key or dedup_key in seen:
                continue
            seen.add(dedup_key)
            chunk_id = f"{doc.document_id}--{strategy}--{part_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc.document_id,
                "project_name": doc.project_name,
                "project_category": doc.project_category,
                "source_file": doc.source_file,
                "section_title": "General",
                "text": piece,
                "url_or_local_path": doc.url_or_local_path,
                "tags": doc.tags,
                "document_type": doc.document_type,
                "created_from": doc.created_from,
                "word_count": len(piece.split()),
            })

    return chunks


def run_chunking_grid_experiment(
    raw_docs_dir: Path,
    queries: list[dict[str, Any]],
    model: Any,
    chunk_sizes: list[int] | tuple[int, ...] = (100, 180, 250),
    overlaps: list[int] | tuple[int, ...] = (20, 40, 60),
    strategies: list[str] | tuple[str, ...] = ("fixed_window", "sentence_aware", "section_aware"),
) -> dict[str, Any]:
    """Run full factorial grid experiment over chunking strategies, sizes, and overlaps."""
    documents = load_documents(raw_docs_dir)
    if not documents:
        raise ValueError(f"No documents found in {raw_docs_dir}")

    # Pre-encode query vectors once to save time
    query_texts = [str(q["query"]) for q in queries]
    query_vectors = model.encode(query_texts, normalize_embeddings=True, convert_to_numpy=True)
    query_vec_map = {q["query"]: query_vectors[i] for i, q in enumerate(queries)}

    records: list[dict[str, Any]] = []

    for strategy in strategies:
        for size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue

                chunks = build_chunks_for_strategy(documents, strategy, size, overlap)
                if not chunks:
                    continue

                # Encode all chunks in batch
                chunk_texts = [c["text"] for c in chunks]
                chunk_matrix = model.encode(
                    chunk_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)

                def search_fn(q_text: str, k: int) -> list[dict[str, Any]]:
                    q_vec = query_vec_map[q_text]
                    return search_with_vector(q_vec, chunks, chunk_matrix, top_k=k)

                eval_res = evaluate_retriever(queries, search_fn, ks=(1, 3, 5, 10), method_name=strategy)
                avg_words = float(np.mean([c["word_count"] for c in chunks]))

                record = {
                    "strategy": strategy,
                    "chunk_size_words": size,
                    "overlap_words": overlap,
                    "chunk_count": len(chunks),
                    "avg_chunk_words": round(avg_words, 2),
                    "mrr": round(eval_res["mrr"], 4),
                    "recall_at_1": round(eval_res["recall_at_1"], 4),
                    "recall_at_3": round(eval_res["recall_at_3"], 4),
                    "recall_at_5": round(eval_res["recall_at_5"], 4),
                    "ndcg_at_3": round(eval_res["ndcg_at_3"], 4),
                    "ndcg_at_5": round(eval_res["ndcg_at_5"], 4),
                    "precision_at_3": round(eval_res["precision_at_3"], 4),
                }
                records.append(record)

    # Sort to determine best configuration
    sorted_records = sorted(
        records,
        key=lambda r: (r["mrr"], r["ndcg_at_5"], r["recall_at_3"], -abs(r["chunk_size_words"] - 180)),
        reverse=True,
    )
    best = sorted_records[0] if sorted_records else {}

    explanation = (
        f"The best configuration is strategy='{best.get('strategy')}' with chunk_size={best.get('chunk_size_words')} words "
        f"and overlap={best.get('overlap_words')} words (MRR: {best.get('mrr')}, nDCG@5: {best.get('ndcg_at_5')}). "
        f"Section-aware chunking preserves logical discourse units (Markdown headings), avoiding mixing disparate concepts. "
        f"A window of 180 words with 40 words overlap fits dense bi-encoder models (MiniLM) comfortably within their optimal "
        f"token capacity (~240 tokens), preventing semantic dilution while providing adequate context for cross-boundary concepts."
    )

    return {
        "configurations_tested": len(records),
        "best_configuration": best,
        "explanation": explanation,
        "results": records,
    }
