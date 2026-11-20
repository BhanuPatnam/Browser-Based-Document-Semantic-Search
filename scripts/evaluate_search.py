#!/usr/bin/env python
"""Comprehensive Information Retrieval benchmark across lexical, dense, hybrid, and reranked methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieval import HybridRetriever
from src.lexical_retrieval import BM25Retriever, TFIDFRetriever
from src.model_evaluation import evaluate_retriever
from src.reranking import CrossEncoderReranker
from src.semantic_search import load_index, search_with_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data/processed/document_chunks.json")
    parser.add_argument("--embeddings", type=Path, default=PROJECT_ROOT / "data/processed/embeddings.json")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data/processed/evaluation_queries.json")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--skip-reranking", action="store_true", help="Skip Cross-Encoder reranking")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {args.chunks} and embeddings from {args.embeddings}...")
    chunks, matrix, model_name = load_index(args.chunks, args.embeddings)
    queries: list[dict[str, Any]] = json.loads(args.queries.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks and {len(queries)} evaluation queries.")

    print(f"Loading dense bi-encoder model '{model_name}'...")
    bi_encoder = SentenceTransformer(model_name)

    # Pre-encode queries
    q_texts = [q["query"] for q in queries]
    q_vectors = bi_encoder.encode(q_texts, normalize_embeddings=True, convert_to_numpy=True)
    q_vec_map = {q["query"]: q_vectors[i] for i, q in enumerate(queries)}

    # 1. Initialize Lexical Retrievers
    print("Fitting TF-IDF retriever...")
    tfidf_retriever = TFIDFRetriever().fit(chunks)

    print("Fitting BM25 retriever...")
    bm25_retriever = BM25Retriever().fit(chunks)

    # 2. Initialize Hybrid Retriever
    print("Initializing Hybrid retriever (alpha=0.5)...")
    hybrid_retriever = HybridRetriever(chunks, matrix, bm25_retriever=bm25_retriever, alpha=0.5)

    # 3. Initialize Cross-Encoder Reranker if not skipped
    cross_encoder = None
    if not args.skip_reranking:
        print("Loading CrossEncoder reranker 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
        try:
            cross_encoder = CrossEncoderReranker()
            _ = cross_encoder.model  # Trigger load
        except Exception as exc:
            print(f"Warning: Could not load CrossEncoder ({exc}). Skipping reranking.")
            cross_encoder = None

    # Search functions for each method
    def search_tfidf(q: str, k: int) -> list[dict[str, Any]]:
        return tfidf_retriever.search(q, top_k=k)

    def search_bm25(q: str, k: int) -> list[dict[str, Any]]:
        return bm25_retriever.search(q, top_k=k)

    def search_dense(q: str, k: int) -> list[dict[str, Any]]:
        q_vec = q_vec_map[q]
        return search_with_vector(q_vec, chunks, matrix, top_k=k)

    def search_hybrid(q: str, k: int) -> list[dict[str, Any]]:
        q_vec = q_vec_map[q]
        return hybrid_retriever.search(q, q_vec, top_k=k, alpha=0.5)

    def search_rerank(q: str, k: int) -> list[dict[str, Any]]:
        if cross_encoder is None:
            return search_dense(q, k)
        # Stage 1: Dense retrieve top 10
        candidates = search_dense(q, max(10, k))
        # Stage 2: CrossEncoder rerank
        return cross_encoder.rerank(q, candidates, top_k=k)

    methods = [
        ("TF-IDF", search_tfidf),
        ("BM25", search_bm25),
        ("Sentence-BERT Dense", search_dense),
        ("Hybrid (BM25 + Dense, a=0.5)", search_hybrid),
    ]
    if cross_encoder is not None:
        methods.append(("Dense + Cross-Encoder Rerank", search_rerank))

    comparison_rows: list[dict[str, Any]] = []
    full_reports: dict[str, Any] = {}

    print("\nRunning evaluation across all retrieval methods...")
    for name, s_fn in methods:
        eval_res = evaluate_retriever(queries, s_fn, ks=(1, 3, 5, 10), method_name=name)
        full_reports[name] = eval_res
        row = {
            "Method": name,
            "MRR": round(eval_res["mrr"], 4),
            "Recall@1": round(eval_res["recall_at_1"], 4),
            "Recall@3": round(eval_res["recall_at_3"], 4),
            "Recall@5": round(eval_res["recall_at_5"], 4),
            "Recall@10": round(eval_res["recall_at_10"], 4),
            "Precision@1": round(eval_res["precision_at_1"], 4),
            "Precision@3": round(eval_res["precision_at_3"], 4),
            "Precision@5": round(eval_res["precision_at_5"], 4),
            "nDCG@3": round(eval_res["ndcg_at_3"], 4),
            "nDCG@5": round(eval_res["ndcg_at_5"], 4),
            "nDCG@10": round(eval_res["ndcg_at_10"], 4),
        }
        comparison_rows.append(row)
        print(f"  {name:32s} -> MRR: {row['MRR']:.4f} | Recall@3: {row['Recall@3']:.4f} | nDCG@5: {row['nDCG@5']:.4f}")

    # Save Comparison Report
    df_comparison = pd.DataFrame(comparison_rows)
    json_path = args.outputs_dir / "retrieval_model_comparison.json"
    csv_path = args.outputs_dir / "retrieval_model_comparison.csv"

    json_payload = {
        "status": "completed",
        "benchmark_queries_count": len(queries),
        "corpus_chunks_count": len(chunks),
        "comparison_table": comparison_rows,
        "detailed_reports": full_reports,
    }
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nSaved comparison reports to:\n - {json_path}\n - {csv_path}")

    # Backwards-compatibility outputs for legacy scripts
    dense_res = full_reports.get("Sentence-BERT Dense", {})
    if dense_res:
        (args.outputs_dir / "model_metrics.json").write_text(
            json.dumps(dense_res, indent=2) + "\n", encoding="utf-8"
        )
        (args.outputs_dir / "mrr_results.json").write_text(
            json.dumps({"mrr": dense_res.get("mrr")}, indent=2) + "\n", encoding="utf-8"
        )
        (args.outputs_dir / "recall_at_k_results.json").write_text(
            json.dumps({k: dense_res[k] for k in dense_res if k.startswith("recall_at_")}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
