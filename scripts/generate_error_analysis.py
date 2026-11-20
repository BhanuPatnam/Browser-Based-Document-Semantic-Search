#!/usr/bin/env python
"""Generate retrieval error analysis report for dense and hybrid search."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.error_analysis import analyze_retrieval_errors, generate_error_analysis_markdown
from src.hybrid_retrieval import HybridRetriever
from src.lexical_retrieval import BM25Retriever
from src.semantic_search import load_index, search_with_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data/processed/document_chunks.json")
    parser.add_argument("--embeddings", type=Path, default=PROJECT_ROOT / "data/processed/embeddings.json")
    parser.add_argument("--queries", type=Path, default=PROJECT_ROOT / "data/processed/evaluation_queries.json")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--threshold-rank", type=int, default=1, help="Flag queries where rank > threshold")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    chunks, matrix, model_name = load_index(args.chunks, args.embeddings)
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    model = SentenceTransformer(model_name)

    q_texts = [q["query"] for q in queries]
    q_vectors = model.encode(q_texts, normalize_embeddings=True, convert_to_numpy=True)
    q_vec_map = {q["query"]: q_vectors[i] for i, q in enumerate(queries)}

    def search_dense(q: str, k: int):
        q_vec = q_vec_map[q]
        return search_with_vector(q_vec, chunks, matrix, top_k=k)

    print("Analyzing dense retrieval errors...")
    errors = analyze_retrieval_errors(queries, search_dense, threshold_rank=args.threshold_rank)

    json_path = args.outputs_dir / "error_analysis.json"
    md_path = args.outputs_dir / "error_analysis.md"

    json_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
    md_report = generate_error_analysis_markdown(errors)
    md_path.write_text(md_report, encoding="utf-8")

    print(f"Identified {len(errors)} error/suboptimal cases.")
    print(f"Saved reports to:\n - {json_path}\n - {md_path}")


if __name__ == "__main__":
    main()
