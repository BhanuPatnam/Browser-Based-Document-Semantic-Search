#!/usr/bin/env python
"""Run chunking strategy experiments across fixed-window, sentence-aware, and section-aware strategies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking_experiments import run_chunking_grid_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-docs-dir", type=Path, default=PROJECT_ROOT / "data/raw_documents")
    parser.add_argument("--queries-path", type=Path, default=PROJECT_ROOT / "data/processed/evaluation_queries.json")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading queries from {args.queries_path}...")
    queries = json.loads(args.queries_path.read_text(encoding="utf-8"))

    print(f"Loading embedding model '{args.model_name}'...")
    model = SentenceTransformer(args.model_name)

    print("Running chunking grid experiment...")
    print("  Strategies: fixed_window, sentence_aware, section_aware")
    print("  Chunk sizes: 100, 180, 250 words")
    print("  Overlaps: 20, 40, 60 words")

    results = run_chunking_grid_experiment(
        raw_docs_dir=args.raw_docs_dir,
        queries=queries,
        model=model,
        chunk_sizes=[100, 180, 250],
        overlaps=[20, 40, 60],
        strategies=["fixed_window", "sentence_aware", "section_aware"],
    )

    json_path = args.outputs_dir / "chunking_experiment_results.json"
    csv_path = args.outputs_dir / "chunking_experiment_results.csv"

    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    df = pd.DataFrame(results["results"])
    df.to_csv(csv_path, index=False)

    print(f"\nExperiment completed successfully ({results['configurations_tested']} configurations evaluated).")
    print(f"Saved results to:\n - {json_path}\n - {csv_path}")
    print("\nBest Configuration:")
    print(json.dumps(results["best_configuration"], indent=2))
    print("\nExplanation:")
    print(results["explanation"])


if __name__ == "__main__":
    main()
