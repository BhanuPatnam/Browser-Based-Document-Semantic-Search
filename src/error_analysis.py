"""Error analysis for semantic and lexical information retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def analyze_retrieval_errors(
    queries: list[dict[str, Any]],
    search_fn: Callable[[str, int], list[dict[str, Any]]],
    threshold_rank: int = 1,
    max_k: int = 10,
) -> list[dict[str, Any]]:
    """Identify and diagnose queries where relevant documents are not ranked at the top.

    Args:
        queries: Evaluation query dictionaries with target/relevant IDs.
        search_fn: Function (query_str, top_k) -> list of result dicts.
        threshold_rank: Minimum target rank to flag (e.g., rank > 1 or rank >= 3).
        max_k: Max results to retrieve for diagnosis.

    Returns:
        List of structured error diagnosis records.
    """
    diagnoses: list[dict[str, Any]] = []

    for q in queries:
        query_text = str(q.get("query", ""))
        target_docs = set(q.get("relevant_document_ids", []))
        target_chunks = set(q.get("relevant_chunk_ids", []))

        results = search_fn(query_text, max_k)

        # Find first rank of relevant document
        first_rel_rank = None
        first_rel_item = None
        for rank, item in enumerate(results, start=1):
            if item.get("document_id") in target_docs or item.get("chunk_id") in target_chunks:
                first_rel_rank = rank
                first_rel_item = item
                break

        # Flag if not found in top_k or rank > threshold_rank
        is_suboptimal = (first_rel_rank is None) or (first_rel_rank > threshold_rank)

        if is_suboptimal:
            top_retrieved = [
                {
                    "rank": r.get("rank", idx + 1),
                    "document_id": r.get("document_id"),
                    "project_name": r.get("project_name"),
                    "section_title": r.get("section_title"),
                    "similarity_score": round(float(r.get("similarity_score", 0.0)), 4),
                }
                for idx, r in enumerate(results[:5])
            ]

            # Factual diagnosis based on query semantics and retrieved competitors
            cause, improvement = diagnose_cause(query_text, target_docs, top_retrieved, first_rel_rank)

            diagnoses.append({
                "query": query_text,
                "expected_document_ids": list(target_docs),
                "actual_first_relevant_rank": first_rel_rank if first_rel_rank is not None else -1,
                "found_in_top_k": first_rel_rank is not None,
                "top_retrieved_documents": top_retrieved,
                "probable_cause": cause,
                "suggested_improvement": improvement,
            })

    return diagnoses


def diagnose_cause(
    query: str,
    expected_docs: set[str],
    top_retrieved: list[dict[str, Any]],
    first_rank: int | None,
) -> tuple[str, str]:
    """Diagnose the root cause of a suboptimal rank based on corpus content and query terms."""
    q_lower = query.lower()
    top_1_doc = top_retrieved[0]["document_id"] if top_retrieved else ""

    if first_rank is None:
        return (
            "Complete retrieval omission: No passage from the target document was retrieved within top-K. "
            "The query terms likely share semantic overlap with competing documents in the corpus.",
            "Apply hybrid BM25 lexical weighting to anchor exact keyword matches or expand query terms.",
        )

    if "metrics" in q_lower or "evaluate" in q_lower:
        if top_1_doc == "03-cross-encoder-bi-encoder-ranking-system":
            return (
                "Semantic competition on evaluation terminology: Project 03 features dense discussions of "
                "evaluation metrics (MRR, nDCG, Precision@K, Cross-Encoder reranking scores), which causes the bi-encoder "
                "to assign it high cosine similarity alongside Project 07.",
                "Implement two-stage Cross-Encoder reranking to resolve fine-grained distinction between project targets, "
                "or incorporate project-specific lexical filters.",
            )

    if "hugging face" in q_lower or "spaces" in q_lower:
        return (
            "Multi-target document dispersion: Multiple documents in the corpus discuss Hugging Face Spaces deployments. "
            "Bi-encoder ranks passages based on local text prominence of 'Spaces' rather than global document hierarchy.",
            "Use hybrid BM25 + dense retrieval with section weighting or deduplicate chunks by parent document.",
        )

    return (
        f"Dense embedding competition: Target document was ranked at #{first_rank}. "
        f"Top document '{top_1_doc}' shares strong semantic representation with query phrasing.",
        "Apply Cross-Encoder second-stage reranking or hybrid lexical-dense scoring with higher BM25 weight.",
    )


def generate_error_analysis_markdown(errors: list[dict[str, Any]]) -> str:
    """Format error analysis list as a clean GitHub-flavored Markdown report."""
    lines = [
        "# Retrieval Error Analysis Report",
        "",
        "This report identifies queries where dense retrieval did not rank a relevant target at rank 1, "
        "or where relevant documents ranked below the acceptable threshold.",
        "",
        f"**Total suboptimal queries identified:** {len(errors)}",
        "",
        "---",
    ]

    for idx, err in enumerate(errors, start=1):
        lines.extend([
            f"### Case {idx}: \"{err['query']}\"",
            f"- **Expected Relevant Document(s):** `{', '.join(err['expected_document_ids'])}`",
            f"- **First Relevant Rank:** `#{err['actual_first_relevant_rank']}`" if err['actual_first_relevant_rank'] > 0 else "- **First Relevant Rank:** `Not found in top-K`",
            "- **Top Retrieved Passages:**",
        ])
        for top in err["top_retrieved_documents"][:3]:
            lines.append(f"  - Rank #{top['rank']}: **{top['document_id']}** ({top['section_title']}) — Score: `{top['similarity_score']}`")
        lines.extend([
            f"- **Probable Cause:** {err['probable_cause']}",
            f"- **Suggested Improvement:** {err['suggested_improvement']}",
            "",
            "---",
        ])

    return "\n".join(lines)
