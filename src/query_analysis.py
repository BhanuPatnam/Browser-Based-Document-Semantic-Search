"""Query preprocessing and rule-based expansion for technical/NLP retrieval."""

from __future__ import annotations

import re

# Technical-domain synonyms and expansion terms for ML / NLP / IR concepts
DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "image segmentation": ["semantic segmentation", "mask prediction", "u-net"],
    "semantic segmentation": ["image segmentation", "mask prediction", "u-net"],
    "semantic search": ["dense retrieval", "vector search", "sentence-bert", "cosine similarity"],
    "dense retrieval": ["semantic search", "vector search", "embedding retrieval"],
    "question answering": ["qa", "reading comprehension", "longformer"],
    "qa": ["question answering", "longformer"],
    "cross-encoder": ["reranking", "two-stage retrieval", "bi-encoder ranking"],
    "bi-encoder": ["dense retrieval", "sentence-bert", "embedding model"],
    "reranking": ["cross-encoder", "second-stage ranking", "candidate scoring"],
    "summarization": ["abstractive summarization", "bertscore", "text summary"],
    "quality analytics": ["root cause", "capa", "sop", "investigation"],
    "root cause": ["quality analytics", "capa", "troubleshooting"],
    "bilstm": ["bidirectional lstm", "attention", "emotion detection"],
    "lstm": ["bidirectional lstm", "recurrent neural network", "rnn"],
    "hugging face": ["huggingface spaces", "hf spaces", "spaces deployment"],
    "github pages": ["static deployment", "browser deployment", "client-side inference"],
    "transformers.js": ["browser ai", "onnx runtime", "in-browser model"],
    "evaluation metrics": ["recall@k", "mrr", "ndcg", "precision@k"],
}


def clean_query(query: str) -> str:
    """Normalize query text while strictly preserving technical terms, model names, and metrics.

    Preserves:
      - Hyphenated identifiers (e.g. all-MiniLM-L6-v2, U-Net, Sentence-BERT)
      - Metrics and acronyms (e.g. Recall@1, MRR, nDCG@5, SOP, CAPA, VQA, LSTM)
      - Decimal numbers and version numbers (e.g. 1.0, 3.12)
    """
    if not isinstance(query, str):
        return ""
    text = query.strip()
    if not text:
        return ""

    # Replace newlines and multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove terminal question marks or exclamation marks while keeping internal punctuation
    text = re.sub(r"[?!]+$", "", text).strip()
    return text


def expand_query(
    query: str,
    synonym_map: dict[str, list[str]] | None = None,
) -> tuple[str, list[str]]:
    """Perform rule-based query expansion for known technical terms.

    Args:
        query: Raw or cleaned user query string.
        synonym_map: Optional dictionary of keyword -> list of expansion synonyms.

    Returns:
        tuple of (expanded_query_string, list_of_expanded_terms_added)
    """
    cleaned = clean_query(query)
    if not cleaned:
        return "", []

    mapping = synonym_map if synonym_map is not None else DOMAIN_SYNONYMS
    query_lower = cleaned.lower()

    added_terms: list[str] = []
    seen_terms: set[str] = set(query_lower.split())

    # Check for multi-word or single-word matches in synonym map
    # Sort keys by length descending to match longest phrases first
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    for key in sorted_keys:
        # Match as whole word/phrase
        pattern = r"\b" + re.escape(key.lower()) + r"\b"
        if re.search(pattern, query_lower):
            for candidate in mapping[key]:
                cand_lower = candidate.lower()
                if cand_lower not in seen_terms and cand_lower not in query_lower:
                    added_terms.append(candidate)
                    seen_terms.add(cand_lower)

    if added_terms:
        expanded_query = f"{cleaned} ({' '.join(added_terms)})"
    else:
        expanded_query = cleaned

    return expanded_query, added_terms
