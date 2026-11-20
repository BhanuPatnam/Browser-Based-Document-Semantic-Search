import numpy as np
import pytest

from src.lexical_retrieval import BM25Retriever, TFIDFRetriever, tokenize_technical


def test_tokenize_technical_preserves_identifiers():
    text = "Deploying all-MiniLM-L6-v2 with U-Net and Recall@1 in Sentence-BERT."
    tokens = tokenize_technical(text)
    assert "all-minilm-l6-v2" in tokens
    assert "u-net" in tokens
    assert "recall@1" in tokens
    assert "sentence-bert" in tokens


def test_tfidf_retriever_search_and_filter():
    chunks = [
        {"chunk_id": "c1", "text": "Deep learning with convolutional neural networks and U-Net", "project_category": "CV", "document_type": "readme"},
        {"chunk_id": "c2", "text": "Transformer based natural language processing and BERT", "project_category": "NLP", "document_type": "readme"},
        {"chunk_id": "c3", "text": "Medical image segmentation using PyTorch and U-Net", "project_category": "CV", "document_type": "note"},
    ]
    retriever = TFIDFRetriever().fit(chunks)

    results = retriever.search("U-Net segmentation", top_k=2)
    assert len(results) == 2
    assert results[0]["chunk_id"] in ("c1", "c3")
    assert results[0]["retrieval_method"] == "tfidf"

    # Category filter where no document matches
    nlp_results = retriever.search("U-Net", top_k=5, category="NLP")
    assert len(nlp_results) == 0


def test_tfidf_retriever_edge_cases():
    retriever = TFIDFRetriever()
    with pytest.raises(RuntimeError, match="not fitted"):
        retriever.search("test")

    with pytest.raises(ValueError, match="empty chunk list"):
        retriever.fit([])

    retriever.fit([{"chunk_id": "1", "text": "hello world"}])
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("hello", top_k=0)

    # Empty query returns empty list
    assert retriever.search("", top_k=2) == []
    # Out of vocabulary query returns empty list because min_score > 0
    oov = retriever.search("xyzzy123nonexistent", top_k=1)
    assert len(oov) == 0


def test_bm25_retriever_search_and_scoring():
    chunks = [
        {"chunk_id": "c1", "text": "Sentence-BERT dense vector similarity search", "project_category": "NLP", "document_type": "readme"},
        {"chunk_id": "c2", "text": "Computer vision models and residual networks", "project_category": "CV", "document_type": "readme"},
        {"chunk_id": "c3", "text": "Recurrent neural network language model", "project_category": "NLP", "document_type": "note"},
    ]
    bm25 = BM25Retriever().fit(chunks)

    scores = bm25.score_all("Sentence-BERT search")
    assert len(scores) == 3
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

    results = bm25.search("Sentence-BERT", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["retrieval_method"] == "bm25"


def test_bm25_retriever_edge_cases():
    bm25 = BM25Retriever()
    with pytest.raises(RuntimeError, match="not fitted"):
        bm25.search("test")

    with pytest.raises(ValueError, match="empty chunk list"):
        bm25.fit([])

    bm25.fit([{"chunk_id": "1", "text": "sample document"}])
    with pytest.raises(ValueError, match="top_k"):
        bm25.search("sample", top_k=-1)

    assert bm25.search("", top_k=2) == []
