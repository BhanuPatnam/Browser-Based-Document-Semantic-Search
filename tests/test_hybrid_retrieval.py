import numpy as np
import pytest

from src.hybrid_retrieval import HybridRetriever, normalize_scores


def test_normalize_scores():
    # Regular array
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    norm = normalize_scores(arr)
    assert pytest.approx(norm[0]) == 0.0
    assert pytest.approx(norm[1]) == 0.5
    assert pytest.approx(norm[2]) == 1.0

    # Constant array
    constant = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    norm_const = normalize_scores(constant)
    assert np.all(norm_const == 1.0)

    # Empty array
    assert normalize_scores(np.array([])).size == 0


def test_hybrid_retriever_weighting_and_fusion():
    chunks = [
        {"chunk_id": "c1", "text": "Medical image segmentation U-Net", "project_category": "CV"},
        {"chunk_id": "c2", "text": "Question answering with Longformer", "project_category": "NLP"},
        {"chunk_id": "c3", "text": "Recurrent neural network with attention", "project_category": "NLP"},
    ]
    # Synthetic dense matrix: c1 = [1, 0, 0], c2 = [0, 1, 0], c3 = [0, 0, 1]
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    hybrid = HybridRetriever(chunks, matrix, alpha=0.5)

    # Query with strong lexical match to c1 and dense vector aligned with c1
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = hybrid.search("Medical segmentation", query_vec, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["retrieval_method"] == "hybrid"
    assert "bm25_score_norm" in results[0]
    assert "dense_score_norm" in results[0]

    # Test alpha = 1.0 (pure lexical) vs alpha = 0.0 (pure dense)
    res_lexical = hybrid.search("Longformer", np.array([1.0, 0.0, 0.0]), top_k=1, alpha=1.0)
    assert res_lexical[0]["chunk_id"] == "c2"

    res_dense = hybrid.search("Irrelevant text", np.array([0.0, 1.0, 0.0]), top_k=1, alpha=0.0)
    assert res_dense[0]["chunk_id"] == "c2"


def test_hybrid_retriever_validation():
    chunks = [{"chunk_id": "1", "text": "text"}]
    matrix = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="empty"):
        HybridRetriever([], matrix)

    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(chunks, matrix, alpha=1.5)

    with pytest.raises(ValueError, match="match"):
        HybridRetriever(chunks, np.array([[1.0], [2.0]], dtype=np.float32))

    hybrid = HybridRetriever(chunks, matrix)
    with pytest.raises(ValueError, match="top_k"):
        hybrid.search("q", np.array([1.0]), top_k=0)

    with pytest.raises(ValueError, match="alpha"):
        hybrid.search("q", np.array([1.0]), top_k=1, alpha=-0.1)
