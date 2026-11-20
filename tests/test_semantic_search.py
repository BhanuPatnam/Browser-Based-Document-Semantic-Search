import numpy as np
import pytest

from src.semantic_search import cosine_similarity_matrix, search_with_vector


def test_cosine_similarity_ranks_expected_vector_first():
    chunks = [
        {"chunk_id": "a", "project_category": "NLP", "document_type": "readme"},
        {"chunk_id": "b", "project_category": "CV", "document_type": "readme"},
    ]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    results = search_with_vector(np.array([0.9, 0.1]), chunks, matrix, top_k=2)
    assert results[0]["chunk_id"] == "a"
    assert results[0]["rank"] == 1


def test_cosine_similarity_handles_zero_safely():
    scores = cosine_similarity_matrix(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
    assert float(scores[0]) == 0.0


def test_search_rejects_invalid_top_k_and_dimension_mismatch():
    chunks = [{"chunk_id": "a"}]
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="top_k"):
        search_with_vector(np.array([1.0, 0.0]), chunks, matrix, top_k=0)
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity_matrix(np.array([1.0]), matrix)


def test_malformed_vectors_and_non_finite_values():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # NaN in query
    nan_query = np.array([np.nan, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity_matrix(nan_query, matrix)

    # Inf in matrix
    inf_matrix = np.array([[np.inf, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity_matrix(np.array([1.0, 0.0]), inf_matrix)


def test_dimension_and_shape_edge_cases():
    chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # 1D matrix passed instead of 2D
    with pytest.raises(ValueError, match="two-dimensional"):
        cosine_similarity_matrix(np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    # Mismatched rows between chunks and matrix
    with pytest.raises(ValueError, match="same number of rows"):
        search_with_vector(np.array([1.0, 0.0]), [{"chunk_id": "a"}], matrix, top_k=1)
