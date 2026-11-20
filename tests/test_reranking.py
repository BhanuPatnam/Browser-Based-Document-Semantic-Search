from unittest.mock import MagicMock
import pytest

from src.reranking import CrossEncoderReranker


def test_cross_encoder_rerank_logic():
    reranker = CrossEncoderReranker()
    # Mock model prediction to avoid heavy computation during fast unit test
    mock_model = MagicMock()
    # Query: "bi-encoder ranking"
    # cand 1 gets lower score (0.1), cand 2 gets higher score (0.9)
    mock_model.predict.return_value = [0.1, 0.9]
    reranker._model = mock_model

    candidates = [
        {"chunk_id": "c1", "text": "passage one", "rank": 1, "similarity_score": 0.85},
        {"chunk_id": "c2", "text": "passage two", "rank": 2, "similarity_score": 0.70},
    ]

    reranked = reranker.rerank("bi-encoder ranking", candidates, top_k=2)

    assert len(reranked) == 2
    # c2 was promoted to rank 1 by cross-encoder
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["rank"] == 1
    assert reranked[0]["initial_rank"] == 2
    assert reranked[0]["rerank_score"] == 0.9
    assert reranked[0]["dense_score"] == 0.70

    assert reranked[1]["chunk_id"] == "c1"
    assert reranked[1]["rank"] == 2


def test_cross_encoder_edge_cases():
    reranker = CrossEncoderReranker()
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("test", [{"text": "sample"}], top_k=0)

    assert reranker.rerank("test", [], top_k=5) == []
    assert reranker.rerank("", [{"chunk_id": "c1", "text": "text"}], top_k=1) == [{"chunk_id": "c1", "text": "text"}]
