import math
import pytest

from src.model_evaluation import (
    dcg_at_k,
    evaluate_retriever,
    hit_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_metrics_perfect_ranking():
    query = {"relevant_document_ids": ["doc_A"]}
    results = [
        {"document_id": "doc_A"},
        {"document_id": "doc_B"},
        {"document_id": "doc_C"},
    ]

    assert precision_at_k(results, query, k=1) == 1.0
    assert precision_at_k(results, query, k=3) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(results, query, k=1) == 1.0
    assert hit_at_k(results, query, k=1) == 1.0
    assert reciprocal_rank(results, query) == 1.0
    assert ndcg_at_k(results, query, k=1) == 1.0
    assert ndcg_at_k(results, query, k=3) == 1.0


def test_metrics_suboptimal_ranking():
    query = {"relevant_document_ids": ["doc_A"]}
    results = [
        {"document_id": "doc_B"},
        {"document_id": "doc_A"},
        {"document_id": "doc_C"},
    ]

    assert precision_at_k(results, query, k=1) == 0.0
    assert recall_at_k(results, query, k=1) == 0.0
    assert hit_at_k(results, query, k=1) == 0.0

    assert precision_at_k(results, query, k=2) == 0.5
    assert recall_at_k(results, query, k=2) == 1.0
    assert hit_at_k(results, query, k=2) == 1.0
    assert reciprocal_rank(results, query) == 0.5

    # At k=2: DCG = 1 / log2(2+1) = 1 / log2(3). IDCG = 1 / log2(2) = 1.0.
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(results, query, k=2) == pytest.approx(expected_ndcg)


def test_metrics_multiple_relevant_documents():
    query = {"relevant_document_ids": ["doc_A", "doc_B"]}
    results = [
        {"document_id": "doc_A"},
        {"document_id": "doc_C"},
        {"document_id": "doc_B"},
    ]

    assert recall_at_k(results, query, k=1) == 0.5
    assert recall_at_k(results, query, k=3) == 1.0
    assert precision_at_k(results, query, k=3) == pytest.approx(2.0 / 3.0)


def test_metrics_no_relevant_documents():
    query = {"relevant_document_ids": ["doc_Z"]}
    results = [{"document_id": "doc_A"}, {"document_id": "doc_B"}]

    assert precision_at_k(results, query, k=2) == 0.0
    assert recall_at_k(results, query, k=2) == 0.0
    assert reciprocal_rank(results, query) == 0.0
    assert ndcg_at_k(results, query, k=2) == 0.0


def test_metrics_validation():
    query = {"relevant_document_ids": ["doc_A"]}
    results = [{"document_id": "doc_A"}]

    with pytest.raises(ValueError, match="positive"):
        precision_at_k(results, query, k=0)

    with pytest.raises(ValueError, match="positive"):
        recall_at_k(results, query, k=-1)

    with pytest.raises(ValueError, match="positive"):
        dcg_at_k(results, query, k=0)

    with pytest.raises(ValueError, match="positive"):
        idcg_at_k(query, k=0)


def test_evaluate_retriever_summary():
    queries = [
        {"query": "q1", "relevant_document_ids": ["doc_A"]},
        {"query": "q2", "relevant_document_ids": ["doc_B"]},
    ]

    def mock_search(q: str, k: int):
        if q == "q1":
            return [{"document_id": "doc_A"}, {"document_id": "doc_C"}][:k]
        return [{"document_id": "doc_C"}, {"document_id": "doc_B"}][:k]

    summary = evaluate_retriever(queries, mock_search, ks=(1, 2), method_name="mock")
    assert summary["method"] == "mock"
    assert summary["query_count"] == 2
    # q1 RR=1.0, q2 RR=0.5 -> Mean MRR = 0.75
    assert summary["mrr"] == 0.75
    assert summary["hit_at_1"] == 0.5
    assert summary["hit_at_2"] == 1.0
