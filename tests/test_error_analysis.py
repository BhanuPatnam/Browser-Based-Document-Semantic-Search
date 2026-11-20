from src.error_analysis import analyze_retrieval_errors, diagnose_cause, generate_error_analysis_markdown


def test_diagnose_cause_and_error_detection():
    queries = [
        {
            "query": "What metrics evaluate this document semantic search engine?",
            "relevant_document_ids": ["07-document-semantic-search-sentence-bert"],
        }
    ]

    # Mock search function where relevant doc is at rank 2
    def mock_suboptimal_search(q: str, k: int):
        return [
            {"document_id": "03-cross-encoder-bi-encoder-ranking-system", "section_title": "Metrics", "similarity_score": 0.8},
            {"document_id": "07-document-semantic-search-sentence-bert", "section_title": "Eval", "similarity_score": 0.7},
        ][:k]

    errors = analyze_retrieval_errors(queries, mock_suboptimal_search, threshold_rank=1)
    assert len(errors) == 1
    assert errors[0]["actual_first_relevant_rank"] == 2
    assert "metrics" in errors[0]["probable_cause"].lower() or "semantic competition" in errors[0]["probable_cause"].lower()

    md_report = generate_error_analysis_markdown(errors)
    assert "# Retrieval Error Analysis Report" in md_report
    assert "Case 1:" in md_report
