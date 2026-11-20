from src.query_analysis import clean_query, expand_query


def test_clean_query_preserves_technical_terms():
    raw = "  Which model uses all-MiniLM-L6-v2 and Recall@1 with U-Net???  "
    cleaned = clean_query(raw)
    assert "all-MiniLM-L6-v2" in cleaned
    assert "Recall@1" in cleaned
    assert "U-Net" in cleaned
    assert not cleaned.endswith("?")


def test_expand_query_adds_domain_synonyms():
    q = "Find projects doing image segmentation."
    expanded, added = expand_query(q)
    assert "semantic segmentation" in added or "u-net" in added
    assert len(added) > 0
    assert "semantic segmentation" in expanded or "u-net" in expanded


def test_expand_query_handles_no_synonyms_and_empty():
    assert expand_query("") == ("", [])
    q_simple = "General random text with no known domain terms"
    exp, added = expand_query(q_simple)
    assert exp == q_simple
    assert added == []
