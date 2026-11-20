import pytest

from src.chunking_experiments import (
    chunk_fixed_window_text,
    chunk_sentence_aware_text,
)


def test_chunk_fixed_window():
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    chunks = chunk_fixed_window_text(text, chunk_size_words=5, overlap_words=2)
    # step = 5 - 2 = 3
    # chunk 0: word1..word5
    # chunk 1: word4..word8
    # chunk 2: word7..word10
    assert len(chunks) == 3
    assert chunks[0] == "word1 word2 word3 word4 word5"
    assert chunks[1] == "word4 word5 word6 word7 word8"
    assert chunks[2] == "word7 word8 word9 word10"


def test_chunk_sentence_aware():
    text = "First sentence here. Second sentence follows. Third sentence closes."
    chunks = chunk_sentence_aware_text(text, chunk_size_words=6, overlap_words=2)
    assert len(chunks) >= 2
    # Ensure full sentences are preserved and not broken arbitrarily
    assert "First sentence here." in chunks[0]


def test_chunking_validation():
    with pytest.raises(ValueError, match="positive"):
        chunk_fixed_window_text("text", chunk_size_words=0)

    with pytest.raises(ValueError, match="overlap_words"):
        chunk_fixed_window_text("text", chunk_size_words=10, overlap_words=10)

    with pytest.raises(ValueError, match="overlap_words"):
        chunk_sentence_aware_text("text", chunk_size_words=10, overlap_words=-1)
