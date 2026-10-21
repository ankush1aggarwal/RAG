import pytest

from app.ingestion.chunking import (
    chunk_text_paragraph_aware,
    chunks_from_documents,
    split_paragraphs,
)
from app.ingestion.normalize import normalize_whitespace_preserve_paragraphs
from app.ingestion.schemas import DocumentRecord


def test_normalize_preserves_paragraph_breaks():
    raw = "First para line.\n\nSecond para here."
    assert normalize_whitespace_preserve_paragraphs(raw) == "First para line.\n\nSecond para here."


def test_split_paragraphs():
    assert split_paragraphs("a\n\nb") == ["a", "b"]


def test_paragraph_merge_up_to_max_words():
    text = "one two three\n\nfour five six"
    chunks = chunk_text_paragraph_aware(text, max_words=10, overlap_fraction=0.2)
    assert len(chunks) == 1
    assert "one two three" in chunks[0]
    assert "four five six" in chunks[0]


def test_long_paragraph_subsplit():
    words = [f"w{i}" for i in range(250)]
    para = " ".join(words)
    chunks = chunk_text_paragraph_aware(para, max_words=100, overlap_fraction=0.2)
    assert len(chunks) > 1


def test_chunks_from_documents_paragraph_strategy():
    documents = [
        DocumentRecord(
            doc_id="1",
            doc_text="alpha beta\n\ngamma delta",
            doc_title="",
            word_len=4,
            char_len=20,
        ),
    ]
    chunks = chunks_from_documents(
        documents,
        chunk_size=100,
        overlap_fraction=0.2,
        chunking_strategy="paragraph",
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_id == 0


def test_paragraph_rejects_invalid_max_words():
    with pytest.raises(ValueError, match="max_words"):
        chunk_text_paragraph_aware("a", max_words=0)
