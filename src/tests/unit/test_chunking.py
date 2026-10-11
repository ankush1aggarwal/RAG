import pytest

from app.ingestion.chunking import chunk_text_fixed_size_with_overlap, chunks_from_documents
from app.ingestion.schemas import DocumentRecord


def test_chunking_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text_fixed_size_with_overlap("a b", chunk_size=0)


def test_chunking_rejects_invalid_overlap_fraction():
    with pytest.raises(ValueError, match="overlap_fraction"):
        chunk_text_fixed_size_with_overlap("a b", overlap_fraction=1.0)


def test_short_text_produces_single_chunk_when_above_overlap_threshold():
    text = "one two three four five"
    chunks = chunk_text_fixed_size_with_overlap(text, chunk_size=10, overlap_fraction=0.2)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_short_text_keeps_first_window_even_when_shorter_than_overlap_width():
    text = "one two three four five"
    chunks = chunk_text_fixed_size_with_overlap(text, chunk_size=100, overlap_fraction=0.2)
    assert chunks == [text]


def test_tail_window_skipped_when_only_overlap_sized():
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunk_text_fixed_size_with_overlap(text, chunk_size=10, overlap_fraction=0.2)
    # step=8 → windows at 0, 8, 16, 24; final window has 1 word (≤ overlap_int=2) and is dropped.
    assert len(chunks) == 3
    assert chunks[-1].split()[0] == "w16"
    assert chunks[-1].split()[-1] == "w24"


def test_long_text_produces_multiple_chunks_with_overlap():
    words = [f"w{i}" for i in range(250)]
    text = " ".join(words)
    chunks = chunk_text_fixed_size_with_overlap(text, chunk_size=100, overlap_fraction=0.2)

    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words


def test_chunks_from_documents_assigns_ids_and_lengths():
    documents = [
        DocumentRecord(doc_id="7", doc_text="alpha beta", word_len=2, char_len=10),
        DocumentRecord(
            doc_id="8",
            doc_text=" ".join(f"token{i}" for i in range(150)),
            word_len=150,
            char_len=1000,
        ),
    ]
    chunks = chunks_from_documents(documents, chunk_size=100, overlap_fraction=0.2)

    doc7_chunks = [c for c in chunks if c.doc_id == "7"]
    assert len(doc7_chunks) == 1
    assert doc7_chunks[0].chunk_text == "alpha beta"
    assert doc7_chunks[0].chunk_id == 0

    doc8_chunks = [c for c in chunks if c.doc_id == "8"]
    assert len(doc8_chunks) >= 2
    assert doc8_chunks[0].chunk_id == 0
    assert doc8_chunks[1].chunk_id == 1
