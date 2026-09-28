import numpy as np

from app.ingestion.schemas import ChunkRecord, DocumentRecord, SearchHit


def test_chunk_record_set_embedding_from_list():
    chunk = ChunkRecord(
        doc_id=0,
        chunk_id=0,
        chunk_text="hello",
        word_len=1,
        char_len=5,
    )
    chunk.set_embedding([0.1, 0.2, 0.3])
    assert chunk.chunk_embed is not None
    assert len(chunk.chunk_embed) == 3
    assert all(abs(a - b) < 1e-6 for a, b in zip(chunk.chunk_embed, [0.1, 0.2, 0.3], strict=True))


def test_chunk_record_set_embedding_from_numpy():
    chunk = ChunkRecord(
        doc_id=1,
        chunk_id=0,
        chunk_text="world",
        word_len=1,
        char_len=5,
    )
    chunk.set_embedding(np.array([1.0, 2.0], dtype=np.float64))
    assert chunk.chunk_embed == [1.0, 2.0]


def test_search_hit_from_payload():
    hit = SearchHit.from_payload(
        {
            "doc_id": 10,
            "chunk_id": 2,
            "chunk_text": "sample text",
        },
        score=0.87,
    )
    assert hit.doc_id == 10
    assert hit.chunk_id == 2
    assert hit.chunk_text == "sample text"
    assert hit.score == 0.87


def test_document_record_fields():
    doc = DocumentRecord(doc_id=0, doc_text="a b", word_len=2, char_len=3)
    assert doc.doc_text == "a b"
