import pytest

from app.ingestion.schemas import ChunkRecord
from app.retrieval.qdrant_store import QdrantChunkStore


class _FakeEmbedder:
    def embed_documents(self, texts):  # noqa: ANN001
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        if "finance" in text.lower():
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


@pytest.fixture
def vector_store() -> QdrantChunkStore:
    store = QdrantChunkStore.from_path(None, "test_collection")
    store.create_collection(vector_size=3)
    return store


def _chunk(doc_id: int, chunk_id: int, text: str, vector: list[float]) -> ChunkRecord:
    record = ChunkRecord(
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_text=text,
        word_len=len(text.split()),
        char_len=len(text),
    )
    record.set_embedding(vector)
    return record


def test_upsert_requires_embeddings(vector_store: QdrantChunkStore):
    chunk = ChunkRecord(
        doc_id=0,
        chunk_id=0,
        chunk_text="no vector",
        word_len=2,
        char_len=9,
    )
    with pytest.raises(ValueError, match="no embedding"):
        vector_store.upsert_chunks([chunk])


def test_search_returns_ranked_hits(vector_store: QdrantChunkStore):
    chunks = [
        _chunk(0, 0, "finance investing stocks", [1.0, 0.0, 0.0]),
        _chunk(1, 0, "cooking recipes pasta", [0.0, 1.0, 0.0]),
    ]
    vector_store.upsert_chunks(chunks)

    hits = vector_store.search([1.0, 0.0, 0.0], limit=2)

    assert len(hits) == 2
    assert hits[0].doc_id == 0
    assert hits[0].score >= hits[1].score


def test_search_text_uses_embedder(vector_store: QdrantChunkStore):
    chunks = [
        _chunk(0, 0, "finance investing stocks", [1.0, 0.0, 0.0]),
        _chunk(1, 0, "cooking recipes pasta", [0.0, 1.0, 0.0]),
    ]
    vector_store.upsert_chunks(chunks)

    hits = vector_store.search_text("finance question", _FakeEmbedder(), limit=1)

    assert len(hits) == 1
    assert hits[0].doc_id == 0
