from unittest.mock import MagicMock

import pytest

from app.ingestion.schemas import ChunkRecord
from app.retrieval.qdrant_store import QdrantChunkStore, VectorDimensionMismatchError


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
    store.prepare_collection(vector_size=3, recreate=True)
    return store


def _chunk(doc_id: str, chunk_id: int, text: str, vector: list[float]) -> ChunkRecord:
    record = ChunkRecord(
        doc_id=doc_id,
        doc_title="Example title",
        chunk_id=chunk_id,
        chunk_text=text,
        word_len=len(text.split()),
        char_len=len(text),
        doc_word_len=100,
    )
    record.set_embedding(vector)
    return record


def test_prepare_collection_reuses_existing_when_not_recreate():
    store = QdrantChunkStore.from_path(None, "existing_collection")
    store.prepare_collection(3, recreate=True)
    store.upsert_chunks(
        [
            _chunk("1", 0, "finance stocks", [1.0, 0.0, 0.0]),
        ]
    )

    store.prepare_collection(3, recreate=False)
    assert store.collection_exists()
    hits = store.search([1.0, 0.0, 0.0], limit=1)
    assert len(hits) == 1


def test_prepare_collection_rejects_dimension_mismatch_on_existing():
    store = QdrantChunkStore.from_path(None, "dim_mismatch_collection")
    store.prepare_collection(3, recreate=True)

    with pytest.raises(VectorDimensionMismatchError):
        store.prepare_collection(4, recreate=False)


def test_prepare_collection_recreate_replaces_collection():
    store = QdrantChunkStore.from_path(None, "recreate_collection")
    store.prepare_collection(3, recreate=True)
    store.assert_collection_vector_size(3)

    store.prepare_collection(3, recreate=True)
    store.assert_collection_vector_size(3)


def test_upsert_requires_embeddings(vector_store: QdrantChunkStore):
    chunk = ChunkRecord(
        doc_id="3",
        doc_title="",
        chunk_id=0,
        chunk_text="no vector",
        word_len=2,
        char_len=9,
        doc_word_len=50,
    )
    with pytest.raises(ValueError, match="no embedding"):
        vector_store.upsert_chunks([chunk])


def test_search_returns_ranked_hits_with_doc_title(vector_store: QdrantChunkStore):
    chunks = [
        _chunk("160", 0, "finance investing stocks", [1.0, 0.0, 0.0]),
        _chunk("766", 0, "cooking recipes pasta", [0.0, 1.0, 0.0]),
    ]
    vector_store.upsert_chunks(chunks)

    hits = vector_store.search([1.0, 0.0, 0.0], limit=2)

    assert len(hits) == 2
    assert hits[0].doc_id == "160"
    assert hits[0].doc_title == "Example title"
    assert hits[0].score >= hits[1].score


def test_close_calls_underlying_client():
    client = MagicMock()
    store = QdrantChunkStore(client, "c")
    store.close()
    client.close.assert_called_once()
    store.close()
    client.close.assert_called_once()


def test_search_text_uses_embedder(vector_store: QdrantChunkStore):
    chunks = [
        _chunk("160", 0, "finance investing stocks", [1.0, 0.0, 0.0]),
        _chunk("766", 0, "cooking recipes pasta", [0.0, 1.0, 0.0]),
    ]
    vector_store.upsert_chunks(chunks)

    hits = vector_store.search_text("finance question", _FakeEmbedder(), limit=1)

    assert len(hits) == 1
    assert hits[0].doc_id == "160"
