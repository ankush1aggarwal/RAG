import numpy as np
import pytest

from app.ingestion.schemas import ChunkRecord
from app.retrieval.embeddings import attach_embeddings, create_embedding_provider


class _FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t)), 0.0, 1.0] for t in texts], dtype=np.float32)

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


def test_attach_embeddings_fills_all_chunks():
    chunks = [
        ChunkRecord(doc_id=0, chunk_id=0, chunk_text="ab", word_len=1, char_len=2),
        ChunkRecord(doc_id=0, chunk_id=1, chunk_text="abcd", word_len=1, char_len=4),
    ]
    attach_embeddings(chunks, _FakeEmbedder(), batch_size=1)

    assert chunks[0].chunk_embed == [2.0, 0.0, 1.0]
    assert chunks[1].chunk_embed == [4.0, 0.0, 1.0]


def test_create_embedding_provider_rejects_unknown_method():
    with pytest.raises(ValueError, match="Invalid embedding method"):
        create_embedding_provider("invalid")  # type: ignore[arg-type]


def test_create_embedding_provider_cohere_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        create_embedding_provider("cohere")
