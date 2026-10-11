import numpy as np

from app.ingestion.schemas import ChunkRecord
from app.retrieval.embeddings import embedding_dimension, validate_chunk_embedding_dimensions


class _StubEmbedder:
    dimension = 3

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0, 1.0]


def test_embedding_dimension_reads_provider_property():
    assert embedding_dimension(_StubEmbedder()) == 3


def test_validate_chunk_embedding_dimensions_accepts_matching_vectors():
    chunk = ChunkRecord(
        doc_id="1",
        chunk_id=0,
        chunk_text="hello",
        word_len=1,
        char_len=5,
        doc_word_len=1,
    )
    chunk.set_embedding([0.1, 0.2, 0.3])
    validate_chunk_embedding_dimensions([chunk], expected_dim=3)
