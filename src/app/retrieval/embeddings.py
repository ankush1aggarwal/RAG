"""Embedding providers for documents and queries."""

from __future__ import annotations

import os
import time
from typing import Literal, Protocol

import numpy as np

EmbeddingMethod = Literal["sentence_transformer", "cohere"]


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._encoder = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._encoder.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._encoder.encode(texts), dtype=np.float32)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0].tolist()


class CohereEmbedding:
    def __init__(
        self,
        api_key: str,
        model: str = "embed-v4.0",
        batch_size: int = 96,
        batch_sleep_seconds: float = 60.0,
    ) -> None:
        import cohere

        self._client = cohere.ClientV2(api_key)
        self._model = model
        self._batch_size = batch_size
        self._batch_sleep_seconds = batch_sleep_seconds
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed_query("dimension probe")
            self._dimension = len(probe)
        return self._dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embed(
                texts=batch,
                input_type="search_document",
                model=self._model,
                embedding_types=["float"],
            )
            embeddings.extend(response.embeddings.float)
            if i + self._batch_size < len(texts):
                time.sleep(self._batch_sleep_seconds)
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embed(
            texts=[text],
            input_type="search_query",
            model=self._model,
            embedding_types=["float"],
        )
        return list(response.embeddings.float[0])


def embedding_model_for_method(
    method: EmbeddingMethod,
    *,
    sentence_transformer_model: str,
    cohere_embed_model: str,
) -> str:
    """Model id stored in the ingest manifest (matches create_embedding_provider)."""
    if method == "sentence_transformer":
        return sentence_transformer_model
    if method == "cohere":
        return cohere_embed_model
    raise ValueError(f"Invalid embedding method: {method}")


def create_embedding_provider(
    method: EmbeddingMethod,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    cohere_api_key: str | None = None,
    cohere_embed_model: str = "embed-v4.0",
    cohere_embed_batch_size: int = 96,
    cohere_embed_batch_sleep_seconds: float = 60.0,
) -> EmbeddingProvider:
    if method == "sentence_transformer":
        return SentenceTransformerEmbedding(model_name=model_name)
    if method == "cohere":
        key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is required for cohere embedding method")
        return CohereEmbedding(
            api_key=key,
            model=cohere_embed_model,
            batch_size=cohere_embed_batch_size,
            batch_sleep_seconds=cohere_embed_batch_sleep_seconds,
        )
    raise ValueError(f"Invalid embedding method: {method}")


def embedding_dimension(provider: EmbeddingProvider) -> int:
    """Return vector dimension for the provider's embedding model."""
    dimension = getattr(provider, "dimension", None)
    if dimension is not None:
        return int(dimension)
    probe = provider.embed_query("dimension probe")
    return len(probe)


def validate_chunk_embedding_dimensions(chunks: list, expected_dim: int) -> None:
    for idx, chunk in enumerate(chunks):
        if chunk.chunk_embed is None:
            raise ValueError(f"Chunk at index {idx} has no embedding")
        if len(chunk.chunk_embed) != expected_dim:
            raise ValueError(
                f"Chunk at index {idx} has dimension {len(chunk.chunk_embed)}, "
                f"expected {expected_dim}"
            )


def attach_embeddings(
    chunks: list,
    provider: EmbeddingProvider,
    batch_size: int = 96,
) -> None:
    """Fill ``chunk_embed`` on chunk records in batches."""
    texts = [c.chunk_text for c in chunks]
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        vectors = provider.embed_documents(batch_texts)
        for offset, vector in enumerate(vectors):
            chunks[start + offset].set_embedding(vector)
