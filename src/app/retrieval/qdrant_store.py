"""Qdrant vector store for chunk embeddings."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.ingestion.schemas import ChunkRecord, SearchHit
from app.retrieval.embeddings import EmbeddingProvider


class VectorDimensionMismatchError(ValueError):
    """Raised when collection or chunk vectors do not match the embedding model."""


class QdrantChunkStore:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._closed = False

    @classmethod
    def from_path(cls, path: str | None, collection_name: str) -> QdrantChunkStore:
        if path is None:
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(path=path)
        return cls(client, collection_name)

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self._collection_name)

    def point_count(self) -> int:
        if not self.collection_exists():
            return 0
        result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )
        return int(result.count)

    def vector_size(self) -> int | None:
        if not self.collection_exists():
            return None
        info = self._client.get_collection(self._collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            if not vectors:
                return None
            first = next(iter(vectors.values()))
            return int(first.size)
        return int(vectors.size)

    def drop_collection(self) -> bool:
        """Delete this collection from Qdrant. Returns True if it existed."""
        if not self.collection_exists():
            return False
        self._client.delete_collection(self._collection_name)
        return True

    def prepare_collection(self, vector_size: int, *, recreate: bool = False) -> None:
        if self.collection_exists():
            if recreate:
                self._client.delete_collection(self._collection_name)
            else:
                self.assert_collection_vector_size(vector_size)
                return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def assert_collection_vector_size(self, expected_size: int) -> None:
        actual = self.vector_size()
        if actual is None:
            raise RuntimeError(f"Collection '{self._collection_name}' does not exist.")
        if actual != expected_size:
            raise VectorDimensionMismatchError(
                f"Collection '{self._collection_name}' vector size is {actual}, "
                f"but the embedding model produces dimension {expected_size}."
            )

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        points = []
        for idx, chunk in enumerate(chunks):
            if chunk.chunk_embed is None:
                raise ValueError(f"Chunk at index {idx} has no embedding")
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=chunk.chunk_embed,
                    payload={
                        "doc_id": chunk.doc_id,
                        "doc_title": chunk.doc_title,
                        "chunk_id": chunk.chunk_id,
                        "chunk_text": chunk.chunk_text,
                        "word_len": chunk.word_len,
                        "char_len": chunk.char_len,
                        "doc_word_len": chunk.doc_word_len,
                    },
                )
            )
        self._client.upload_points(collection_name=self._collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 3,
    ) -> list[SearchHit]:
        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[SearchHit] = []
        for point in hits.points:
            if point.payload is None:
                continue
            results.append(SearchHit.from_payload(point.payload, score=point.score))
        return results

    def search_text(
        self,
        query: str,
        embedder: EmbeddingProvider,
        limit: int = 3,
    ) -> list[SearchHit]:
        query_vector = embedder.embed_query(query)
        return self.search(query_vector=query_vector, limit=limit)

    def close(self) -> None:
        """Release the underlying Qdrant client (important for local persistent storage)."""
        if self._closed:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._closed = True
