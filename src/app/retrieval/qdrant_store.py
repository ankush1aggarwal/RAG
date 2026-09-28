"""Qdrant vector store for chunk embeddings."""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.ingestion.schemas import ChunkRecord, SearchHit
from app.retrieval.embeddings import EmbeddingProvider


class QdrantChunkStore:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @classmethod
    def from_path(cls, path: str | None, collection_name: str) -> "QdrantChunkStore":
        if path is None:
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(path=path)
        return cls(client, collection_name)

    def create_collection(self, vector_size: int) -> None:
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
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
                        "chunk_id": chunk.chunk_id,
                        "chunk_text": chunk.chunk_text,
                        "word_len": chunk.word_len,
                        "char_len": chunk.char_len,
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
