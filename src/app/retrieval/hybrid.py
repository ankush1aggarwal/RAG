"""Hybrid dense + BM25 retrieval."""

from __future__ import annotations

from app.ingestion.schemas import SearchHit
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.retrieval.qdrant_store import QdrantChunkStore


def hybrid_search_text(
    query: str,
    *,
    store: QdrantChunkStore,
    bm25_index: Bm25ChunkIndex,
    embedder: EmbeddingProvider,
    limit: int = 3,
    candidate_limit: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchHit]:
    """Retrieve candidates from dense and BM25, fuse with RRF."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    per_list = candidate_limit if candidate_limit is not None else max(limit * 3, limit)

    dense_hits = store.search_text(query, embedder, limit=per_list)
    bm25_hits = bm25_index.search(query, limit=per_list)
    return reciprocal_rank_fusion(
        [dense_hits, bm25_hits],
        k=rrf_k,
        limit=limit,
    )
