"""Hybrid dense + BM25 retrieval."""

from __future__ import annotations

from app.ingestion.schemas import SearchHit
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from app.retrieval.qdrant_store import QdrantChunkStore


def resolve_hybrid_candidate_limit(limit: int, candidate_limit: int | None = None) -> int:
    if candidate_limit is not None:
        return candidate_limit
    return max(limit * 3, limit)


def hybrid_search_text(
    query: str,
    *,
    store: QdrantChunkStore,
    bm25_index: Bm25ChunkIndex,
    embedder: EmbeddingProvider,
    limit: int = 3,
    candidate_limit: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
    fusion_weights: tuple[float, float] | None = None,
) -> list[SearchHit]:
    """Retrieve candidates from dense and BM25, fuse with (weighted) RRF."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    per_list = resolve_hybrid_candidate_limit(limit, candidate_limit)

    dense_hits = store.search_text(query, embedder, limit=per_list)
    bm25_hits = bm25_index.search(query, limit=per_list)
    list_weights = None
    if fusion_weights is not None:
        list_weights = [fusion_weights[0], fusion_weights[1]]
    return reciprocal_rank_fusion(
        [dense_hits, bm25_hits],
        list_weights=list_weights,
        k=rrf_k,
        limit=limit,
    )


def hybrid_candidate_lists(
    query: str,
    *,
    store: QdrantChunkStore,
    bm25_index: Bm25ChunkIndex,
    embedder: EmbeddingProvider,
    candidate_limit: int,
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Dense + BM25 lists for reuse when tuning fusion weights."""
    dense_hits = store.search_text(query, embedder, limit=candidate_limit)
    bm25_hits = bm25_index.search(query, limit=candidate_limit)
    return dense_hits, bm25_hits


def fuse_hybrid_candidate_lists(
    dense_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    *,
    limit: int,
    fusion_weights: tuple[float, float] | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchHit]:
    list_weights = None
    if fusion_weights is not None:
        list_weights = [fusion_weights[0], fusion_weights[1]]
    return reciprocal_rank_fusion(
        [dense_hits, bm25_hits],
        list_weights=list_weights,
        k=rrf_k,
        limit=limit,
    )
