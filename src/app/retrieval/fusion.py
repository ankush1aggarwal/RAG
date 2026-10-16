"""Rank fusion for multi-retriever search."""

from __future__ import annotations

from collections import defaultdict

from app.ingestion.schemas import SearchHit

ChunkKey = tuple[str, int]
DEFAULT_RRF_K = 60


def chunk_key(hit: SearchHit) -> ChunkKey:
    return (hit.doc_id, hit.chunk_id)


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int = 10,
) -> list[SearchHit]:
    """Merge ranked lists with RRF, keyed by ``(doc_id, chunk_id)``."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")

    fused_scores: dict[ChunkKey, float] = defaultdict(float)
    metadata: dict[ChunkKey, SearchHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            key = chunk_key(hit)
            fused_scores[key] += 1.0 / (k + rank)
            if key not in metadata:
                metadata[key] = hit
            elif not metadata[key].doc_title and hit.doc_title:
                metadata[key] = metadata[key].model_copy(update={"doc_title": hit.doc_title})

    ordered_keys = sorted(
        fused_scores.keys(),
        key=lambda key: (-fused_scores[key], key[0], key[1]),
    )[:limit]

    return [
        SearchHit(
            doc_id=metadata[key].doc_id,
            doc_title=metadata[key].doc_title,
            chunk_id=metadata[key].chunk_id,
            chunk_text=metadata[key].chunk_text,
            score=fused_scores[key],
        )
        for key in ordered_keys
    ]
