"""Tune dense/BM25 RRF weights on a labeled validation set."""

from __future__ import annotations

from dataclasses import dataclass

from app.evaluation.fiqa_qrels import FiqaEvalDataset
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    aggregate_query_metrics,
    compute_query_metrics,
    doc_ranking_from_hits,
)
from app.ingestion.schemas import SearchHit
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.fusion_weights import FusionWeights
from app.retrieval.hybrid import (
    fuse_hybrid_candidate_lists,
    hybrid_candidate_lists,
    resolve_hybrid_candidate_limit,
)
from app.retrieval.qdrant_store import QdrantChunkStore


@dataclass(frozen=True)
class HybridListCache:
    """Per-query dense and BM25 candidate lists."""

    by_query_id: dict[str, tuple[list[SearchHit], list[SearchHit]]]


def prefetch_hybrid_list_cache(
    dataset: FiqaEvalDataset,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex,
    top_k: int,
    progress_every: int = 50,
) -> HybridListCache:
    candidate_limit = resolve_hybrid_candidate_limit(top_k, None)
    cache: dict[str, tuple[list[SearchHit], list[SearchHit]]] = {}
    for index, query in enumerate(dataset.queries, start=1):
        cache[query.query_id] = hybrid_candidate_lists(
            query.question,
            store=store,
            bm25_index=bm25_index,
            embedder=embedder,
            candidate_limit=candidate_limit,
        )
        if progress_every > 0 and index % progress_every == 0:
            print(f"  cached hybrid lists {index}/{dataset.num_queries}...")
    return HybridListCache(by_query_id=cache)


def _mean_metric(
    dataset: FiqaEvalDataset,
    cache: HybridListCache,
    *,
    dense_weight: float,
    bm25_weight: float,
    top_k: int,
    metric_name: str,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> float:
    fusion = (dense_weight, bm25_weight)
    per_query = []
    for query in dataset.queries:
        dense_hits, bm25_hits = cache.by_query_id[query.query_id]
        hits = fuse_hybrid_candidate_lists(
            dense_hits,
            bm25_hits,
            limit=top_k,
            fusion_weights=fusion,
        )
        ranked = doc_ranking_from_hits(hits)
        per_query.append(
            compute_query_metrics(ranked, query.relevance_by_doc_id, k_values=k_values)
        )
    aggregates = aggregate_query_metrics(per_query, k_values=k_values)
    if metric_name not in aggregates.values:
        raise ValueError(f"Unknown metric {metric_name!r}")
    return aggregates.values[metric_name]


def tune_fusion_weights(
    dataset: FiqaEvalDataset,
    cache: HybridListCache,
    *,
    top_k: int = 10,
    metric_name: str = "ndcg@10",
    grid_steps: int = 21,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> FusionWeights:
    """
    Grid-search ``dense_weight`` in [0, 1] with ``bm25_weight = 1 - dense_weight``.

    Maximizes mean ``metric_name`` (default NDCG@10) on the validation queries.
    """
    if grid_steps < 2:
        raise ValueError("grid_steps must be at least 2")

    best_dense = 0.5
    best_score = float("-inf")
    for step in range(grid_steps):
        dense_w = step / (grid_steps - 1)
        bm25_w = 1.0 - dense_w
        score = _mean_metric(
            dataset,
            cache,
            dense_weight=dense_w,
            bm25_weight=bm25_w,
            top_k=top_k,
            metric_name=metric_name,
            k_values=k_values,
        )
        if score > best_score:
            best_score = score
            best_dense = dense_w

    return FusionWeights(
        dense=best_dense,
        bm25=1.0 - best_dense,
        metric_optimized=metric_name,
        qrels_split=dataset.qrels_split,
        num_queries=dataset.num_queries,
        score_at_selection=best_score,
    )
