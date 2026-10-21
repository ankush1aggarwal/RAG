"""E6 chunk-size experiment helpers."""

from __future__ import annotations

from app.evaluation.retrieval_eval import RetrievalEvalResult
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    format_retrieval_metrics_markdown_table,
)

DEFAULT_CHUNK_SIZES: tuple[int, ...] = (100, 200, 300)
DEFAULT_OVERLAP_FRACTION: float = 0.2


def e6_collection_name(chunk_size: int) -> str:
    """Dedicated Qdrant/BM25 collection name for a chunk-size arm."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    return f"fiqa_cs{chunk_size}"


def resolve_e6_collection(
    chunk_size: int,
    *,
    baseline_collection: str | None = "fiqa",
    use_prefixed_collections_only: bool = False,
) -> str:
    """
    Map chunk size to collection.

    By default, 100-word reuses ``fiqa`` (Week 4 baseline) when not forcing prefixed names.
    Other sizes use ``fiqa_cs{N}``.
    """
    if use_prefixed_collections_only or baseline_collection is None:
        return e6_collection_name(chunk_size)
    if chunk_size == 100:
        return baseline_collection
    return e6_collection_name(chunk_size)


def format_e6_metrics_table(
    results: dict[int, RetrievalEvalResult],
    chunk_sizes: tuple[int, ...],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> str:
    entries = [(str(size), results[size]) for size in chunk_sizes]
    return format_retrieval_metrics_markdown_table("Chunk size", entries, k_values=k_values)


def format_e6_latency_table(
    results: dict[int, RetrievalEvalResult],
    chunk_sizes: tuple[int, ...],
) -> str:
    lines = [
        "| Chunk size | mean (ms) | p50 (ms) | p95 (ms) |",
        "|-----------:|----------:|---------:|---------:|",
    ]
    for size in chunk_sizes:
        block = results[size].latency_summary.get("first_stage_retrieval", {})
        if isinstance(block, dict):
            lines.append(
                f"| {size} | {block.get('mean_ms', 0):.1f} | "
                f"{block.get('p50_ms', 0):.1f} | {block.get('p95_ms', 0):.1f} |"
            )
    return "\n".join(lines)


def best_chunk_size_on_validation(
    results: dict[int, RetrievalEvalResult],
    *,
    metric_key: str = "ndcg@10",
) -> int:
    def score(size: int) -> float:
        return results[size].aggregates.values.get(metric_key, 0.0)

    return max(results.keys(), key=score)


def render_e6_report(
    results: dict[int, RetrievalEvalResult],
    chunk_sizes: tuple[int, ...],
    *,
    qrels_split: str,
    overlap_fraction: float,
    collections: dict[int, str],
    point_counts: dict[int, int],
    embedding_model: str,
    corpus_note: str,
    winner_size: int | None = None,
) -> str:
    lines = [
        "# Week 5 — E6 chunk size sensitivity",
        "",
        f"**Qrels split:** `{qrels_split}`",
        f"**Overlap (fixed):** {overlap_fraction:.0%}",
        f"**Embedding:** {embedding_model}",
        f"**Corpus:** {corpus_note}",
        "**Retriever:** dense (no rerank)",
        "",
        "## Collections (one index per chunk size)",
        "",
        "| Chunk size (words) | Qdrant collection | Points (chunks) |",
        "|-------------------:|-------------------|----------------:|",
    ]
    for size in chunk_sizes:
        lines.append(f"| {size} | `{collections[size]}` | {point_counts.get(size, 0):,} |")

    lines.extend(
        [
            "",
            "## Retrieval metrics (document-level)",
            "",
            format_e6_metrics_table(results, chunk_sizes),
            "",
            "## First-stage retrieval latency",
            "",
            format_e6_latency_table(results, chunk_sizes),
            "",
        ]
    )
    if winner_size is not None:
        lines.append(
            f"**Validation winner (NDCG@10):** `{winner_size}` words "
            f"(collection `{collections[winner_size]}`)."
        )
    lines.append(
        "\n_Old collections are not deleted automatically — confirm before removing any index._"
    )
    return "\n".join(lines)
