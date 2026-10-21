"""E7 chunk-overlap experiment helpers."""

from __future__ import annotations

from app.evaluation.retrieval_eval import RetrievalEvalResult
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    format_retrieval_metrics_markdown_table,
)

DEFAULT_E7_CHUNK_SIZE: int = 200
DEFAULT_OVERLAP_FRACTIONS: tuple[float, ...] = (0.1, 0.2, 0.3)
BASELINE_OVERLAP_FRACTION: float = 0.2
BASELINE_OVERLAP_COLLECTION: str = "fiqa_cs200"


def overlap_percent(overlap_fraction: float) -> int:
    return int(round(overlap_fraction * 100))


def e7_collection_name(overlap_fraction: float) -> str:
    """Map overlap arm to Qdrant/BM25 collection (reuse E6 winner index at 20%)."""
    if abs(overlap_fraction - BASELINE_OVERLAP_FRACTION) < 1e-9:
        return BASELINE_OVERLAP_COLLECTION
    pct = overlap_percent(overlap_fraction)
    return f"fiqa_cs200_ov{pct}"


def overlap_artifact_tag(overlap_fraction: float) -> str:
    return f"ov{overlap_percent(overlap_fraction)}"


def best_overlap_on_validation(
    results: dict[float, RetrievalEvalResult],
    *,
    metric_key: str = "ndcg@10",
) -> float:
    def score(overlap: float) -> float:
        return results[overlap].aggregates.values.get(metric_key, 0.0)

    return max(results.keys(), key=score)


def format_e7_metrics_table(
    results: dict[float, RetrievalEvalResult],
    overlaps: tuple[float, ...],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> str:
    entries = [(f"{overlap_percent(overlap)}%", results[overlap]) for overlap in overlaps]
    return format_retrieval_metrics_markdown_table("Overlap", entries, k_values=k_values)


def format_e7_latency_table(
    results: dict[float, RetrievalEvalResult],
    overlaps: tuple[float, ...],
) -> str:
    lines = [
        "| Overlap | mean (ms) | p50 (ms) | p95 (ms) |",
        "|--------:|----------:|---------:|---------:|",
    ]
    for overlap in overlaps:
        block = results[overlap].latency_summary.get("first_stage_retrieval", {})
        if isinstance(block, dict):
            lines.append(
                f"| {overlap_percent(overlap)}% | {block.get('mean_ms', 0):.1f} | "
                f"{block.get('p50_ms', 0):.1f} | {block.get('p95_ms', 0):.1f} |"
            )
    return "\n".join(lines)


def render_e7_report(
    results: dict[float, RetrievalEvalResult],
    overlaps: tuple[float, ...],
    *,
    chunk_size: int,
    qrels_split: str,
    collections: dict[float, str],
    point_counts: dict[float, int],
    embedding_model: str,
    corpus_note: str,
    winner_overlap: float | None = None,
) -> str:
    lines = [
        "# Week 5 — E7 chunk overlap sensitivity",
        "",
        f"**Chunk size (fixed):** {chunk_size} words (E6 winner)",
        f"**Qrels split:** `{qrels_split}`",
        f"**Embedding:** {embedding_model}",
        f"**Corpus:** {corpus_note}",
        "**Retriever:** dense (no rerank)",
        "",
        "## Collections",
        "",
        "| Overlap | Qdrant collection | Points (chunks) |",
        "|--------:|-------------------|----------------:|",
    ]
    for overlap in overlaps:
        lines.append(
            f"| {overlap_percent(overlap)}% | `{collections[overlap]}` | "
            f"{point_counts.get(overlap, 0):,} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval metrics (document-level)",
            "",
            format_e7_metrics_table(results, overlaps),
            "",
            "## First-stage retrieval latency",
            "",
            format_e7_latency_table(results, overlaps),
            "",
        ]
    )
    if winner_overlap is not None:
        lines.append(
            f"**Validation winner (NDCG@10):** {overlap_percent(winner_overlap)}% overlap "
            f"(`{collections[winner_overlap]}`)."
        )
    return "\n".join(lines)
