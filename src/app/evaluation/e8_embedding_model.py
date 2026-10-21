"""E8 embedding-model experiment helpers."""

from __future__ import annotations

import re

from app.evaluation.e7_chunk_overlap import BASELINE_OVERLAP_COLLECTION, DEFAULT_E7_CHUNK_SIZE
from app.evaluation.retrieval_eval import RetrievalEvalResult
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    format_retrieval_metrics_markdown_table,
)

DEFAULT_OVERLAP_FRACTION: float = 0.2
BASELINE_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
BASELINE_COLLECTION: str = BASELINE_OVERLAP_COLLECTION
DEFAULT_ALTERNATIVE_MODEL: str = "BAAI/bge-base-en-v1.5"


def model_slug(model_id: str) -> str:
    """Filesystem-safe slug for collection names (e.g. BAAI/bge-base-en-v1.5 → bge_base_en_v1_5)."""
    slug = model_id.split("/")[-1] if "/" in model_id else model_id
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_").lower()
    return slug or "embed"


def e8_collection_for_model(
    embedding_model: str,
    *,
    chunk_size: int = DEFAULT_E7_CHUNK_SIZE,
    baseline_model: str = BASELINE_EMBEDDING_MODEL,
    baseline_collection: str = BASELINE_COLLECTION,
) -> str:
    if embedding_model == baseline_model:
        return baseline_collection
    return f"fiqa_cs{chunk_size}_{model_slug(embedding_model)}"


def e8_artifact_tag(embedding_model: str) -> str:
    return model_slug(embedding_model)


def best_model_on_validation(
    results: dict[str, RetrievalEvalResult],
    *,
    metric_key: str = "ndcg@10",
) -> str:
    def score(model: str) -> float:
        return results[model].aggregates.values.get(metric_key, 0.0)

    return max(results.keys(), key=score)


def format_e8_metrics_table(
    results: dict[str, RetrievalEvalResult],
    model_order: tuple[str, ...],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> str:
    entries = []
    for model in model_order:
        short = model.split("/")[-1] if "/" in model else model
        entries.append((short, results[model]))
    return format_retrieval_metrics_markdown_table("Model", entries, k_values=k_values)


def format_e8_latency_table(
    results: dict[str, RetrievalEvalResult],
    model_order: tuple[str, ...],
) -> str:
    lines = [
        "| Model | mean (ms) | p50 (ms) | p95 (ms) |",
        "|-------|----------:|---------:|---------:|",
    ]
    for model in model_order:
        block = results[model].latency_summary.get("first_stage_retrieval", {})
        short = model.split("/")[-1] if "/" in model else model
        if isinstance(block, dict):
            lines.append(
                f"| {short} | {block.get('mean_ms', 0):.1f} | "
                f"{block.get('p50_ms', 0):.1f} | {block.get('p95_ms', 0):.1f} |"
            )
    return "\n".join(lines)


def render_e8_report(
    results: dict[str, RetrievalEvalResult],
    model_order: tuple[str, ...],
    *,
    collections: dict[str, str],
    dimensions: dict[str, int],
    point_counts: dict[str, int],
    chunk_size: int,
    overlap_fraction: float,
    qrels_split: str,
    corpus_note: str,
    winner_model: str | None = None,
) -> str:
    lines = [
        "# Week 5 — E8 embedding model comparison",
        "",
        f"**Chunk size / overlap:** {chunk_size} words / {overlap_fraction:.0%} (E6+E7)",
        f"**Qrels split:** `{qrels_split}`",
        f"**Corpus:** {corpus_note}",
        "**Retriever:** dense (no rerank)",
        "",
        "## Indexes",
        "",
        "| Model | Collection | Dim | Points |",
        "|-------|------------|----:|-------:|",
    ]
    for model in model_order:
        short = model.split("/")[-1] if "/" in model else model
        lines.append(
            f"| `{short}` | `{collections[model]}` | {dimensions.get(model, 0)} | "
            f"{point_counts.get(model, 0):,} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval metrics (document-level)",
            "",
            format_e8_metrics_table(results, model_order),
            "",
            "## First-stage retrieval latency",
            "",
            format_e8_latency_table(results, model_order),
            "",
        ]
    )
    if winner_model is not None:
        short = winner_model.split("/")[-1] if "/" in winner_model else winner_model
        lines.append(
            f"**Validation winner (NDCG@10):** `{short}` (`{collections[winner_model]}`)."
        )
    return "\n".join(lines)
