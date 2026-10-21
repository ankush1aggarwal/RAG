"""E9 paragraph-aware chunking vs E8 fixed-word baseline (BGE)."""

from __future__ import annotations

from app.evaluation.e8_embedding_model import DEFAULT_ALTERNATIVE_MODEL as BGE_MODEL
from app.evaluation.retrieval_eval import RetrievalEvalResult
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    format_retrieval_metrics_markdown_table,
)

DEFAULT_MAX_WORDS: int = 200
DEFAULT_OVERLAP_FRACTION: float = 0.2

BASELINE_ARM = "fixed_200w"
TREATMENT_ARM = "paragraph"

BASELINE_COLLECTION = "fiqa_cs200_bge_base_en_v1_5"
TREATMENT_COLLECTION = "fiqa_para_bge_base_en_v1_5"

EMBEDDING_MODEL = BGE_MODEL

ARM_LABELS = {
    BASELINE_ARM: "fixed 200w / 20% (E8)",
    TREATMENT_ARM: "paragraph-aware / max 200w",
}


def e9_artifact_tag(arm: str) -> str:
    if arm == TREATMENT_ARM:
        return "paragraph"
    return "fixed_200w_bge"


def collection_for_arm(arm: str) -> str:
    if arm == TREATMENT_ARM:
        return TREATMENT_COLLECTION
    return BASELINE_COLLECTION


def best_arm_on_validation(
    results: dict[str, RetrievalEvalResult],
    *,
    metric_key: str = "ndcg@10",
) -> str:
    def score(arm: str) -> float:
        return results[arm].aggregates.values.get(metric_key, 0.0)

    return max(results.keys(), key=score)


def format_e9_metrics_table(
    results: dict[str, RetrievalEvalResult],
    arm_order: tuple[str, ...],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> str:
    entries = [(ARM_LABELS.get(arm, arm), results[arm]) for arm in arm_order]
    return format_retrieval_metrics_markdown_table("Arm", entries, k_values=k_values)


def format_e9_latency_table(
    results: dict[str, RetrievalEvalResult],
    arm_order: tuple[str, ...],
) -> str:
    lines = [
        "| Arm | mean (ms) | p50 (ms) | p95 (ms) |",
        "|-----|----------:|---------:|---------:|",
    ]
    for arm in arm_order:
        block = results[arm].latency_summary.get("first_stage_retrieval", {})
        label = ARM_LABELS.get(arm, arm)
        if isinstance(block, dict):
            lines.append(
                f"| {label} | {block.get('mean_ms', 0):.1f} | "
                f"{block.get('p50_ms', 0):.1f} | {block.get('p95_ms', 0):.1f} |"
            )
    return "\n".join(lines)


def render_e9_report(
    results: dict[str, RetrievalEvalResult],
    arm_order: tuple[str, ...],
    *,
    collections: dict[str, str],
    point_counts: dict[str, int],
    max_words: int,
    overlap_fraction: float,
    qrels_split: str,
    corpus_note: str,
    winner_arm: str | None = None,
    probe_chunk_count: int | None = None,
) -> str:
    lines = [
        "# Week 5 — E9 paragraph-aware chunking",
        "",
        f"**Embedding:** `{EMBEDDING_MODEL}`",
        f"**Baseline:** E8 fixed {max_words}w / {overlap_fraction:.0%} overlap",
        f"**Treatment:** paragraph merge up to {max_words}w; "
        f"sub-split with {overlap_fraction:.0%} overlap",
        f"**Qrels split:** `{qrels_split}`",
        f"**Corpus:** {corpus_note}",
        "**Retriever:** dense (no rerank)",
        "",
        "## Indexes",
        "",
        "| Arm | Collection | Points |",
        "|-----|------------|-------:|",
    ]
    for arm in arm_order:
        lines.append(
            f"| {ARM_LABELS.get(arm, arm)} | `{collections[arm]}` | "
            f"{point_counts.get(arm, 0):,} |"
        )
    if probe_chunk_count is not None:
        lines.extend(
            [
                "",
                f"**Paragraph ingest probe (chunks):** {probe_chunk_count:,}",
            ]
        )
    lines.extend(
        [
            "",
            "## Retrieval metrics (document-level)",
            "",
            format_e9_metrics_table(results, arm_order),
            "",
            "## First-stage retrieval latency",
            "",
            format_e9_latency_table(results, arm_order),
            "",
        ]
    )
    if winner_arm is not None:
        lines.append(
            f"**Validation winner (NDCG@10):** {ARM_LABELS.get(winner_arm, winner_arm)} "
            f"(`{collections[winner_arm]}`)."
        )
    return "\n".join(lines)
