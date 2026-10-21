"""E3 candidate-diversity experiment reporting."""

from __future__ import annotations

from statistics import mean

from app.evaluation.retrieval_eval import RetrievalEvalResult
from app.evaluation.retrieval_metrics import DEFAULT_K_VALUES
from app.retrieval.candidate_diversity import E3_ARMS, E3Arm


def format_e3_metrics_table(
    results: dict[str, RetrievalEvalResult],
    arms: tuple[E3Arm, ...],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> str:
    lines = [
        "| Arm | Metric | " + " | ".join(f"@{k}" for k in k_values) + " |",
        "|-----|--------|" + "|".join(["------"] * len(k_values)) + "|",
    ]
    for arm in arms:
        aggregates = results[arm.arm_id].aggregates.values
        for metric_label, prefix in (
            ("NDCG", "ndcg"),
            ("MRR", "mrr"),
            ("Recall", "recall"),
            ("Cand. recall", "candidate_recall"),
        ):
            cells = [f"{aggregates.get(f'{prefix}@{k}', 0.0):.4f}" for k in k_values]
            lines.append(f"| {arm.arm_id} | {metric_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_e3_latency_table(results: dict[str, RetrievalEvalResult], arms: tuple[E3Arm, ...]) -> str:
    lines = [
        "| Arm | Stage | mean (ms) | p50 (ms) | p95 (ms) |",
        "|-----|-------|-----------|----------|----------|",
    ]
    for arm in arms:
        lat = results[arm.arm_id].latency_summary
        for stage_key, stage_label in (
            ("first_stage_retrieval", "first-stage retrieval"),
            ("rerank", "rerank"),
            ("retrieve_and_rerank", "retrieve + rerank"),
        ):
            block = lat.get(stage_key)
            if not isinstance(block, dict):
                continue
            lines.append(
                f"| {arm.arm_id} | {stage_label} | "
                f"{block['mean_ms']:.1f} | {block['p50_ms']:.1f} | {block['p95_ms']:.1f} |"
            )
    return "\n".join(lines)


def aggregate_pool_diagnostics(result: RetrievalEvalResult) -> dict[str, float]:
    sizes = [row.candidate_pool_size for row in result.per_query if row.candidate_pool_size is not None]
    unique = [
        row.candidate_pool_unique_docs
        for row in result.per_query
        if row.candidate_pool_unique_docs is not None
    ]
    dup_rates = [
        row.candidate_pool_duplicate_slot_rate
        for row in result.per_query
        if row.candidate_pool_duplicate_slot_rate is not None
    ]
    if not sizes:
        return {
            "mean_pool_size": 0.0,
            "mean_unique_docs": 0.0,
            "mean_duplicate_slot_rate": 0.0,
        }
    return {
        "mean_pool_size": mean(sizes),
        "mean_unique_docs": mean(unique) if unique else 0.0,
        "mean_duplicate_slot_rate": mean(dup_rates) if dup_rates else 0.0,
    }


def render_e3_report(
    results: dict[str, RetrievalEvalResult],
    arms: tuple[E3Arm, ...],
    *,
    collection_name: str,
    qrels_split: str,
    embedding_model: str,
    corpus_note: str,
    reranker_name: str,
) -> str:
    lines = [
        "# Week 5 — E3 candidate diversity",
        "",
        f"**Collection:** `{collection_name}`",
        f"**Qrels split:** `{qrels_split}`",
        f"**Embedding:** {embedding_model}",
        f"**Corpus:** {corpus_note}",
        f"**Retriever:** dense | **Reranker:** {reranker_name}",
        "",
        "## Pool diagnostics (mean over queries, pre-rerank)",
        "",
        "| Arm | mean pool size | mean unique docs | mean duplicate-slot rate |",
        "|-----|---------------:|-----------------:|-------------------------:|",
    ]
    for arm in arms:
        result = results[arm.arm_id]
        diag = aggregate_pool_diagnostics(result)
        lines.append(
            f"| {arm.label} | {diag['mean_pool_size']:.1f} | "
            f"{diag['mean_unique_docs']:.1f} | {diag['mean_duplicate_slot_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Retrieval metrics (document-level, after rerank)",
            "",
            format_e3_metrics_table(results, arms),
            "",
            "## Latency",
            "",
            format_e3_latency_table(results, arms),
            "",
            "_Select policy on validation; run test once for the chosen arm._",
        ]
    )
    return "\n".join(lines)


def default_arms() -> tuple[E3Arm, ...]:
    return E3_ARMS
