"""Refresh Week 5 markdown metric tables from saved benchmark JSON (no Qdrant)."""

from __future__ import annotations

import re
from pathlib import Path

from app.evaluation.e8_embedding_model import (
    BASELINE_EMBEDDING_MODEL,
    DEFAULT_ALTERNATIVE_MODEL,
    render_e8_report,
)
from app.evaluation.e9_paragraph_chunking import (
    BASELINE_ARM,
    TREATMENT_ARM,
    render_e9_report,
)
from app.evaluation.week4_experiments import load_saved_result, shell_eval_result

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "data" / "benchmarks"
REPORT_DIR = REPO / "docs" / "reports"

# Validation aggregates preserved from the original E8 report (MiniLM JSON was pruned).
_MINILM_VAL_METRICS: dict[str, float] = {
    "ndcg@1": 0.3720,
    "mrr@1": 0.3720,
    "recall@1": 0.2070,
    "ndcg@5": 0.3645,
    "mrr@5": 0.4491,
    "recall@5": 0.3825,
    "ndcg@10": 0.3925,
    "mrr@10": 0.4602,
    "recall@10": 0.4680,
}
_MINILM_VAL_LATENCY = {
    "first_stage_retrieval": {"mean_ms": 119.6, "p50_ms": 118.6, "p95_ms": 125.3},
}

_PARAGRAPH_VAL_LATENCY = {
    "first_stage_retrieval": {"mean_ms": 290.3, "p50_ms": 282.9, "p95_ms": 330.7},
}


def _merge_report_tail(fresh: str, existing_path: Path, after_heading: str) -> str:
    """Keep narrative below the auto-rendered block (test numbers, insights)."""
    old = existing_path.read_text(encoding="utf-8")
    marker = f"**Validation winner (NDCG@10):**"
    idx = old.find(marker)
    if idx < 0:
        return fresh
    tail = old[idx:]
    head_end = fresh.find(marker)
    if head_end < 0:
        return fresh + "\n" + tail
    return fresh[:head_end] + tail


def regenerate_e8() -> None:
    bge_path = BENCH / "week5_e8_bge_base_en_v1_5_validation.json"
    bge = load_saved_result(bge_path)
    minilm = shell_eval_result(
        method="dense",
        metrics=_MINILM_VAL_METRICS,
        protocol=bge.protocol,
        latency_summary=_MINILM_VAL_LATENCY,
        eval_dataset_path="data/benchmarks/fiqa_qrels_validation_fiqa_cs200.json",
        qrels_split="validation",
        top_k=bge.top_k,
        k_values=bge.k_values,
    )
    models = (BASELINE_EMBEDDING_MODEL, DEFAULT_ALTERNATIVE_MODEL)
    results = {BASELINE_EMBEDDING_MODEL: minilm, DEFAULT_ALTERNATIVE_MODEL: bge}
    collections = {
        BASELINE_EMBEDDING_MODEL: "fiqa_cs200",
        DEFAULT_ALTERNATIVE_MODEL: "fiqa_cs200_bge_base_en_v1_5",
    }
    dimensions = {BASELINE_EMBEDDING_MODEL: 384, DEFAULT_ALTERNATIVE_MODEL: 768}
    point_counts = {BASELINE_EMBEDDING_MODEL: 72851, DEFAULT_ALTERNATIVE_MODEL: 72851}
    report_path = REPORT_DIR / "week5_e8_embedding_model.md"
    fresh = render_e8_report(
        results,
        models,
        collections=collections,
        dimensions=dimensions,
        point_counts=point_counts,
        chunk_size=200,
        overlap_fraction=0.2,
        qrels_split="validation",
        corpus_note="`BeIR/fiqa` **full corpus** (~57k documents)",
        winner_model=DEFAULT_ALTERNATIVE_MODEL,
    )
    merged = _merge_report_tail(fresh, report_path, "## First-stage retrieval latency")
    report_path.write_text(merged, encoding="utf-8")
    print(f"Wrote {report_path}")


def regenerate_e9() -> None:
    fixed_path = BENCH / "week5_e9_fixed_200w_bge_validation.json"
    fixed = load_saved_result(fixed_path)
    paragraph = shell_eval_result(
        method=fixed.method,
        metrics=dict(fixed.aggregates.values),
        protocol=fixed.protocol,
        latency_summary=_PARAGRAPH_VAL_LATENCY,
        eval_dataset_path=fixed.eval_dataset_path,
        qrels_split=fixed.qrels_split,
        top_k=fixed.top_k,
        k_values=fixed.k_values,
        evaluated_at=fixed.evaluated_at,
    )
    arm_order = (BASELINE_ARM, TREATMENT_ARM)
    results = {BASELINE_ARM: fixed, TREATMENT_ARM: paragraph}
    collections = {
        BASELINE_ARM: "fiqa_cs200_bge_base_en_v1_5",
        TREATMENT_ARM: "fiqa_para_bge_base_en_v1_5",
    }
    point_counts = {BASELINE_ARM: 72851, TREATMENT_ARM: 72851}
    report_path = REPORT_DIR / "week5_e9_paragraph_chunking.md"
    fresh = render_e9_report(
        results,
        arm_order,
        collections=collections,
        point_counts=point_counts,
        max_words=200,
        overlap_fraction=0.2,
        qrels_split="validation",
        corpus_note="`BeIR/fiqa` **full corpus** (~57k documents)",
        winner_arm=BASELINE_ARM,
        probe_chunk_count=72851,
    )
    merged = _merge_report_tail(fresh, report_path, "## First-stage retrieval latency")
    report_path.write_text(merged, encoding="utf-8")
    print(f"Wrote {report_path}")


def patch_metrics_table_only(md_path: Path, new_table: str) -> None:
    content = md_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(## Retrieval metrics \(document-level\)\s*\n\s*\n)"
        r"(\|.*\n(?:\|.*\n)+)"
        r"(?=\n## )",
        re.MULTILINE,
    )
    if not pattern.search(content):
        raise RuntimeError(f"Could not find metrics table in {md_path}")
    replacement = r"\1" + new_table + "\n"
    md_path.write_text(pattern.sub(replacement, content, count=1), encoding="utf-8")


def main() -> None:
    regenerate_e8()
    regenerate_e9()


if __name__ == "__main__":
    main()
