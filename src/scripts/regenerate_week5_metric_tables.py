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
from app.evaluation.e3_experiments import format_e3_metrics_table
from app.evaluation.e6_chunk_size import format_e6_metrics_table
from app.evaluation.e7_chunk_overlap import format_e7_metrics_table
from app.evaluation.week4_experiments import (
    drop_candidate_recall_metric,
    load_saved_result,
    shell_eval_result,
)
from app.retrieval.candidate_diversity import e3_arm_by_id

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
    paragraph_metrics = drop_candidate_recall_metric(
        dict(fixed.aggregates.values),
        candidate_recall_k=fixed.protocol.candidate_recall_k,
    )
    paragraph = shell_eval_result(
        method=fixed.method,
        metrics=paragraph_metrics,
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


_OVERFETCH_CAP2_VAL_METRICS: dict[str, float] = {
    "ndcg@1": 0.3534,
    "mrr@1": 0.3534,
    "recall@1": 0.1827,
    "ndcg@5": 0.3431,
    "mrr@5": 0.4288,
    "recall@5": 0.3630,
    "ndcg@10": 0.3686,
    "mrr@10": 0.4405,
    "recall@10": 0.4408,
}

_E6_VAL_METRICS: dict[int, dict[str, float]] = {
    100: {
        "ndcg@1": 0.3620,
        "mrr@1": 0.3620,
        "recall@1": 0.2046,
        "ndcg@5": 0.3580,
        "mrr@5": 0.4328,
        "recall@5": 0.3781,
        "ndcg@10": 0.3853,
        "mrr@10": 0.4463,
        "recall@10": 0.4620,
    },
    200: {
        "ndcg@1": 0.3720,
        "mrr@1": 0.3720,
        "recall@1": 0.2070,
        "ndcg@5": 0.3645,
        "mrr@5": 0.4491,
        "recall@5": 0.3825,
        "ndcg@10": 0.3925,
        "mrr@10": 0.4602,
        "recall@10": 0.4680,
    },
    300: {
        "ndcg@1": 0.3700,
        "mrr@1": 0.3700,
        "recall@1": 0.2110,
        "ndcg@5": 0.3628,
        "mrr@5": 0.4475,
        "recall@5": 0.3793,
        "ndcg@10": 0.3907,
        "mrr@10": 0.4581,
        "recall@10": 0.4613,
    },
}

_E7_VAL_METRICS: dict[float, dict[str, float]] = {
    0.1: {
        "ndcg@1": 0.3520,
        "mrr@1": 0.3520,
        "recall@1": 0.1967,
        "ndcg@5": 0.3526,
        "mrr@5": 0.4317,
        "recall@5": 0.3695,
        "ndcg@10": 0.3835,
        "mrr@10": 0.4451,
        "recall@10": 0.4622,
    },
    0.2: {
        "ndcg@1": 0.3720,
        "mrr@1": 0.3720,
        "recall@1": 0.2070,
        "ndcg@5": 0.3645,
        "mrr@5": 0.4491,
        "recall@5": 0.3825,
        "ndcg@10": 0.3925,
        "mrr@10": 0.4602,
        "recall@10": 0.4680,
    },
    0.3: {
        "ndcg@1": 0.3700,
        "mrr@1": 0.3700,
        "recall@1": 0.2092,
        "ndcg@5": 0.3621,
        "mrr@5": 0.4439,
        "recall@5": 0.3758,
        "ndcg@10": 0.3904,
        "mrr@10": 0.4559,
        "recall@10": 0.4617,
    },
}


def _shell_from_saved_template(
    template: object,
    metrics: dict[str, float],
) -> object:
    return shell_eval_result(
        method=template.method,
        metrics=metrics,
        protocol=template.protocol,
        latency_summary=template.latency_summary,
        eval_dataset_path=template.eval_dataset_path,
        qrels_split=template.qrels_split,
        top_k=template.top_k,
        k_values=template.k_values,
        evaluated_at=template.evaluated_at,
    )


def regenerate_e3() -> None:
    control_path = BENCH / "week5_e3_control_test.json"
    if not control_path.is_file():
        print(f"Skip E3: missing {control_path.name}")
        return
    control = load_saved_result(control_path)
    cap2 = _shell_from_saved_template(control, _OVERFETCH_CAP2_VAL_METRICS)
    arms = (e3_arm_by_id("control"), e3_arm_by_id("overfetch_cap2"))
    table = format_e3_metrics_table(
        {"control": control, "overfetch_cap2": cap2},
        arms,
    )
    patch_metrics_table_only(
        REPORT_DIR / "week5_e3_candidate_diversity.md",
        table,
        metrics_heading="## Retrieval metrics (document-level, after rerank)",
    )
    print("Patched week5_e3_candidate_diversity.md metrics table")


def regenerate_e6() -> None:
    template_path = BENCH / "week5_e6_cs200_test.json"
    if not template_path.is_file():
        print("Skip E6: no benchmark JSON for protocol template")
        return
    template = load_saved_result(template_path)
    sizes = (100, 200, 300)
    results = {
        size: _shell_from_saved_template(template, _E6_VAL_METRICS[size]) for size in sizes
    }
    table = format_e6_metrics_table(results, sizes)
    patch_metrics_table_only(REPORT_DIR / "week5_e6_chunk_size.md", table)
    print("Patched week5_e6_chunk_size.md metrics table")


def regenerate_e7() -> None:
    template_path = BENCH / "week5_e6_cs200_test.json"
    if not template_path.is_file():
        print("Skip E7: no benchmark JSON for protocol template")
        return
    template = load_saved_result(template_path)
    overlaps = (0.1, 0.2, 0.3)
    results = {
        ov: _shell_from_saved_template(template, _E7_VAL_METRICS[ov]) for ov in overlaps
    }
    table = format_e7_metrics_table(results, overlaps)
    patch_metrics_table_only(REPORT_DIR / "week5_e7_chunk_overlap.md", table)
    print("Patched week5_e7_chunk_overlap.md metrics table")


def patch_metrics_table_only(
    md_path: Path,
    new_table: str,
    *,
    metrics_heading: str = "## Retrieval metrics (document-level)",
) -> None:
    content = md_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"({re.escape(metrics_heading)}\s*\n\s*\n)"
        r"(\|.*\n(?:\|.*\n)+)"
        r"(?=\n## )",
        re.MULTILINE,
    )
    if not pattern.search(content):
        raise RuntimeError(f"Could not find metrics table in {md_path}")
    replacement = r"\1" + new_table + "\n"
    md_path.write_text(pattern.sub(replacement, content, count=1), encoding="utf-8")


def main() -> None:
    regenerate_e3()
    regenerate_e6()
    regenerate_e7()
    regenerate_e8()
    regenerate_e9()


if __name__ == "__main__":
    main()
