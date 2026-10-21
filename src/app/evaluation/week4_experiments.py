"""Week 4 experiment reports (full corpus, weight tuning, test baseline, rerank)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.evaluation.repro import try_git_commit_short
from app.evaluation.retrieval_eval import (
    RETRIEVAL_METHODS,
    RetrievalEvalResult,
    format_candidate_recall_summary,
    format_comparison_table,
    format_latency_table,
    render_query_examples,
)
from app.pipeline.rag import RetrievalMethod
from app.retrieval.fusion_weights import FusionWeights, load_fusion_weights

EXPERIMENT2_REPORT = "week4_exp2_full_corpus_validation.md"
EXPERIMENT3_REPORT = "week4_exp3_weighted_hybrid_validation.md"
EXPERIMENT4_REPORT = "week4_exp4_test_baseline.md"
EXPERIMENT5_REPORT = "week4_exp5_test_rerank.md"
MASTER_REPORT = "week4_retrieval_program_report.md"
PROTOCOL_ARTIFACT = "week4_eval_protocol.json"

TEST_METHODS: tuple[RetrievalMethod, ...] = ("dense", "bm25", "weighted_hybrid")


def render_labeled_eval_report(
    title: str,
    results: dict[RetrievalMethod, RetrievalEvalResult],
    *,
    collection_name: str,
    embedding_method: str,
    embedding_model: str,
    corpus_note: str,
    experiment_notes: list[str],
    example_count: int = 3,
    generated_at: datetime | None = None,
) -> str:
    when = generated_at or datetime.now(UTC)
    any_result = next(iter(results.values()))
    commit = any_result.protocol.git_commit or try_git_commit_short()
    lines: list[str] = [
        f"# {title}",
        "",
        f"**Generated:** {when.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Git commit:** `{commit or 'unknown'}`  ",
        f"**Index collection:** `{collection_name}`  ",
        f"**Dense encoder:** `{embedding_method}` / `{embedding_model}`  ",
        f"**Corpus / index:** {corpus_note}  ",
        f"**Qrels split:** `{any_result.qrels_split}`  ",
        f"**Eval dataset:** `{any_result.eval_dataset_path}`  ",
        f"**Retrieval depth (top-k):** `{any_result.top_k}`  ",
        f"**Queries evaluated:** `{any_result.aggregates.num_queries}`  ",
        "",
        "### Timing labels",
        "",
        "- **First-stage retrieval:** dense/BM25/hybrid search only; **embedding model load excluded** "
        "(construct embedder before timed region).",
        "- **Rerank:** cross-encoder or hosted reranker scoring (when `use_rerank` is true).",
        "- **Retrieve + rerank:** sum of the two stages; **excludes** stub LLM generation time.",
        "",
        "### Reproducibility protocol",
        "",
        f"```json\n{json.dumps(any_result.protocol.to_dict(), indent=2)}\n```",
        "",
    ]
    for note in experiment_notes:
        lines.append(note)
        lines.append("")
    lines.extend(
        [
            "## Quality metrics (mean over queries)",
            "",
            format_comparison_table(results, any_result.k_values),
            "",
            format_candidate_recall_summary(results),
            "",
            "## Stage latency (milliseconds)",
            "",
            format_latency_table(results),
            "",
            render_query_examples(results, example_count=example_count),
        ]
    )
    return "\n".join(lines)


def render_experiment2_report(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    **kwargs: Any,
) -> str:
    notes = [
        "**Experiment 2:** Full-corpus index; BEIR **validation** qrels. "
        "Compare **dense**, **BM25**, and equal-weight **hybrid (RRF)**. "
        "Each query runs retrieval + stub LLM with **citations** from top chunks."
    ]
    return render_labeled_eval_report(
        "Week 4 — Experiment 2 (full corpus, validation)",
        results,
        experiment_notes=notes,
        **kwargs,
    )


def render_experiment3_report(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    weights: FusionWeights,
    **kwargs: Any,
) -> str:
    dense, bm25 = weights.normalized()
    notes = [
        "**Experiment 3:** Grid-search RRF weights on **validation** to maximize "
        f"`{weights.metric_optimized}` (score **{weights.score_at_selection:.4f}**).",
        f"- **Selected weights (normalized):** dense={dense:.3f}, BM25={bm25:.3f}",
        "- Compare equal-weight **hybrid** vs **weighted hybrid** on validation.",
    ]
    return render_labeled_eval_report(
        "Week 4 — Experiment 3 (weighted hybrid tuning, validation)",
        results,
        experiment_notes=notes,
        **kwargs,
    )


def render_experiment4_report(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    weights: FusionWeights,
    **kwargs: Any,
) -> str:
    dense, bm25 = weights.normalized()
    notes = [
        "**Experiment 4:** Held-out **test** qrels; **no reranker**. "
        "Methods: **dense**, **BM25**, **weighted hybrid** (frozen weights from Exp 3).",
        f"- **Fusion weights:** dense={dense:.3f}, BM25={bm25:.3f}",
        "- Test set touched once for final baseline reporting.",
    ]
    return render_labeled_eval_report(
        "Week 4 — Experiment 4 (test baseline, no rerank)",
        results,
        experiment_notes=notes,
        **kwargs,
    )


def render_experiment5_report(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    weights: FusionWeights,
    reranker_name: str,
    **kwargs: Any,
) -> str:
    dense, bm25 = weights.normalized()
    notes = [
        "**Experiment 5:** Same as Experiment 4 plus **local cross-encoder rerank** "
        f"(`{reranker_name}`) before generation.",
        f"- **Fusion weights (frozen):** dense={dense:.3f}, BM25={bm25:.3f}",
        "- Compare rerank overhead via latency table vs Experiment 4.",
    ]
    return render_labeled_eval_report(
        "Week 4 — Experiment 5 (test + local rerank)",
        results,
        experiment_notes=notes,
        **kwargs,
    )


def render_master_program_report(
    *,
    benchmark_dir: Path,
    report_dir: Path,
    collection_name: str,
    embedding_method: str,
    embedding_model: str,
    corpus_note: str,
    generated_at: datetime | None = None,
) -> str:
    when = generated_at or datetime.now(UTC)
    weights_path = benchmark_dir / "week4_fusion_weights.json"
    weights_block = ""
    if weights_path.is_file():
        w = load_fusion_weights(weights_path)
        dense, bm25 = w.normalized()
        weights_block = (
            f"Validation-tuned weights: dense={dense:.3f}, BM25={bm25:.3f} "
            f"(optimized {w.metric_optimized}={w.score_at_selection:.4f})"
        )

    lines = [
        "# Week 4 — Retrieval program report",
        "",
        f"**Generated:** {when.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Collection:** `{collection_name}`  ",
        f"**Encoder:** `{embedding_method}` / `{embedding_model}`  ",
        f"**Corpus:** {corpus_note}  ",
        "",
        "## Experiment flow",
        "",
        "| Step | ID | Split | Methods | Rerank | Purpose |",
        "|------|-----|-------|---------|--------|---------|",
        "| 1 | Exp 1 | validation (1k slice) | dense, BM25, hybrid | no | "
        "Week 3 narrative / sanity |",
        "| 2 | Exp 2 | validation | dense, BM25, hybrid | no | Full-corpus labeled baseline |",
        "| 3 | Exp 3 | validation | hybrid, weighted hybrid | no | Tune RRF weights |",
        "| 4 | Exp 4 | test | dense, BM25, weighted hybrid | no | Frozen-weight test baseline |",
        "| 5 | Exp 5 | test | dense, BM25, weighted hybrid | **local CE** | Rerank vs Exp 4 |",
        "",
        "## Artifacts",
        "",
        f"- Protocol: `{benchmark_dir / PROTOCOL_ARTIFACT}`",
        f"- Weights: `{weights_path}`",
        f"- Per-experiment JSON: `{benchmark_dir}/week4_exp*.json`",
        f"- Per-experiment markdown: `{report_dir}/week4_exp*.md`",
        "",
        "## Learnings and trade-offs",
        "",
        "- **Dense vs lexical:** BM25 helps keyword-heavy FiQA queries; dense helps paraphrase. "
        "Hybrid/weighted fusion trades latency for recall on hard queries.",
        "- **Weight tuning (Exp 3):** Validation grid-search avoids hand-tuning RRF; "
        "weights must stay frozen before test (Exp 4/5).",
        f"- **{weights_block or 'Run Exp 3 to populate fusion weights.'}**",
        "- **Rerank (Exp 5):** Cross-encoder rescoring improves precision@k on many benchmarks "
        "but adds model load + per-query scoring cost (see p95 latency).",
        "- **Reproducibility:** Stub LLM (`CitationPreservingEvalLLM`) keeps "
        "generation deterministic; citations reflect retrieved chunks. "
        "Use Ollama/Cohere only for qualitative demos.",
        "- **Metrics vs latency:** NDCG/MRR/MAP from qrels; per-query latency on same hardware.",
        "",
        "## Linked reports",
        "",
        f"- [{EXPERIMENT2_REPORT}]({EXPERIMENT2_REPORT})",
        f"- [{EXPERIMENT3_REPORT}]({EXPERIMENT3_REPORT})",
        f"- [{EXPERIMENT4_REPORT}]({EXPERIMENT4_REPORT})",
        f"- [{EXPERIMENT5_REPORT}]({EXPERIMENT5_REPORT})",
        "- [week4_retrieval_eval.md](week4_retrieval_eval.md) (Experiment 1)",
        "- [week4_experiment_plan.md](week4_experiment_plan.md)",
        "",
    ]
    return "\n".join(lines)


def standard_eval_methods(include_weighted: bool = False) -> tuple[RetrievalMethod, ...]:
    if include_weighted:
        return (*RETRIEVAL_METHODS, "weighted_hybrid")
    return RETRIEVAL_METHODS


def load_saved_result(path: Path) -> RetrievalEvalResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    from app.evaluation.retrieval_eval import EvalProtocol, QueryEvalRow
    from app.evaluation.retrieval_metrics import QueryMetrics

    proto_raw = dict(payload["protocol"])
    proto_raw["k_values"] = tuple(proto_raw["k_values"])
    fw = proto_raw.get("fusion_weights")
    if fw is not None:
        proto_raw["fusion_weights"] = tuple(fw)
    proto_raw.setdefault("candidate_recall_k", 50)
    proto_raw.setdefault("git_commit", None)
    protocol = EvalProtocol(**proto_raw)
    rows = []
    for item in payload["per_query"]:
        rows.append(
            QueryEvalRow(
                query_id=item["query_id"],
                question=item["question"],
                ranked_doc_ids=tuple(item["ranked_doc_ids"]),
                relevant_doc_ids=tuple(item.get("relevant_doc_ids", [])),
                metrics=QueryMetrics(
                    ndcg={int(k): v for k, v in item["metrics"]["ndcg"].items()},
                    mrr={int(k): v for k, v in item["metrics"]["mrr"].items()},
                    map_score={int(k): v for k, v in item["metrics"]["map"].items()},
                    precision={int(k): v for k, v in item["metrics"]["precision"].items()},
                    recall={int(k): v for k, v in item["metrics"]["recall"].items()},
                    candidate_recall={
                        int(k): v
                        for k, v in item["metrics"].get("candidate_recall", {}).items()
                    },
                ),
                first_stage_latency_ms=float(
                    item.get(
                        "first_stage_latency_ms",
                        item.get("retrieval_latency_ms", 0),
                    )
                ),
                rerank_latency_ms=float(item.get("rerank_latency_ms", 0)),
                retrieval_latency_ms=float(item.get("retrieval_latency_ms", 0)),
                generation_latency_ms=item.get("generation_latency_ms"),
                answer=item.get("answer"),
                citations=tuple(item.get("citations", [])),
                top_hit_snippets=tuple(item.get("top_hit_snippets", [])),
            )
        )
    from app.evaluation.retrieval_metrics import AggregatedMetrics

    return RetrievalEvalResult(
        method=payload["method"],
        eval_dataset_path=payload["eval_dataset_path"],
        qrels_split=payload["qrels_split"],
        top_k=int(payload["top_k"]),
        k_values=tuple(payload["k_values"]),
        aggregates=AggregatedMetrics(
            values=payload["metrics"],
            num_queries=int(payload["num_queries"]),
        ),
        per_query=tuple(rows),
        evaluated_at=payload["evaluated_at"],
        protocol=protocol,
        latency_summary=payload.get("latency_summary", {}),
    )
