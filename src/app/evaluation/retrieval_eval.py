"""Run labeled FiQA retrieval evaluation against a prepared qrels JSON file."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.evaluation.fiqa_qrels import FiqaEvalDataset, load_fiqa_eval_dataset
from app.evaluation.repro import repo_relative_path, try_git_commit_short
from app.evaluation.retrieval_benchmark import percentile
from app.evaluation.retrieval_metrics import (
    DEFAULT_K_VALUES,
    AggregatedMetrics,
    QueryMetrics,
    aggregate_query_metrics,
    compute_query_metrics,
    doc_ranking_from_hits,
)
from app.generation.llm import LLMProvider
from app.generation.prompt import build_rag_prompt
from app.pipeline.rag import RetrievalMethod, citations_from_hits, retrieve_hits
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.candidate_diversity import E3Arm, pool_diagnostics, shape_candidate_pool
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.qdrant_store import QdrantChunkStore
from app.retrieval.reranker import RerankerProvider, resolve_rerank_candidate_limit

RETRIEVAL_METHODS: tuple[RetrievalMethod, ...] = ("dense", "bm25", "hybrid")
METHOD_LABELS: dict[RetrievalMethod, str] = {
    "dense": "dense (Qdrant)",
    "bm25": "BM25",
    "hybrid": "hybrid (RRF)",
    "weighted_hybrid": "weighted hybrid (RRF)",
}

RetrieveFn = Literal["dense", "bm25", "hybrid"]


@dataclass(frozen=True)
class EvalProtocol:
    """Frozen run configuration persisted with each experiment artifact."""

    top_k: int
    k_values: tuple[int, ...]
    generation_top_k: int
    use_rerank: bool
    reranker: str | None
    rerank_candidate_limit: int
    fusion_weights: tuple[float, float] | None
    llm_label: str
    include_rag_responses: bool
    candidate_recall_k: int = 50
    git_commit: str | None = None
    e3_arm_id: str | None = None
    e3_overfetch_limit: int | None = None
    e3_per_doc_cap: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "k_values": list(self.k_values),
            "generation_top_k": self.generation_top_k,
            "use_rerank": self.use_rerank,
            "reranker": self.reranker,
            "rerank_candidate_limit": self.rerank_candidate_limit,
            "fusion_weights": (
                list(self.fusion_weights) if self.fusion_weights is not None else None
            ),
            "llm_label": self.llm_label,
            "include_rag_responses": self.include_rag_responses,
            "candidate_recall_k": self.candidate_recall_k,
            "git_commit": self.git_commit,
            "e3_arm_id": self.e3_arm_id,
            "e3_overfetch_limit": self.e3_overfetch_limit,
            "e3_per_doc_cap": self.e3_per_doc_cap,
        }


@dataclass(frozen=True)
class QueryEvalRow:
    query_id: str
    question: str
    ranked_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    metrics: QueryMetrics
    first_stage_latency_ms: float
    rerank_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float | None
    answer: str | None
    citations: tuple[dict[str, Any], ...]
    top_hit_snippets: tuple[str, ...]
    candidate_pool_size: int | None = None
    candidate_pool_unique_docs: int | None = None
    candidate_pool_duplicate_slots: int | None = None
    candidate_pool_duplicate_slot_rate: float | None = None


@dataclass(frozen=True)
class RetrievalEvalResult:
    method: RetrievalMethod
    eval_dataset_path: str
    qrels_split: str
    top_k: int
    k_values: tuple[int, ...]
    aggregates: AggregatedMetrics
    per_query: tuple[QueryEvalRow, ...]
    evaluated_at: str
    protocol: EvalProtocol
    latency_summary: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "eval_dataset_path": self.eval_dataset_path,
            "qrels_split": self.qrels_split,
            "top_k": self.top_k,
            "k_values": list(self.k_values),
            "evaluated_at": self.evaluated_at,
            "num_queries": self.aggregates.num_queries,
            "metrics": self.aggregates.values,
            "protocol": self.protocol.to_dict(),
            "latency_summary": self.latency_summary,
            "per_query": [
                {
                    "query_id": row.query_id,
                    "question": row.question,
                    "ranked_doc_ids": list(row.ranked_doc_ids),
                    "relevant_doc_ids": list(row.relevant_doc_ids),
                    "first_stage_latency_ms": row.first_stage_latency_ms,
                    "rerank_latency_ms": row.rerank_latency_ms,
                    "retrieval_latency_ms": row.retrieval_latency_ms,
                    "generation_latency_ms": row.generation_latency_ms,
                    "answer": row.answer,
                    "citations": list(row.citations),
                    "top_hit_snippets": list(row.top_hit_snippets),
                    "candidate_pool_size": row.candidate_pool_size,
                    "candidate_pool_unique_docs": row.candidate_pool_unique_docs,
                    "candidate_pool_duplicate_slots": row.candidate_pool_duplicate_slots,
                    "candidate_pool_duplicate_slot_rate": row.candidate_pool_duplicate_slot_rate,
                    "metrics": {
                        "ndcg": row.metrics.ndcg,
                        "mrr": row.metrics.mrr,
                        "map": row.metrics.map_score,
                        "precision": row.metrics.precision,
                        "recall": row.metrics.recall,
                        "candidate_recall": row.metrics.candidate_recall,
                    },
                }
                for row in self.per_query
            ],
        }


def summarize_latency_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 95),
    }


def required_top_k(k_values: tuple[int, ...], requested_top_k: int) -> int:
    return max(requested_top_k, max(k_values))


def evaluate_retrieval_dataset(
    dataset: FiqaEvalDataset,
    *,
    method: RetrievalMethod,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int = 10,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    eval_dataset_path: str = "",
    progress_every: int = 50,
    fusion_weights: tuple[float, float] | None = None,
    use_rerank: bool = False,
    reranker: RerankerProvider | None = None,
    rerank_candidate_limit: int = 50,
    candidate_recall_k: int = 50,
    generation_top_k: int | None = None,
    llm: LLMProvider | None = None,
    include_rag_responses: bool = True,
    git_commit: str | None = None,
    e3_arm: E3Arm | None = None,
) -> RetrievalEvalResult:
    if method in ("bm25", "hybrid", "weighted_hybrid") and bm25_index is None:
        raise ValueError("BM25 index is required for bm25 and hybrid evaluation")
    if use_rerank and reranker is None:
        raise ValueError("reranker is required when use_rerank is True")
    if method == "weighted_hybrid" and fusion_weights is None:
        raise ValueError("fusion_weights required for weighted_hybrid")

    effective_k = required_top_k(k_values, top_k)
    gen_k = generation_top_k if generation_top_k is not None else min(top_k, effective_k)
    weights = fusion_weights if method == "weighted_hybrid" else None
    protocol = EvalProtocol(
        top_k=effective_k,
        k_values=k_values,
        generation_top_k=gen_k,
        use_rerank=use_rerank,
        reranker=reranker.name if reranker is not None else None,
        rerank_candidate_limit=rerank_candidate_limit,
        fusion_weights=weights,
        llm_label=getattr(llm, "label", type(llm).__name__) if llm else "none",
        include_rag_responses=include_rag_responses and llm is not None,
        candidate_recall_k=candidate_recall_k,
        git_commit=git_commit or try_git_commit_short(),
        e3_arm_id=e3_arm.arm_id if e3_arm is not None else None,
        e3_overfetch_limit=e3_arm.overfetch_limit if e3_arm is not None else None,
        e3_per_doc_cap=e3_arm.per_doc_cap if e3_arm is not None else None,
    )

    if use_rerank:
        if e3_arm is not None:
            pool_limit = e3_arm.first_stage_retrieve_limit()
        else:
            pool_limit = resolve_rerank_candidate_limit(
                effective_k,
                candidate_limit=rerank_candidate_limit,
            )
    else:
        pool_limit = max(effective_k, candidate_recall_k)

    per_query_metrics: list[QueryMetrics] = []
    rows: list[QueryEvalRow] = []
    first_stage_latencies: list[float] = []
    rerank_latencies: list[float] = []
    retrieve_and_rerank_latencies: list[float] = []

    for index, query in enumerate(dataset.queries, start=1):
        stage_start = time.perf_counter()
        hits = retrieve_hits(
            query.question,
            retrieval_method=method,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=effective_k,
            fusion_weights=weights,
            candidate_limit=pool_limit,
        )
        first_stage_ms = (time.perf_counter() - stage_start) * 1000.0
        if e3_arm is not None:
            hits = shape_candidate_pool(hits, e3_arm)
        pool_diag = pool_diagnostics(hits) if use_rerank else None
        candidate_ranked = doc_ranking_from_hits(hits)

        rerank_ms = 0.0
        if use_rerank and reranker is not None:
            rerank_start = time.perf_counter()
            hits = reranker.rerank(query.question, hits, top_k=effective_k)
            rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        total_retrieve_ms = first_stage_ms + rerank_ms
        first_stage_latencies.append(first_stage_ms)
        rerank_latencies.append(rerank_ms)
        retrieve_and_rerank_latencies.append(total_retrieve_ms)

        ranked = doc_ranking_from_hits(hits)
        metrics = compute_query_metrics(
            ranked,
            query.relevance_by_doc_id,
            k_values=k_values,
            candidate_ranked_doc_ids=candidate_ranked,
            candidate_recall_k=candidate_recall_k,
        )
        per_query_metrics.append(metrics)

        answer: str | None = None
        generation_ms: float | None = None
        citation_payloads: list[dict[str, Any]] = []
        if include_rag_responses and llm is not None:
            prompt_hits = hits[:gen_k]
            gen_start = time.perf_counter()
            prompt = build_rag_prompt(query.question, prompt_hits)
            answer = llm.generate(prompt.user, system=prompt.system)
            generation_ms = (time.perf_counter() - gen_start) * 1000.0
            citation_payloads = [
                {
                    "document_id": cite.document_id,
                    "chunk_id": cite.chunk_id,
                    "title": cite.title,
                }
                for cite in citations_from_hits(prompt_hits)
            ]

        snippets = tuple(
            (hit.chunk_text[:120] + ("..." if len(hit.chunk_text) > 120 else ""))
            for hit in hits[:3]
        )
        rows.append(
            QueryEvalRow(
                query_id=query.query_id,
                question=query.question,
                ranked_doc_ids=tuple(ranked),
                relevant_doc_ids=query.relevant_doc_ids,
                metrics=metrics,
                first_stage_latency_ms=first_stage_ms,
                rerank_latency_ms=rerank_ms,
                retrieval_latency_ms=total_retrieve_ms,
                generation_latency_ms=generation_ms,
                answer=answer,
                citations=tuple(citation_payloads),
                top_hit_snippets=snippets,
                candidate_pool_size=pool_diag.pool_size if pool_diag else None,
                candidate_pool_unique_docs=pool_diag.unique_doc_count if pool_diag else None,
                candidate_pool_duplicate_slots=pool_diag.duplicate_slots if pool_diag else None,
                candidate_pool_duplicate_slot_rate=(
                    pool_diag.duplicate_slot_rate if pool_diag else None
                ),
            )
        )
        if progress_every > 0 and index % progress_every == 0:
            print(f"  evaluated {index}/{dataset.num_queries} queries...")

    aggregates = aggregate_query_metrics(per_query_metrics, k_values=k_values)
    rel_path = repo_relative_path(eval_dataset_path) if eval_dataset_path else ""
    latency_summary: dict[str, Any] = {
        "first_stage_retrieval": summarize_latency_ms(first_stage_latencies),
        "rerank": summarize_latency_ms(rerank_latencies),
        "retrieve_and_rerank": summarize_latency_ms(retrieve_and_rerank_latencies),
        "model_load_in_timed_region": False,
    }
    return RetrievalEvalResult(
        method=method,
        eval_dataset_path=rel_path,
        qrels_split=dataset.qrels_split,
        top_k=effective_k,
        k_values=k_values,
        aggregates=aggregates,
        per_query=tuple(rows),
        evaluated_at=datetime.now(UTC).isoformat(),
        protocol=protocol,
        latency_summary=latency_summary,
    )


def load_eval_dataset(path: str | Path) -> FiqaEvalDataset:
    return load_fiqa_eval_dataset(path)


def save_eval_result(path: str | Path, result: RetrievalEvalResult) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def evaluate_all_methods(
    dataset: FiqaEvalDataset,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int = 10,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    eval_dataset_path: str = "",
    methods: tuple[RetrievalMethod, ...] = RETRIEVAL_METHODS,
    progress_every: int = 50,
    fusion_weights: tuple[float, float] | None = None,
    use_rerank: bool = False,
    reranker: RerankerProvider | None = None,
    rerank_candidate_limit: int = 50,
    generation_top_k: int | None = None,
    llm: LLMProvider | None = None,
) -> dict[RetrievalMethod, RetrievalEvalResult]:
    results: dict[RetrievalMethod, RetrievalEvalResult] = {}
    for method in methods:
        print(f"\n=== method: {METHOD_LABELS[method]} ===")
        weights = fusion_weights if method == "weighted_hybrid" else None
        if method == "weighted_hybrid" and fusion_weights is None:
            raise ValueError("fusion_weights required when methods include weighted_hybrid")
        results[method] = evaluate_retrieval_dataset(
            dataset,
            method=method,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=top_k,
            k_values=k_values,
            eval_dataset_path=eval_dataset_path,
            progress_every=progress_every,
            fusion_weights=weights,
            use_rerank=use_rerank,
            reranker=reranker,
            rerank_candidate_limit=rerank_candidate_limit,
            generation_top_k=generation_top_k,
            llm=llm,
        )
    return results


def _latency_block(summary: dict[str, float]) -> tuple[float, float, float]:
    return summary["mean_ms"], summary["p50_ms"], summary["p95_ms"]


def format_latency_table(results: dict[RetrievalMethod, RetrievalEvalResult]) -> str:
    any_rerank = any(r.protocol.use_rerank for r in results.values())
    if any_rerank:
        lines = [
            "| Method | Stage | mean (ms) | p50 (ms) | p95 (ms) |",
            "|--------|-------|-----------|----------|----------|",
        ]
        for method, result in results.items():
            lat = result.latency_summary
            label = METHOD_LABELS.get(method, method)
            for stage_key, stage_label in (
                ("first_stage_retrieval", "first-stage retrieval"),
                ("rerank", "rerank"),
                ("retrieve_and_rerank", "retrieve + rerank (total)"),
            ):
                block = lat.get(stage_key)
                if not isinstance(block, dict):
                    continue
                mean_m, p50, p95 = _latency_block(block)
                lines.append(
                    f"| {label} | {stage_label} | {mean_m:.1f} | {p50:.1f} | {p95:.1f} |"
                )
        return "\n".join(lines)

    lines = [
        "| Method | Stage | mean (ms) | p50 (ms) | p95 (ms) |",
        "|--------|-------|-----------|----------|----------|",
    ]
    for method, result in results.items():
        lat = result.latency_summary
        label = METHOD_LABELS.get(method, method)
        block = lat.get("first_stage_retrieval", lat)
        if isinstance(block, dict) and "mean_ms" in block:
            mean_m, p50, p95 = _latency_block(block)
        else:
            mean_m, p50, p95 = _latency_block(lat)
        lines.append(
            f"| {label} | first-stage retrieval | {mean_m:.1f} | {p50:.1f} | {p95:.1f} |"
        )
    return "\n".join(lines)


def render_query_examples(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    *,
    example_count: int = 3,
) -> str:
    if not results:
        return ""
    first = next(iter(results.values()))
    lines = ["## Per-query examples", ""]
    for query_index in range(min(example_count, len(first.per_query))):
        sample = first.per_query[query_index]
        lines.append(f"### `{sample.query_id}` — {sample.question[:80]}")
        lines.append("")
        lines.append(f"**Qrels relevant docs:** {', '.join(sample.relevant_doc_ids)}")
        lines.append("")
        for method, result in results.items():
            row = result.per_query[query_index]
            label = METHOD_LABELS.get(method, method)
            ndcg10 = row.metrics.ndcg.get(10, 0.0)
            cite_str = ", ".join(
                f"[{c['document_id']}:{c['chunk_id']}]" for c in row.citations
            )
            if row.rerank_latency_ms > 0:
                lines.append(
                    f"**{label}** — first-stage {row.first_stage_latency_ms:.1f} ms, "
                    f"rerank {row.rerank_latency_ms:.1f} ms, NDCG@10 {ndcg10:.3f}"
                )
            else:
                lines.append(
                    f"**{label}** — first-stage retrieval {row.first_stage_latency_ms:.1f} ms, "
                    f"NDCG@10 {ndcg10:.3f}"
                )
            lines.append(f"- Ranked docs (top 5): {', '.join(row.ranked_doc_ids[:5])}")
            lines.append(f"- Citations in response: {cite_str or '_(none)_'}")
            if row.top_hit_snippets:
                lines.append(f"- Top snippet: {row.top_hit_snippets[0]}")
            lines.append("")
    return "\n".join(lines)


def format_comparison_table(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> str:
    """Side-by-side mean metrics for dense / BM25 / hybrid."""
    lines = [
        "| Method | Metric | " + " | ".join(f"@{k}" for k in k_values) + " |",
        "|--------|--------|" + "|".join(["------"] * len(k_values)) + "|",
    ]
    for method in METHOD_LABELS:
        if method not in results:
            continue
        aggregates = results[method].aggregates.values
        label = METHOD_LABELS[method]
        for metric_label, prefix in (
            ("NDCG", "ndcg"),
            ("MRR", "mrr"),
            ("MAP", "map"),
            ("Precision", "precision"),
            ("Recall", "recall"),
        ):
            cells = [f"{aggregates.get(f'{prefix}@{k}', 0.0):.4f}" for k in k_values]
            lines.append(f"| {label} | {metric_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_candidate_recall_summary(
    results: dict[RetrievalMethod, RetrievalEvalResult],
) -> str:
    lines = [
        "### Candidate pool recall (document-level)",
        "",
        "Computed on the **first-stage** ranked document list (depth = "
        "`max(top_k, candidate_recall_k)` or rerank candidate limit). "
        "Does not use reranked order.",
        "",
        "| Method | Candidate Recall@K |",
        "|--------|-------------------:|",
    ]
    for method, result in results.items():
        k = result.protocol.candidate_recall_k
        key = f"candidate_recall@{k}"
        value = result.aggregates.values.get(key)
        if value is None:
            continue
        label = METHOD_LABELS.get(method, method)
        lines.append(f"| {label} | {value:.4f} (@{k}) |")
    return "\n".join(lines)


def render_week4_comparison_report(
    results: dict[RetrievalMethod, RetrievalEvalResult],
    *,
    collection_name: str,
    embedding_method: str,
    embedding_model: str,
    corpus_note: str,
    week3_report_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> str:
    when = generated_at or datetime.now(UTC)
    any_result = next(iter(results.values()))
    k_values = any_result.k_values

    lines: list[str] = [
        "# Week 4 — Retrieval evaluation (qrels) vs Week 3 baselines",
        "",
        f"**Generated:** {when.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Index collection:** `{collection_name}`  ",
        f"**Dense encoder:** `{embedding_method}` / `{embedding_model}`  ",
        f"**Corpus / index:** {corpus_note}  ",
        f"**Qrels split:** `{any_result.qrels_split}`  ",
        f"**Eval dataset:** `{any_result.eval_dataset_path}`  ",
        f"**Retrieval depth (top-k):** `{any_result.top_k}`  ",
        f"**Queries evaluated:** `{any_result.aggregates.num_queries}`  ",
        "",
        "## Week 4 — Labeled metrics (mean over queries)",
        "",
        "Document-level ranking from chunk hits (first-seen `doc_id`). "
        "Graded NDCG uses BEIR relevance scores.",
        "",
        format_comparison_table(results, k_values),
        "",
        "## Week 3 baseline reference (not directly comparable)",
        "",
        (
            "Week 3 [`week3_retrieval_comparison.md`](week3_retrieval_comparison.md) used a "
            "**fixed 10-query** set, **top-k=5**, and **unlabeled** signals (latency p50/p95, "
            "top-k Jaccard overlap). Those numbers measure **speed** and **ranking diversity**, "
            "not NDCG/MRR."
        ),
        "",
    ]

    week3_path = Path(week3_report_path) if week3_report_path else None
    if week3_path is not None and week3_path.is_file():
        lines.extend(
            [
                "### Week 3 latency (ms, p50) — 1k-doc `fiqa_sample`, 10 queries",
                "",
                "| Method | p50 |",
                "|--------|-----|",
                "| dense (Qdrant) | 18.5 |",
                "| BM25 | 23.4 |",
                "| hybrid (RRF) | 41.9 |",
                "",
                "### Week 3 top-k overlap (mean Jaccard @5)",
                "",
                "- Dense vs BM25: **0.19**",
                "- Dense vs hybrid: **0.39**",
                "- BM25 vs hybrid: **0.46**",
                "",
            ]
        )
    else:
        lines.append(
            "_Week 3 report file not found; see "
            "`docs/reports/week3_retrieval_comparison.md`._"
        )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "- Compare **NDCG@10 / MRR@10** across methods on the **same** qrels slice; "
                "higher is better."
            ),
            (
                "- Week 3 **hybrid** had moderate Jaccard vs dense (0.39) but was **~2× slower**; "
                "Week 4 shows whether hybrid improves **labeled** retrieval on your index."
            ),
            (
                "- Partial corpus indexes filter qrels to queries with ≥1 relevant doc in-index; "
                "full-corpus ingest + `prepare_fiqa_eval.py` yields complete validation/test sets."
            ),
            "- Reserve **test** qrels for final reporting only.",
            "",
        ]
    )
    return "\n".join(lines)


def format_metrics_table(aggregates: AggregatedMetrics, k_values: tuple[int, ...]) -> str:
    lines = [
        f"Queries evaluated: {aggregates.num_queries}",
        "",
        "| Metric | " + " | ".join(f"@{k}" for k in k_values) + " |",
        "|--------|" + "|".join(["------"] * len(k_values)) + "|",
    ]
    for label, prefix in (
        ("NDCG", "ndcg"),
        ("MRR", "mrr"),
        ("MAP", "map"),
        ("Precision", "precision"),
        ("Recall", "recall"),
    ):
        cells = [f"{aggregates.values.get(f'{prefix}@{k}', 0.0):.4f}" for k in k_values]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
