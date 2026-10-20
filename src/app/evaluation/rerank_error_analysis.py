"""Compare dense retrieval with vs without cross-encoder rerank (Exp 4 vs Exp 5)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.evaluation.fiqa_qrels import FiqaEvalQuery, load_fiqa_eval_dataset
from app.ingestion.schemas import SearchHit
from app.pipeline.rag import retrieve_hits, retrieve_with_optional_rerank
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.qdrant_store import QdrantChunkStore
from app.retrieval.reranker import RerankerProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RerankCause(StrEnum):
    MISSING_CANDIDATES = "missing_candidates"
    IRRELEVANT_ORDERING = "irrelevant_candidate_ordering"
    DUPLICATE_CHUNKS = "duplicate_chunks"
    QUERY_CHUNK_MISMATCH = "query_wording_chunk_mismatch"
    NO_MATERIAL_CHANGE = "no_material_change"


@dataclass(frozen=True)
class QueryRerankDelta:
    query_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    ndcg10_baseline: float
    ndcg10_rerank: float
    delta_ndcg10: float
    ranked_baseline: tuple[str, ...]
    ranked_rerank: tuple[str, ...]


@dataclass(frozen=True)
class ClassifiedQuery:
    query_id: str
    question: str
    direction: str  # helped | hurt | unchanged
    delta_ndcg10: float
    primary_cause: RerankCause
    detail: str
    ranked_baseline: tuple[str, ...]
    ranked_rerank: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]


def _token_jaccard(a: str, b: str) -> float:
    ta = set(_TOKEN_RE.findall(a.lower()))
    tb = set(_TOKEN_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _metric_at_k(metrics: dict[Any, float], k: int) -> float:
    return float(metrics.get(str(k), metrics.get(k, 0.0)))


def load_per_query_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["query_id"]: row for row in payload["per_query"]}


def build_deltas(
    baseline_path: Path,
    rerank_path: Path,
) -> list[QueryRerankDelta]:
    baseline = load_per_query_rows(baseline_path)
    rerank = load_per_query_rows(rerank_path)
    deltas: list[QueryRerankDelta] = []
    for query_id, base_row in baseline.items():
        rer_row = rerank[query_id]
        ndcg_b = _metric_at_k(base_row["metrics"]["ndcg"], 10)
        ndcg_r = _metric_at_k(rer_row["metrics"]["ndcg"], 10)
        deltas.append(
            QueryRerankDelta(
                query_id=query_id,
                question=base_row["question"],
                relevant_doc_ids=tuple(base_row["relevant_doc_ids"]),
                ndcg10_baseline=ndcg_b,
                ndcg10_rerank=ndcg_r,
                delta_ndcg10=ndcg_r - ndcg_b,
                ranked_baseline=tuple(base_row["ranked_doc_ids"]),
                ranked_rerank=tuple(rer_row["ranked_doc_ids"]),
            )
        )
    return deltas


def _relevant_rank(ranked: tuple[str, ...], relevant: set[str]) -> int | None:
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return index
    return None


def _chunks_per_doc_in_prefix(hits: list[SearchHit], n: int) -> Counter[str]:
    return Counter(str(hit.doc_id) for hit in hits[:n])


def classify_dense_rerank_query(
    query: FiqaEvalQuery,
    *,
    baseline_ranked: tuple[str, ...],
    rerank_ranked: tuple[str, ...],
    delta_ndcg10: float,
    dense_candidate_hits: list[SearchHit],
    reranked_hits: list[SearchHit],
    candidate_doc_limit: int,
) -> ClassifiedQuery:
    relevant = set(query.relevant_doc_ids)
    direction = "unchanged"
    if delta_ndcg10 > 1e-9:
        direction = "helped"
    elif delta_ndcg10 < -1e-9:
        direction = "hurt"

    candidate_docs = {str(hit.doc_id) for hit in dense_candidate_hits[:candidate_doc_limit]}
    missing_relevant = [doc_id for doc_id in relevant if doc_id not in candidate_docs]

    dup_counter = _chunks_per_doc_in_prefix(dense_candidate_hits, 10)
    duplicate_slots = sum(count - 1 for count in dup_counter.values() if count > 1)

    base_rank = _relevant_rank(baseline_ranked, relevant)
    rer_rank = _relevant_rank(rerank_ranked, relevant)

    baseline_top_text = dense_candidate_hits[0].chunk_text if dense_candidate_hits else ""
    rerank_top_text = reranked_hits[0].chunk_text if reranked_hits else ""
    overlap_base = _token_jaccard(query.question, baseline_top_text)
    overlap_rer = _token_jaccard(query.question, rerank_top_text)

    cause = RerankCause.NO_MATERIAL_CHANGE
    detail = "NDCG@10 unchanged."

    if direction == "unchanged":
        return ClassifiedQuery(
            query_id=query.query_id,
            question=query.question,
            direction=direction,
            delta_ndcg10=delta_ndcg10,
            primary_cause=cause,
            detail=detail,
            ranked_baseline=baseline_ranked,
            ranked_rerank=rerank_ranked,
            relevant_doc_ids=query.relevant_doc_ids,
        )

    if missing_relevant and base_rank is None and rer_rank is None:
        cause = RerankCause.MISSING_CANDIDATES
        detail = (
            f"Relevant doc(s) {missing_relevant} absent from dense top-{candidate_doc_limit} "
            f"candidate pool; rerank cannot recover them."
        )
    elif duplicate_slots >= 2 and direction == "hurt":
        cause = RerankCause.DUPLICATE_CHUNKS
        detail = (
            f"Dense top-10 chunk slots include {duplicate_slots} duplicate doc_id slot(s) "
            f"(same document repeated), shrinking candidate diversity before rerank."
        )
    elif base_rank is not None and rer_rank is not None and base_rank != rer_rank:
        cause = RerankCause.IRRELEVANT_ORDERING
        detail = (
            f"Relevant doc best rank moved from {base_rank} (dense top-{len(baseline_ranked)}) "
            f"to {rer_rank} (reranked top-{len(rerank_ranked)}) within the candidate pool."
        )
    elif base_rank is None and rer_rank is not None:
        cause = RerankCause.IRRELEVANT_ORDERING
        detail = (
            f"Relevant doc entered top-{len(rerank_ranked)} after rerank (was outside dense "
            f"top-{len(baseline_ranked)} doc ranking)."
        )
    elif base_rank is not None and rer_rank is None:
        cause = RerankCause.IRRELEVANT_ORDERING
        detail = (
            f"Relevant doc dropped out of top-{len(rerank_ranked)} after rerank "
            f"(was rank {base_rank} for dense)."
        )
    elif abs(overlap_rer - overlap_base) >= 0.08:
        cause = RerankCause.QUERY_CHUNK_MISMATCH
        detail = (
            f"Top-hit lexical overlap with query shifted (dense top-1 Jaccard={overlap_base:.3f} "
            f"→ rerank top-1 Jaccard={overlap_rer:.3f}); CE may favor wording over dense geometry."
        )
    elif missing_relevant and direction == "helped":
        cause = RerankCause.IRRELEVANT_ORDERING
        detail = (
            f"Metric improved but some relevant docs {missing_relevant} still missing from "
            f"top-{candidate_doc_limit} pool; gain from reordering among retrieved candidates."
        )
    else:
        cause = RerankCause.IRRELEVANT_ORDERING
        detail = "Ranking changed without candidate-pool or duplicate-chunk dominance."

    return ClassifiedQuery(
        query_id=query.query_id,
        question=query.question,
        direction=direction,
        delta_ndcg10=delta_ndcg10,
        primary_cause=cause,
        detail=detail,
        ranked_baseline=baseline_ranked,
        ranked_rerank=rerank_ranked,
        relevant_doc_ids=query.relevant_doc_ids,
    )


def analyze_dense_rerank(
    *,
    dataset_path: Path,
    baseline_result_path: Path,
    rerank_result_path: Path,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    reranker: RerankerProvider,
    candidate_limit: int = 50,
    top_k: int = 10,
    deep_dive_limit: int = 8,
) -> tuple[list[ClassifiedQuery], dict[str, Any]]:
    dataset = load_fiqa_eval_dataset(dataset_path)
    query_by_id = {q.query_id: q for q in dataset.queries}
    deltas = build_deltas(baseline_result_path, rerank_result_path)

    classified: list[ClassifiedQuery] = []
    for delta in deltas:
        query = query_by_id[delta.query_id]
        if abs(delta.delta_ndcg10) < 1e-9:
            classified.append(
                classify_dense_rerank_query(
                    query,
                    baseline_ranked=delta.ranked_baseline,
                    rerank_ranked=delta.ranked_rerank,
                    delta_ndcg10=delta.delta_ndcg10,
                    dense_candidate_hits=[],
                    reranked_hits=[],
                    candidate_doc_limit=candidate_limit,
                )
            )
            continue

        dense_hits = retrieve_hits(
            query.question,
            retrieval_method="dense",
            store=store,
            embedder=embedder,
            bm25_index=None,
            top_k=top_k,
            candidate_limit=candidate_limit,
        )
        reranked_hits, _, _ = retrieve_with_optional_rerank(
            query.question,
            retrieval_method="dense",
            store=store,
            embedder=embedder,
            bm25_index=None,
            top_k=top_k,
            use_rerank=True,
            reranker=reranker,
            rerank_candidate_limit=candidate_limit,
        )
        classified.append(
            classify_dense_rerank_query(
                query,
                baseline_ranked=delta.ranked_baseline,
                rerank_ranked=delta.ranked_rerank,
                delta_ndcg10=delta.delta_ndcg10,
                dense_candidate_hits=dense_hits,
                reranked_hits=reranked_hits,
                candidate_doc_limit=candidate_limit,
            )
        )

    helped = sorted(
        [c for c in classified if c.direction == "helped"],
        key=lambda c: c.delta_ndcg10,
        reverse=True,
    )
    hurt = sorted(
        [c for c in classified if c.direction == "hurt"],
        key=lambda c: c.delta_ndcg10,
    )

    def _group_counts(items: list[ClassifiedQuery]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            counts[item.primary_cause.value] += 1
        return dict(counts)

    summary: dict[str, Any] = {
        "baseline_artifact": str(baseline_result_path),
        "rerank_artifact": str(rerank_result_path),
        "candidate_limit": candidate_limit,
        "top_k": top_k,
        "num_queries": len(classified),
        "num_helped": len(helped),
        "num_hurt": len(hurt),
        "num_unchanged": sum(1 for c in classified if c.direction == "unchanged"),
        "helped_by_cause": _group_counts(helped),
        "hurt_by_cause": _group_counts(hurt),
        "top_helped": [c.query_id for c in helped[:deep_dive_limit]],
        "top_hurt": [c.query_id for c in hurt[:deep_dive_limit]],
    }
    return classified, summary


def render_rerank_error_analysis_report(
    classified: list[ClassifiedQuery],
    summary: dict[str, Any],
    *,
    deep_dive_limit: int = 8,
) -> str:
    by_id = {c.query_id: c for c in classified}
    helped_ids = summary["top_helped"]
    hurt_ids = summary["top_hurt"]

    lines = [
        "## Dense rerank error analysis (Exp 4 vs Exp 5)",
        "",
        "Comparison uses **dense retrieval only**: Experiment 4 (no rerank) vs Experiment 5 "
        f"(CE rerank, top-{summary['candidate_limit']} candidates → top-{summary['top_k']}).",
        "",
        "### Summary",
        "",
        f"- Queries: **{summary['num_queries']}** | "
        f"helped (ΔNDCG@10>0): **{summary['num_helped']}** | "
        f"hurt: **{summary['num_hurt']}** | unchanged: **{summary['num_unchanged']}**",
        "",
        "**Helped — primary cause counts**",
        "",
    ]
    lines.extend(_cause_table(summary["helped_by_cause"]))
    lines.extend(
        [
            "",
            "**Hurt — primary cause counts**",
            "",
        ]
    )
    lines.extend(_cause_table(summary["hurt_by_cause"]))
    lines.extend(
        [
            "",
            "### Cause definitions",
            "",
            "| Cause | Meaning |",
            "|-------|---------|",
            "| `missing_candidates` | Relevant doc not in dense top-N pool |",
            "| `irrelevant_candidate_ordering` | Relevant doc in pool; @10 rank changes after CE |",
            "| `duplicate_chunks` | Same `doc_id` in multiple of first 10 chunk slots |",
            "| `query_wording_chunk_mismatch` | CE shifts top hit lexical overlap with query |",
            "",
            f"### Top {deep_dive_limit} helped queries",
            "",
        ]
    )
    for query_id in helped_ids:
        lines.extend(_format_case(by_id[query_id]))
    lines.extend([f"### Top {deep_dive_limit} hurt queries", ""])
    for query_id in hurt_ids:
        lines.extend(_format_case(by_id[query_id]))
    return "\n".join(lines)


def _cause_table(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["| Cause | Count |", "|-------|-------|", "| _(none)_ | 0 |"]
    lines = ["| Cause | Count |", "|-------|-------|"]
    for cause, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{cause}` | {count} |")
    return lines


def _short_doc_list(ranked: tuple[str, ...]) -> str:
    head = ", ".join(ranked[:5])
    return f"{head}..." if len(ranked) > 5 else head


def _format_case(item: ClassifiedQuery) -> list[str]:
    q_short = item.question[:100] + ("..." if len(item.question) > 100 else "")
    return [
        f"#### `{item.query_id}` (ΔNDCG@10={item.delta_ndcg10:+.4f}, "
        f"**{item.primary_cause.value}**)",
        "",
        f"**Q:** {q_short}",
        "",
        f"**Relevant docs:** {', '.join(item.relevant_doc_ids)}",
        "",
        f"- Dense @10: {_short_doc_list(item.ranked_baseline)}",
        f"- Rerank @10: {_short_doc_list(item.ranked_rerank)}",
        f"- **Diagnosis:** {item.detail}",
        "",
    ]


def append_analysis_to_report(report_path: Path, section_md: str) -> None:
    text = report_path.read_text(encoding="utf-8")
    marker = "## Dense rerank error analysis"
    if marker in text:
        text = text.split(marker, maxsplit=1)[0].rstrip()
    report_path.write_text(text + "\n\n" + section_md.strip() + "\n", encoding="utf-8")
