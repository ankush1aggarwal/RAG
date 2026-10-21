"""Document-level retrieval metrics (NDCG, MRR, MAP, precision, recall)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True)
class QueryMetrics:
    """Per-query metric values keyed by cutoff k."""

    ndcg: dict[int, float]
    mrr: dict[int, float]
    map_score: dict[int, float]
    precision: dict[int, float]
    recall: dict[int, float]
    candidate_recall: dict[int, float]


@dataclass(frozen=True)
class AggregatedMetrics:
    """Mean metrics across queries; field names like ``ndcg@5``."""

    values: dict[str, float]
    num_queries: int


def doc_ranking_from_hits(hits: Iterable[object]) -> list[str]:
    """First-seen ``doc_id`` order from chunk hits (retrieval rank)."""
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
    return ordered


def _grades_for_ranking(
    ranked_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> list[float]:
    return [float(relevance_by_doc_id.get(doc_id, 0)) for doc_id in ranked_doc_ids[:k]]


def dcg_at_k(relevances: list[float], k: int) -> float:
    total = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(index + 1)
    return total


def ndcg_at_k(
    ranked_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    retrieved = _grades_for_ranking(ranked_doc_ids, relevance_by_doc_id, k)
    ideal = sorted(relevance_by_doc_id.values(), reverse=True)
    dcg = dcg_at_k(retrieved, k)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    top = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_doc_ids)
    return hits / k


def recall_at_k(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    if not relevant_doc_ids:
        return 0.0
    top = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr_at_k(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    ranked_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    if not relevant_doc_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_doc_ids)


def compute_query_metrics(
    ranked_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    *,
    candidate_ranked_doc_ids: list[str] | None = None,
    candidate_recall_k: int | None = None,
) -> QueryMetrics:
    relevant = {doc_id for doc_id, grade in relevance_by_doc_id.items() if grade > 0}
    ndcg: dict[int, float] = {}
    mrr: dict[int, float] = {}
    map_score: dict[int, float] = {}
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    candidate_recall: dict[int, float] = {}
    for k in k_values:
        ndcg[k] = ndcg_at_k(ranked_doc_ids, relevance_by_doc_id, k)
        mrr[k] = mrr_at_k(ranked_doc_ids, relevant, k)
        map_score[k] = average_precision_at_k(ranked_doc_ids, relevant, k)
        precision[k] = precision_at_k(ranked_doc_ids, relevant, k)
        recall[k] = recall_at_k(ranked_doc_ids, relevant, k)
    if candidate_ranked_doc_ids is not None and candidate_recall_k is not None:
        candidate_recall[candidate_recall_k] = recall_at_k(
            candidate_ranked_doc_ids,
            relevant,
            candidate_recall_k,
        )
    return QueryMetrics(
        ndcg=ndcg,
        mrr=mrr,
        map_score=map_score,
        precision=precision,
        recall=recall,
        candidate_recall=candidate_recall,
    )


def candidate_recall_at_pool(
    aggregate_values: dict[str, float],
    candidate_recall_k: int,
) -> float | None:
    key = f"candidate_recall@{candidate_recall_k}"
    if key not in aggregate_values:
        return None
    return float(aggregate_values[key])


def format_retrieval_metrics_markdown_table(
    first_col_header: str,
    entries: list[tuple[str, object]],
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    protocol_attr: str = "protocol",
    aggregates_attr: str = "aggregates",
) -> str:
    """Markdown table with @k columns plus CR@{pool} (first-stage candidate recall)."""
    if not entries:
        return ""
    first_result = entries[0][1]
    protocol = getattr(first_result, protocol_attr)
    pool_k = int(getattr(protocol, "candidate_recall_k", 50))
    metric_headers = [f"@{k}" for k in k_values] + [f"CR@{pool_k}"]
    lines = [
        f"| {first_col_header} | Metric | " + " | ".join(metric_headers) + " |",
        "|" + "-" * len(first_col_header) + "|--------|"
        + "|".join(["------"] * len(metric_headers)) + "|",
    ]
    metric_rows = (
        ("NDCG", "ndcg"),
        ("MRR", "mrr"),
        ("Recall", "recall"),
        ("Cand. recall", "candidate_recall"),
    )
    for label, result in entries:
        aggregates = getattr(result, aggregates_attr).values
        for metric_label, prefix in metric_rows:
            if prefix == "candidate_recall":
                cr = candidate_recall_at_pool(aggregates, pool_k)
                cr_cell = f"{cr:.4f}" if cr is not None else "n/a"
                cells = ["—"] * len(k_values) + [cr_cell]
            else:
                cells = [f"{aggregates.get(f'{prefix}@{k}', 0.0):.4f}" for k in k_values] + ["—"]
            lines.append(f"| {label} | {metric_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate_query_metrics(
    per_query: list[QueryMetrics],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> AggregatedMetrics:
    if not per_query:
        return AggregatedMetrics(values={}, num_queries=0)

    totals: dict[str, float] = {}
    count = len(per_query)
    for k in k_values:
        totals[f"ndcg@{k}"] = sum(row.ndcg[k] for row in per_query) / count
        totals[f"mrr@{k}"] = sum(row.mrr[k] for row in per_query) / count
        totals[f"map@{k}"] = sum(row.map_score[k] for row in per_query) / count
        totals[f"precision@{k}"] = sum(row.precision[k] for row in per_query) / count
        totals[f"recall@{k}"] = sum(row.recall[k] for row in per_query) / count
    candidate_ks = {
        k for row in per_query for k in row.candidate_recall
    }
    for k in sorted(candidate_ks):
        totals[f"candidate_recall@{k}"] = (
            sum(row.candidate_recall.get(k, 0.0) for row in per_query) / count
        )
    return AggregatedMetrics(values=totals, num_queries=count)
