import math

from app.evaluation.retrieval_metrics import (
    aggregate_query_metrics,
    average_precision_at_k,
    compute_query_metrics,
    dcg_at_k,
    doc_ranking_from_hits,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.ingestion.schemas import SearchHit


def test_doc_ranking_from_hits_dedupes_by_first_rank():
    hits = [
        SearchHit(doc_id="a", chunk_id=0, chunk_text="x", score=1.0),
        SearchHit(doc_id="b", chunk_id=0, chunk_text="y", score=0.9),
        SearchHit(doc_id="a", chunk_id=1, chunk_text="z", score=0.8),
    ]
    assert doc_ranking_from_hits(hits) == ["a", "b"]


def test_perfect_binary_ranking_metrics():
    ranked = ["r1", "r2", "noise"]
    relevance = {"r1": 1, "r2": 1}
    relevant = {"r1", "r2"}

    assert math.isclose(precision_at_k(ranked, relevant, 1), 1.0)
    assert math.isclose(recall_at_k(ranked, relevant, 1), 0.5)
    assert math.isclose(mrr_at_k(ranked, relevant, 5), 1.0)
    assert math.isclose(ndcg_at_k(ranked, relevance, 2), 1.0)
    assert math.isclose(average_precision_at_k(ranked, relevant, 2), 1.0)


def test_mrr_zero_when_no_relevant_in_cutoff():
    ranked = ["x", "y"]
    assert mrr_at_k(ranked, {"z"}, 2) == 0.0


def test_dcg_and_ndcg_graded():
    ranked = ["a", "b", "c"]
    relevance = {"a": 2, "b": 1}
    assert dcg_at_k([2.0, 1.0, 0.0], 3) > 0.0
    assert 0.0 < ndcg_at_k(ranked, relevance, 3) <= 1.0


def test_compute_and_aggregate_query_metrics():
    q1 = compute_query_metrics(["a", "b"], {"a": 1}, k_values=(1, 5))
    q2 = compute_query_metrics(["x", "a"], {"a": 1}, k_values=(1, 5))
    agg = aggregate_query_metrics([q1, q2], k_values=(1, 5))
    assert agg.num_queries == 2
    assert "ndcg@1" in agg.values
    assert "recall@5" in agg.values
