import math

from app.evaluation.retrieval_metrics import (
    AggregatedMetrics,
    aggregate_query_metrics,
    average_precision_at_k,
    compute_query_metrics,
    dcg_at_k,
    doc_ranking_from_hits,
    format_retrieval_metrics_markdown_table,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.retrieval_eval import EvalProtocol
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


def test_format_metrics_table_shows_cr_at_pool_not_at_k():
    protocol = EvalProtocol(
        top_k=10,
        k_values=(1, 5, 10),
        generation_top_k=10,
        use_rerank=False,
        reranker=None,
        rerank_candidate_limit=50,
        fusion_weights=None,
        llm_label="none",
        include_rag_responses=False,
        candidate_recall_k=50,
    )

    class _Row:
        def __init__(self, values: dict[str, float]):
            self.protocol = protocol
            self.aggregates = AggregatedMetrics(values=values, num_queries=1)

    table = format_retrieval_metrics_markdown_table(
        "Arm",
        [
            (
                "dense",
                _Row(
                    {
                        "ndcg@10": 0.4,
                        "candidate_recall@50": 0.658,
                    }
                ),
            )
        ],
    )
    assert "CR@50" in table
    assert "0.6580" in table
    assert "| dense | Cand. recall | — | — | — | 0.6580 |" in table


def test_candidate_recall_at_k_on_pool_ranking():
    ranked_final = ["noise", "a"]
    pool = ["a", "b", "noise"]
    metrics = compute_query_metrics(
        ranked_final,
        {"a": 1, "b": 1},
        k_values=(10,),
        candidate_ranked_doc_ids=pool,
        candidate_recall_k=50,
    )
    assert metrics.candidate_recall[50] == 1.0
    agg = aggregate_query_metrics([metrics], k_values=(10,))
    assert agg.values["candidate_recall@50"] == 1.0
