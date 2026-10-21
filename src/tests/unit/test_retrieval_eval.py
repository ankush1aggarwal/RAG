from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.eval_llm import CitationPreservingEvalLLM
from app.evaluation.fiqa_qrels import FiqaEvalDataset, FiqaEvalQuery
from app.evaluation.retrieval_eval import (
    evaluate_all_methods,
    evaluate_retrieval_dataset,
    format_latency_table,
    render_query_examples,
    required_top_k,
    save_eval_result,
    summarize_latency_ms,
)
from app.evaluation.e8_embedding_model import (
    DEFAULT_ALTERNATIVE_MODEL,
    format_e8_metrics_table,
)
from app.evaluation.week4_experiments import load_saved_result
from app.ingestion.schemas import SearchHit


def test_required_top_k_at_least_max_metric_cutoff():
    assert required_top_k((1, 5, 10), 3) == 10
    assert required_top_k((1, 5, 10), 20) == 20


@patch("app.evaluation.retrieval_eval.retrieve_hits")
def test_evaluate_retrieval_dataset_aggregates(mock_retrieve: MagicMock):
    mock_retrieve.return_value = [
        SearchHit(doc_id="42", chunk_id=0, chunk_text="hit", score=1.0),
    ]
    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="validation",
        queries=(
            FiqaEvalQuery(
                query_id="1",
                question="fees?",
                relevant_doc_ids=("42",),
                relevance_by_doc_id={"42": 1},
            ),
        ),
    )
    result = evaluate_retrieval_dataset(
        dataset,
        method="dense",
        store=MagicMock(),
        embedder=MagicMock(),
        bm25_index=None,
        top_k=10,
        progress_every=0,
    )
    assert result.aggregates.num_queries == 1
    assert result.aggregates.values["ndcg@1"] == 1.0
    assert result.aggregates.values["mrr@1"] == 1.0
    assert result.latency_summary["first_stage_retrieval"]["mean_ms"] >= 0.0
    assert "candidate_recall@50" in result.aggregates.values


@patch("app.evaluation.retrieval_eval.retrieve_hits")
def test_evaluate_retrieval_dataset_with_stub_llm_includes_citations(mock_retrieve: MagicMock):
    mock_retrieve.return_value = [
        SearchHit(
            doc_id="42",
            doc_title="Fees",
            chunk_id=3,
            chunk_text="Broker fees vary.",
            score=1.0,
        ),
    ]
    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="validation",
        queries=(
            FiqaEvalQuery(
                query_id="1",
                question="fees?",
                relevant_doc_ids=("42",),
                relevance_by_doc_id={"42": 1},
            ),
        ),
    )
    result = evaluate_retrieval_dataset(
        dataset,
        method="dense",
        store=MagicMock(),
        embedder=MagicMock(),
        bm25_index=None,
        top_k=10,
        progress_every=0,
        llm=CitationPreservingEvalLLM(),
    )
    row = result.per_query[0]
    assert row.answer is not None
    assert len(row.citations) == 1
    assert row.citations[0]["document_id"] == "42"
    assert row.citations[0]["chunk_id"] == 3
    assert result.protocol.include_rag_responses is True


def test_summarize_latency_ms():
    summary = summarize_latency_ms([10.0, 20.0, 30.0])
    assert summary["mean_ms"] == 20.0
    assert summary["p50_ms"] == 20.0


@patch("app.evaluation.retrieval_eval.evaluate_retrieval_dataset")
def test_evaluate_all_methods_invokes_each_method(mock_eval: MagicMock):
    mock_eval.return_value = MagicMock()
    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="validation",
        queries=(),
    )
    evaluate_all_methods(
        dataset,
        store=MagicMock(),
        embedder=MagicMock(),
        bm25_index=MagicMock(),
        methods=("dense", "bm25"),
        fusion_weights=(0.5, 0.5),
    )
    assert mock_eval.call_count == 2
    assert mock_eval.call_args_list[0].kwargs["method"] == "dense"


def test_evaluate_all_methods_weighted_requires_fusion_weights():
    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="validation",
        queries=(),
    )
    with pytest.raises(ValueError, match="fusion_weights required"):
        evaluate_all_methods(
            dataset,
            store=MagicMock(),
            embedder=MagicMock(),
            bm25_index=MagicMock(),
            methods=("weighted_hybrid",),
            fusion_weights=None,
        )


@patch("app.evaluation.retrieval_eval.retrieve_hits")
def test_save_and_load_eval_result_roundtrip(mock_retrieve: MagicMock, tmp_path):
    mock_retrieve.return_value = [
        SearchHit(doc_id="1", chunk_id=0, chunk_text="x", score=1.0),
    ]
    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="test",
        queries=(
            FiqaEvalQuery(
                query_id="1",
                question="q",
                relevant_doc_ids=("1",),
                relevance_by_doc_id={"1": 1},
            ),
        ),
    )
    path = tmp_path / "eval.json"
    result = evaluate_retrieval_dataset(
        dataset,
        method="dense",
        store=MagicMock(),
        embedder=MagicMock(),
        bm25_index=None,
        eval_dataset_path=str(path),
        progress_every=0,
    )
    save_eval_result(path, result)
    loaded = load_saved_result(path)
    assert loaded.per_query[0].ranked_doc_ids == ("1",)


def test_format_latency_and_query_examples():
    from app.evaluation.retrieval_eval import EvalProtocol, QueryEvalRow, RetrievalEvalResult
    from app.evaluation.retrieval_metrics import AggregatedMetrics, QueryMetrics

    metrics = QueryMetrics(
        ndcg={10: 1.0},
        mrr={10: 1.0},
        map_score={10: 1.0},
        precision={10: 1.0},
        recall={10: 1.0},
        candidate_recall={50: 1.0},
    )
    row = QueryEvalRow(
        query_id="1",
        question="fees?",
        ranked_doc_ids=("9",),
        relevant_doc_ids=("9",),
        metrics=metrics,
        first_stage_latency_ms=5.0,
        rerank_latency_ms=0.0,
        retrieval_latency_ms=5.0,
        generation_latency_ms=None,
        answer=None,
        citations=(),
        top_hit_snippets=("hi",),
    )
    protocol = EvalProtocol(
        top_k=10,
        k_values=(10,),
        generation_top_k=5,
        use_rerank=False,
        reranker=None,
        rerank_candidate_limit=50,
        fusion_weights=None,
        llm_label="none",
        include_rag_responses=False,
    )
    res = RetrievalEvalResult(
        method="dense",
        eval_dataset_path="p",
        qrels_split="validation",
        top_k=10,
        k_values=(10,),
        aggregates=AggregatedMetrics(values={"ndcg@10": 1.0}, num_queries=1),
        per_query=(row,),
        evaluated_at="t",
        protocol=protocol,
        latency_summary={
            "first_stage_retrieval": {"mean_ms": 5.0, "p50_ms": 5.0, "p95_ms": 5.0},
            "rerank": {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
            "retrieve_and_rerank": {"mean_ms": 5.0, "p50_ms": 5.0, "p95_ms": 5.0},
        },
    )
    table = format_latency_table({"dense": res})
    assert "5.0" in table
    examples = render_query_examples({"dense": res}, example_count=1)
    assert "fees?" in examples


def test_e8_saved_validation_json_surfaces_candidate_recall_in_table():
    path = Path("data/benchmarks/week5_e8_bge_base_en_v1_5_validation.json")
    if not path.is_file():
        pytest.skip("E8 validation artifact not present")
    result = load_saved_result(path)
    table = format_e8_metrics_table(
        {DEFAULT_ALTERNATIVE_MODEL: result},
        (DEFAULT_ALTERNATIVE_MODEL,),
    )
    assert "CR@50" in table
    cr = result.aggregates.values["candidate_recall@50"]
    assert f"{cr:.4f}" in table
