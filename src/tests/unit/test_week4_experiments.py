from pathlib import Path

from app.evaluation.retrieval_eval import (
    EvalProtocol,
    QueryEvalRow,
    RetrievalEvalResult,
    save_eval_result,
)
from app.evaluation.retrieval_metrics import AggregatedMetrics, QueryMetrics
from app.evaluation.week4_experiments import (
    TEST_METHODS,
    load_saved_result,
    render_experiment4_report,
    render_master_program_report,
    standard_eval_methods,
)
from app.retrieval.fusion_weights import FusionWeights


def _minimal_result(method: str = "dense") -> RetrievalEvalResult:
    metrics = QueryMetrics(
        ndcg={10: 0.5},
        mrr={10: 0.4},
        map_score={10: 0.3},
        precision={10: 0.2},
        recall={10: 0.6},
        candidate_recall={50: 1.0},
    )
    row = QueryEvalRow(
        query_id="q1",
        question="What are ETF fees?",
        ranked_doc_ids=("doc-a", "doc-b"),
        relevant_doc_ids=("doc-a",),
        metrics=metrics,
        first_stage_latency_ms=12.5,
        rerank_latency_ms=0.0,
        retrieval_latency_ms=12.5,
        generation_latency_ms=1.0,
        answer="[eval] stub",
        citations=({"document_id": "doc-a", "chunk_id": 0, "title": "Fees"},),
        top_hit_snippets=("snippet one",),
    )
    protocol = EvalProtocol(
        top_k=10,
        k_values=(1, 5, 10),
        generation_top_k=5,
        use_rerank=False,
        reranker=None,
        rerank_candidate_limit=50,
        fusion_weights=None,
        llm_label="eval_stub",
        include_rag_responses=True,
    )
    return RetrievalEvalResult(
        method=method,  # type: ignore[arg-type]
        eval_dataset_path="data/benchmarks/fiqa_qrels_test.json",
        qrels_split="test",
        top_k=10,
        k_values=(1, 5, 10),
        aggregates=AggregatedMetrics(values={"ndcg@10": 0.5}, num_queries=1),
        per_query=(row,),
        evaluated_at="2026-01-01T00:00:00+00:00",
        protocol=protocol,
        latency_summary={
            "first_stage_retrieval": {"mean_ms": 12.5, "p50_ms": 12.5, "p95_ms": 12.5},
            "rerank": {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0},
            "retrieve_and_rerank": {"mean_ms": 12.5, "p50_ms": 12.5, "p95_ms": 12.5},
        },
    )


def test_standard_eval_methods():
    assert standard_eval_methods(include_weighted=False) == ("dense", "bm25", "hybrid")
    assert standard_eval_methods(include_weighted=True)[-1] == "weighted_hybrid"
    assert TEST_METHODS == ("dense", "bm25", "weighted_hybrid")


def test_render_experiment4_report_includes_examples_and_latency():
    results = {
        "dense": _minimal_result("dense"),
        "bm25": _minimal_result("bm25"),
        "weighted_hybrid": _minimal_result("weighted_hybrid"),
    }
    weights = FusionWeights(dense=0.7, bm25=0.3, score_at_selection=0.55)
    text = render_experiment4_report(
        results,
        weights,
        collection_name="fiqa",
        embedding_method="sentence_transformer",
        embedding_model="all-MiniLM-L6-v2",
        corpus_note="full corpus",
        example_count=1,
    )
    assert "Per-query examples" in text
    assert "Stage latency" in text
    assert "Git commit" in text
    assert "doc-a" in text
    assert "0.700" in text or "dense=0.7" in text


def test_render_master_program_report_links_experiments(tmp_path: Path):
    text = render_master_program_report(
        benchmark_dir=tmp_path,
        report_dir=tmp_path,
        collection_name="fiqa",
        embedding_method="sentence_transformer",
        embedding_model="all-MiniLM-L6-v2",
        corpus_note="note",
    )
    assert "Experiment flow" in text
    assert "week4_exp5_test_rerank.md" in text


def test_load_saved_result_roundtrip(tmp_path: Path):
    path = tmp_path / "result.json"
    original = _minimal_result("dense")
    save_eval_result(path, original)
    loaded = load_saved_result(path)
    assert loaded.method == original.method
    assert loaded.per_query[0].citations == original.per_query[0].citations
    assert loaded.protocol.llm_label == "eval_stub"
