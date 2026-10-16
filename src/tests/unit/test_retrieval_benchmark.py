import json

from app.evaluation.retrieval_benchmark import (
    benchmark_bm25_overlap_eligibility_pass,
    chunk_keys,
    jaccard_similarity,
    load_query_set,
    percentile,
)
from app.ingestion.schemas import ChunkRecord, SearchHit
from app.retrieval.bm25_store import Bm25ChunkIndex


def test_jaccard_identical_sets():
    keys = {("1", 0), ("2", 1)}
    assert jaccard_similarity(keys, keys) == 1.0


def test_jaccard_disjoint():
    left = {("1", 0)}
    right = {("2", 0)}
    assert jaccard_similarity(left, right) == 0.0


def test_chunk_keys_respects_limit():
    hits = [
        SearchHit(doc_id="a", chunk_id=0, chunk_text="x", score=1.0),
        SearchHit(doc_id="b", chunk_id=1, chunk_text="y", score=0.5),
    ]
    assert chunk_keys(hits, 1) == {("a", 0)}


def test_percentile_p95():
    samples = [float(i) for i in range(1, 101)]
    assert percentile(samples, 50) == 50.5
    assert 94.0 <= percentile(samples, 95) <= 96.0


def test_bm25_overlap_microbench_reports_speedup_on_synthetic_index():
    chunks = [
        ChunkRecord(
            doc_id="0",
            chunk_id=0,
            chunk_text="broker fees and commission",
            word_len=4,
            char_len=28,
            doc_word_len=4,
        ),
        ChunkRecord(
            doc_id="1",
            chunk_id=0,
            chunk_text="unrelated mortgage topic",
            word_len=3,
            char_len=24,
            doc_word_len=3,
        ),
    ]
    index = Bm25ChunkIndex.from_chunks(chunks)
    bench = benchmark_bm25_overlap_eligibility_pass(index, "broker fees", repeats=5)
    assert bench.indexed_chunks == 2
    assert bench.legacy_mean_ms > 0
    assert bench.precomputed_mean_ms > 0
    assert bench.speedup >= 1.0


def test_load_query_set_from_repo_fixture(tmp_path):
    payload = {
        "version": 1,
        "description": "test",
        "corpus": "BeIR/fiqa",
        "corpus_limit": 1000,
        "collection": "fiqa_sample",
        "top_k": 3,
        "queries": [{"id": "q01", "theme": "x", "question": "hello?"}],
    }
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    spec = load_query_set(path)
    assert spec.top_k == 3
    assert spec.queries[0].question == "hello?"
