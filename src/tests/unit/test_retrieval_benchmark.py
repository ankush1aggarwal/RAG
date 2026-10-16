import json

from app.evaluation.retrieval_benchmark import (
    chunk_keys,
    jaccard_similarity,
    load_query_set,
)
from app.ingestion.schemas import SearchHit


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
