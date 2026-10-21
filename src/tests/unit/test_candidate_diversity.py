from app.ingestion.schemas import SearchHit
from app.retrieval.candidate_diversity import (
    E3Arm,
    duplicate_slot_count,
    e3_arm_by_id,
    pool_diagnostics,
    shape_candidate_pool,
)


def _hit(doc_id: str, chunk_id: int) -> SearchHit:
    return SearchHit(doc_id=doc_id, chunk_id=chunk_id, chunk_text="t", score=1.0)


def test_control_truncates_to_budget():
    arm = E3Arm(arm_id="c", label="", mode="control", rerank_budget=3)
    hits = [_hit("a", 0), _hit("a", 1), _hit("b", 0), _hit("c", 0)]
    shaped = shape_candidate_pool(hits, arm)
    assert len(shaped) == 3
    assert [h.doc_id for h in shaped] == ["a", "a", "b"]


def test_dedup_no_backfill():
    arm = E3Arm(arm_id="d", label="", mode="dedup_no_backfill", rerank_budget=50)
    hits = [_hit("a", 0), _hit("a", 1), _hit("b", 0)]
    shaped = shape_candidate_pool(hits, arm)
    assert [h.chunk_id for h in shaped] == [0, 0]


def test_overfetch_cap_one_fills_budget():
    arm = E3Arm(
        arm_id="o",
        label="",
        mode="overfetch_cap",
        rerank_budget=3,
        per_doc_cap=1,
    )
    hits = [
        _hit("a", 0),
        _hit("a", 1),
        _hit("b", 0),
        _hit("c", 0),
        _hit("d", 0),
    ]
    shaped = shape_candidate_pool(hits, arm)
    assert len(shaped) == 3
    assert [h.doc_id for h in shaped] == ["a", "b", "c"]


def test_duplicate_slot_metrics():
    hits = [_hit("a", 0), _hit("a", 1), _hit("b", 0)]
    assert duplicate_slot_count(hits) == 1
    diag = pool_diagnostics(hits)
    assert diag.unique_doc_count == 2
    assert diag.duplicate_slot_rate == 1 / 3


def test_e3_arm_by_id():
    assert e3_arm_by_id("control").mode == "control"
