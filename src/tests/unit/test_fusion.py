import pytest

from app.ingestion.schemas import SearchHit
from app.retrieval.fusion import reciprocal_rank_fusion


def _hit(doc_id: str, chunk_id: int, text: str = "x") -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        doc_title="",
        chunk_id=chunk_id,
        chunk_text=text,
        score=0.0,
    )


def test_rrf_merges_duplicate_chunk_and_sums_contributions():
    dense = [_hit("1", 0), _hit("2", 0)]
    bm25 = [_hit("1", 0), _hit("3", 1)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=10)

    scores = {(h.doc_id, h.chunk_id): h.score for h in fused}
    assert scores[("1", 0)] == (1 / 61) + (1 / 61)
    assert scores[("2", 0)] == 1 / 62
    assert scores[("3", 1)] == 1 / 62
    assert fused[0].doc_id == "1" and fused[0].chunk_id == 0


def test_rrf_stable_tie_break_by_doc_id_and_chunk_id():
    dense = [_hit("10", 0)]
    bm25 = [_hit("2", 0)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=2)
    assert [(h.doc_id, h.chunk_id) for h in fused] == [("10", 0), ("2", 0)]


def test_rrf_same_score_orders_by_doc_id_then_chunk_id():
    a = [_hit("b", 1)]
    b = [_hit("a", 9)]
    fused = reciprocal_rank_fusion([a, b], k=60, limit=2)
    assert fused[0].doc_id == "a"
    assert fused[1].doc_id == "b"


def test_rrf_respects_limit():
    dense = [_hit("1", 0), _hit("2", 0), _hit("3", 0)]
    fused = reciprocal_rank_fusion([dense], k=60, limit=2)
    assert len(fused) == 2


def test_rrf_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit must be at least 1"):
        reciprocal_rank_fusion([[]], limit=0)


def test_rrf_empty_inputs_return_empty_list():
    assert reciprocal_rank_fusion([], limit=5) == []
    assert reciprocal_rank_fusion([[], []], limit=5) == []
    assert reciprocal_rank_fusion([[], []], k=60, limit=1) == []


def test_rrf_one_nonempty_list_one_empty_matches_single_list():
    dense = [_hit("1", 0), _hit("2", 1)]
    from_single = reciprocal_rank_fusion([dense], k=60, limit=2)
    from_pair = reciprocal_rank_fusion([dense, []], k=60, limit=2)
    assert [(h.doc_id, h.chunk_id, h.score) for h in from_single] == [
        (h.doc_id, h.chunk_id, h.score) for h in from_pair
    ]


def test_rrf_single_list_preserves_dense_rank_order():
    """Dense-only path: RRF with one list keeps input ranking (scores become RRF)."""
    dense = [_hit("1", 0), _hit("2", 1), _hit("3", 0)]
    fused = reciprocal_rank_fusion([dense], k=60, limit=3)

    assert [(h.doc_id, h.chunk_id) for h in fused] == [("1", 0), ("2", 1), ("3", 0)]
    assert fused[0].score == 1 / 61
    assert fused[1].score == 1 / 62
    assert fused[2].score == 1 / 63


def test_rrf_overlap_emits_each_chunk_once():
    dense = [_hit("1", 0), _hit("2", 0)]
    bm25 = [_hit("1", 0), _hit("3", 1)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=10)
    keys = [(h.doc_id, h.chunk_id) for h in fused]

    assert len(keys) == len(set(keys))
    assert keys.count(("1", 0)) == 1


def test_rrf_overlap_outranks_single_list_only_chunk():
    only_dense = [_hit("9", 0)]
    overlap = [_hit("1", 0)]
    dense = [_hit("1", 0), _hit("9", 0)]
    bm25 = [_hit("1", 0), _hit("2", 0)]

    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=3)
    assert fused[0].doc_id == "1" and fused[0].chunk_id == 0
    assert fused[0].score > reciprocal_rank_fusion([only_dense], k=60, limit=1)[0].score


def test_rrf_deterministic_across_repeated_calls():
    dense = [_hit("2", 1), _hit("1", 0)]
    bm25 = [_hit("1", 0), _hit("3", 2)]

    first = reciprocal_rank_fusion([dense, bm25], k=60, limit=5)
    second = reciprocal_rank_fusion([dense, bm25], k=60, limit=5)

    assert [(h.doc_id, h.chunk_id, h.score) for h in first] == [
        (h.doc_id, h.chunk_id, h.score) for h in second
    ]


def test_rrf_preserves_metadata_from_first_occurrence():
    dense = [
        SearchHit(doc_id="1", doc_title="", chunk_id=0, chunk_text="first text", score=0.1),
    ]
    bm25 = [
        SearchHit(doc_id="1", doc_title="Title", chunk_id=0, chunk_text="ignored", score=9.0),
    ]
    fused = reciprocal_rank_fusion([dense, bm25], k=60, limit=1)

    assert fused[0].chunk_text == "first text"
    assert fused[0].doc_title == "Title"
