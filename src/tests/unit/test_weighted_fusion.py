from app.ingestion.schemas import SearchHit
from app.retrieval.fusion import reciprocal_rank_fusion


def _hit(doc_id: str, chunk_id: int) -> SearchHit:
    return SearchHit(doc_id=doc_id, chunk_id=chunk_id, chunk_text="x", score=0.0)


def test_weighted_rrf_prefers_heavier_list():
    dense = [_hit("dense-only", 0)]
    bm25 = [_hit("bm25-only", 0)]
    equal = reciprocal_rank_fusion([dense, bm25], list_weights=[1.0, 1.0], limit=1)
    dense_heavy = reciprocal_rank_fusion([dense, bm25], list_weights=[1.0, 0.0], limit=1)
    bm25_heavy = reciprocal_rank_fusion([dense, bm25], list_weights=[0.0, 1.0], limit=1)
    assert equal[0].doc_id in {"dense-only", "bm25-only"}
    assert dense_heavy[0].doc_id == "dense-only"
    assert bm25_heavy[0].doc_id == "bm25-only"
