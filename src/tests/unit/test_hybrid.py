from unittest.mock import MagicMock

from app.ingestion.schemas import SearchHit
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import hybrid_search_text


def test_hybrid_search_text_fuses_dense_and_bm25():
    store = MagicMock()
    store.search_text.return_value = [
        SearchHit(doc_id="1", chunk_id=0, chunk_text="a", score=0.9),
        SearchHit(doc_id="2", chunk_id=0, chunk_text="b", score=0.5),
    ]
    bm25 = MagicMock()
    bm25.search.return_value = [
        SearchHit(doc_id="1", chunk_id=0, chunk_text="a", score=3.0),
        SearchHit(doc_id="3", chunk_id=1, chunk_text="c", score=2.0),
    ]
    embedder = MagicMock()

    hits = hybrid_search_text(
        "q",
        store=store,
        bm25_index=bm25,
        embedder=embedder,
        limit=2,
        candidate_limit=2,
    )

    store.search_text.assert_called_once_with("q", embedder, limit=2)
    bm25.search.assert_called_once_with("q", limit=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "1" and hits[0].chunk_id == 0
    assert hits[0].score > hits[1].score


def test_hybrid_with_empty_bm25_matches_dense_only_rrf():
    dense_hits = [
        SearchHit(doc_id="1", chunk_id=0, chunk_text="a", score=0.9),
        SearchHit(doc_id="2", chunk_id=1, chunk_text="b", score=0.5),
    ]
    store = MagicMock()
    store.search_text.return_value = dense_hits
    bm25 = MagicMock()
    bm25.search.return_value = []
    embedder = MagicMock()

    hybrid_hits = hybrid_search_text(
        "q",
        store=store,
        bm25_index=bm25,
        embedder=embedder,
        limit=2,
        candidate_limit=2,
    )
    expected = reciprocal_rank_fusion([dense_hits], k=60, limit=2)

    assert [(h.doc_id, h.chunk_id, h.score) for h in hybrid_hits] == [
        (h.doc_id, h.chunk_id, h.score) for h in expected
    ]


def test_hybrid_with_empty_dense_matches_bm25_only_rrf():
    bm25_hits = [SearchHit(doc_id="7", chunk_id=2, chunk_text="c", score=1.5)]
    store = MagicMock()
    store.search_text.return_value = []
    bm25 = MagicMock()
    bm25.search.return_value = bm25_hits
    embedder = MagicMock()

    hybrid_hits = hybrid_search_text(
        "q",
        store=store,
        bm25_index=bm25,
        embedder=embedder,
        limit=1,
        candidate_limit=3,
    )
    expected = reciprocal_rank_fusion([[], bm25_hits], k=60, limit=1)

    assert [(h.doc_id, h.chunk_id, h.score) for h in hybrid_hits] == [
        (h.doc_id, h.chunk_id, h.score) for h in expected
    ]
