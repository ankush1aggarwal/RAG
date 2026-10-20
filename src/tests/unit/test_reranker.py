from unittest.mock import MagicMock

import pytest

from app.ingestion.schemas import SearchHit
from app.pipeline.rag import retrieve_with_optional_rerank
from app.retrieval.reranker import resolve_rerank_candidate_limit


class _FakeReranker:
    name = "fake:reranker"

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        reversed_hits = list(reversed(hits[:top_k]))
        return [hit.model_copy(update={"score": 99.0}) for hit in reversed_hits]


def test_resolve_rerank_candidate_limit():
    assert resolve_rerank_candidate_limit(3, default_candidate_limit=50) == 50
    assert resolve_rerank_candidate_limit(3, candidate_limit=10) == 10


def test_retrieve_with_optional_rerank_reorders():
    store = MagicMock()
    embedder = MagicMock()
    store.search_text.return_value = [
        SearchHit(doc_id="a", chunk_id=0, chunk_text="first", score=1.0),
        SearchHit(doc_id="b", chunk_id=0, chunk_text="second", score=0.5),
    ]
    hits, name, count = retrieve_with_optional_rerank(
        "q",
        retrieval_method="dense",
        store=store,
        embedder=embedder,
        bm25_index=None,
        top_k=2,
        use_rerank=True,
        reranker=_FakeReranker(),
        rerank_candidate_limit=2,
    )
    assert name == "fake:reranker"
    assert count == 2
    assert hits[0].doc_id == "b"
    assert hits[0].score == 99.0


def test_retrieve_with_optional_rerank_requires_provider():
    with pytest.raises(ValueError, match="reranker is required"):
        retrieve_with_optional_rerank(
            "q",
            retrieval_method="dense",
            store=MagicMock(),
            embedder=MagicMock(),
            bm25_index=None,
            top_k=1,
            use_rerank=True,
            reranker=None,
        )
