from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.schemas import SearchHit
from app.pipeline.rag import answer_query, citations_from_hits, retrieve_hits


class _RecordingLLM:
    def __init__(self) -> None:
        self.last_user: str | None = None
        self.last_system: str | None = None

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.last_user = prompt
        self.last_system = system
        return "Generated answer."


def _hit(doc_id: str = "7", chunk_id: int = 2) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        doc_title="Fees",
        chunk_id=chunk_id,
        chunk_text="Broker fees vary.",
        score=0.81,
    )


def test_citations_from_hits_maps_fields():
    cites = citations_from_hits([_hit()])
    assert len(cites) == 1
    assert cites[0].document_id == "7"
    assert cites[0].chunk_id == 2
    assert cites[0].title == "Fees"


def test_answer_query_wires_retrieval_prompt_and_llm():
    store = MagicMock()
    store.search_text.return_value = [_hit()]
    embedder = MagicMock()
    llm = _RecordingLLM()

    result = answer_query(
        "What are broker fees?",
        store=store,
        embedder=embedder,
        llm=llm,
        top_k=4,
    )

    store.search_text.assert_called_once_with(
        "What are broker fees?",
        embedder,
        limit=4,
    )
    assert result.answer == "Generated answer."
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "7"
    assert result.hits[0].score == 0.81
    assert llm.last_system is not None
    assert "Question: What are broker fees?" in (llm.last_user or "")
    assert "Broker fees vary." in (llm.last_user or "")


def test_answer_query_strips_question_whitespace():
    store = MagicMock()
    store.search_text.return_value = []
    embedder = MagicMock()
    llm = _RecordingLLM()

    answer_query("  hello  ", store=store, embedder=embedder, llm=llm)

    store.search_text.assert_called_once_with("hello", embedder, limit=3)


def test_answer_query_rejects_empty_question():
    with pytest.raises(ValueError, match="question must be non-empty"):
        answer_query("  ", store=MagicMock(), embedder=MagicMock(), llm=_RecordingLLM())


def test_answer_query_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        answer_query(
            "q",
            store=MagicMock(),
            embedder=MagicMock(),
            llm=_RecordingLLM(),
            top_k=0,
        )


def test_answer_query_bm25_uses_index_not_dense():
    store = MagicMock()
    bm25 = MagicMock()
    bm25.search.return_value = [_hit(doc_id="9", chunk_id=1)]
    embedder = MagicMock()
    llm = _RecordingLLM()

    result = answer_query(
        "broker fees",
        store=store,
        embedder=embedder,
        llm=llm,
        retrieval_method="bm25",
        bm25_index=bm25,
    )

    store.search_text.assert_not_called()
    bm25.search.assert_called_once_with("broker fees", limit=3)
    assert result.retrieval_method == "bm25"
    assert result.citations[0].document_id == "9"


def test_answer_query_hybrid_calls_fusion_path():
    store = MagicMock()
    store.search_text.return_value = [_hit()]
    bm25 = MagicMock()
    bm25.search.return_value = [_hit(doc_id="8", chunk_id=0)]
    embedder = MagicMock()
    llm = _RecordingLLM()

    with patch("app.pipeline.rag.hybrid_search_text") as hybrid:
        hybrid.return_value = [_hit(doc_id="fused", chunk_id=3)]
        result = answer_query(
            "fees",
            store=store,
            embedder=embedder,
            llm=llm,
            top_k=5,
            retrieval_method="hybrid",
            bm25_index=bm25,
        )

    hybrid.assert_called_once()
    assert hybrid.call_args.kwargs["limit"] == 5
    assert result.retrieval_method == "hybrid"
    assert result.hits[0].doc_id == "fused"


def test_retrieve_hits_bm25_requires_index():
    with pytest.raises(ValueError, match="BM25 index is not available"):
        retrieve_hits(
            "q",
            retrieval_method="bm25",
            store=MagicMock(),
            embedder=MagicMock(),
            bm25_index=None,
            top_k=3,
        )


def test_retrieve_hits_weighted_hybrid_requires_fusion_weights():
    with pytest.raises(ValueError, match="fusion_weights are required"):
        retrieve_hits(
            "q",
            retrieval_method="weighted_hybrid",
            store=MagicMock(),
            embedder=MagicMock(),
            bm25_index=MagicMock(),
            top_k=3,
        )


def test_retrieve_hits_weighted_hybrid_passes_weights_to_hybrid():
    store = MagicMock()
    bm25 = MagicMock()
    embedder = MagicMock()
    with patch("app.pipeline.rag.hybrid_search_text") as hybrid:
        hybrid.return_value = [_hit(doc_id="w", chunk_id=0)]
        hits = retrieve_hits(
            "q",
            retrieval_method="weighted_hybrid",
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            top_k=4,
            fusion_weights=(0.8, 0.2),
        )
    assert hits[0].doc_id == "w"
    assert hybrid.call_args.kwargs["fusion_weights"] == (0.8, 0.2)
