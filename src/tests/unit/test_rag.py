from unittest.mock import MagicMock

import pytest

from app.ingestion.schemas import SearchHit
from app.pipeline.rag import answer_query, citations_from_hits


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
