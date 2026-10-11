import pytest

from app.generation.prompt import (
    DEFAULT_SYSTEM_PROMPT,
    build_rag_prompt,
    format_citation_reference,
    format_context_block,
)
from app.ingestion.schemas import SearchHit


def _hit(
    doc_id: str = "42",
    chunk_id: int = 0,
    title: str = "APR basics",
    text: str = "APR is the annual cost of borrowing.",
) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        doc_title=title,
        chunk_id=chunk_id,
        chunk_text=text,
        score=0.9,
    )


def test_format_citation_reference():
    assert format_citation_reference(_hit()) == "[42:0]"


def test_format_context_block_empty():
    assert format_context_block([]) == "(No context excerpts were retrieved.)"


def test_format_context_block_includes_citation_reference_and_body():
    block = format_context_block([_hit(), _hit(doc_id="99", chunk_id=3, title="", text="Other.")])
    assert "[Source 1]\nCitation reference: [42:0]\nDocument title: APR basics" in block
    assert "APR is the annual cost of borrowing." in block
    assert "[Source 2]\nCitation reference: [99:3]\n\nOther." in block


def test_build_rag_prompt_includes_expected_citation_reference_for_apr_query():
    prompt = build_rag_prompt("what is APR", [_hit()])
    assert "Citation reference: [42:0]" in prompt.user
    assert "Document title: APR basics" in prompt.user
    assert "APR is the annual cost of borrowing." in prompt.user
    assert "Citation reference" in DEFAULT_SYSTEM_PROMPT


def test_build_rag_prompt_uses_defaults():
    prompt = build_rag_prompt("What is APR?", [_hit()])
    assert prompt.system == DEFAULT_SYSTEM_PROMPT
    assert "Question: What is APR?" in prompt.user
    assert "Context:" in prompt.user


def test_build_rag_prompt_with_no_hits():
    prompt = build_rag_prompt("What is APR?", [])
    assert "(No context excerpts were retrieved.)" in prompt.user


def test_build_rag_prompt_strips_question_whitespace():
    prompt = build_rag_prompt("  What is APR?  ", [_hit()])
    assert prompt.user.endswith("Question: What is APR?")


def test_build_rag_prompt_rejects_empty_question():
    with pytest.raises(ValueError, match="question must be non-empty"):
        build_rag_prompt("   ", [_hit()])
