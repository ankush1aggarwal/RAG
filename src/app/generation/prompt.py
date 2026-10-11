"""Prompt construction for grounded RAG generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.ingestion.schemas import SearchHit

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant for financial information questions.
Answer using only the provided context excerpts.
If the context does not contain enough information, say that you cannot answer from the \
provided sources. Do not invent facts.
When you use information from a source, cite it inline using [doc_id:chunk_id] matching \
the source labels in the context.\
"""


@dataclass(frozen=True)
class RagPrompt:
    """Messages ready for LLMProvider.generate(prompt=user, system=system)."""

    system: str
    user: str


def format_context_block(hits: Sequence[SearchHit]) -> str:
    """Format retrieved chunks as labeled excerpts for the user message."""
    if not hits:
        return "(No context excerpts were retrieved.)"

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        title_line = f"title: {hit.doc_title}\n" if hit.doc_title else ""
        blocks.append(
            f"[Source {index} | doc_id:{hit.doc_id} | chunk_id:{hit.chunk_id}]\n"
            f"{title_line}"
            f"{hit.chunk_text.strip()}"
        )
    return "\n\n".join(blocks)


def build_rag_prompt(
    question: str,
    hits: Sequence[SearchHit]
) -> RagPrompt:
    """Build system and user prompts from a question and retrieval hits."""
    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")

    context = format_context_block(hits)
    user = (
        "Use the context excerpts below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    return RagPrompt(system=DEFAULT_SYSTEM_PROMPT, user=user)
