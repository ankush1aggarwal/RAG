"""RAG orchestration: retrieve, prompt, generate."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.generation.llm import LLMProvider
from app.generation.prompt import build_rag_prompt
from app.ingestion.schemas import SearchHit
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.qdrant_store import QdrantChunkStore


class SourceCitation(BaseModel):
    """Traceable source reference aligned with API citation shape."""

    document_id: str
    chunk_id: int
    title: str = ""


class RagAnswer(BaseModel):
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)


def citations_from_hits(hits: list[SearchHit]) -> list[SourceCitation]:
    return [
        SourceCitation(
            document_id=hit.doc_id,
            chunk_id=hit.chunk_id,
            title=hit.doc_title,
        )
        for hit in hits
    ]


def answer_query(
    question: str,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    top_k: int = 3,
) -> RagAnswer:
    """Run dense retrieval, build a grounded prompt, and generate an answer."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")

    hits = store.search_text(question, embedder, limit=top_k)
    prompt = build_rag_prompt(question, hits)
    answer = llm.generate(prompt.user, system=prompt.system)

    return RagAnswer(
        answer=answer,
        citations=citations_from_hits(hits),
        hits=hits,
    )
