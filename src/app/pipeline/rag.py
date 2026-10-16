"""RAG orchestration: retrieve, prompt, generate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.generation.llm import LLMProvider
from app.generation.prompt import build_rag_prompt
from app.ingestion.schemas import SearchHit
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.hybrid import hybrid_search_text
from app.retrieval.qdrant_store import QdrantChunkStore

RetrievalMethod = Literal["dense", "bm25", "hybrid"]


class SourceCitation(BaseModel):
    """Traceable source reference aligned with API citation shape."""

    document_id: str
    chunk_id: int
    title: str = ""


class RagAnswer(BaseModel):
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    retrieval_method: RetrievalMethod = "dense"


def citations_from_hits(hits: list[SearchHit]) -> list[SourceCitation]:
    return [
        SourceCitation(
            document_id=hit.doc_id,
            chunk_id=hit.chunk_id,
            title=hit.doc_title,
        )
        for hit in hits
    ]


def retrieve_hits(
    question: str,
    *,
    retrieval_method: RetrievalMethod,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int,
) -> list[SearchHit]:
    if retrieval_method == "dense":
        return store.search_text(question, embedder, limit=top_k)
    if bm25_index is None:
        raise ValueError("BM25 index is not available; run ingest or check bm25_index_dir")
    if retrieval_method == "bm25":
        return bm25_index.search(question, limit=top_k)
    if retrieval_method == "hybrid":
        return hybrid_search_text(
            question,
            store=store,
            bm25_index=bm25_index,
            embedder=embedder,
            limit=top_k,
        )
    raise ValueError(f"Unsupported retrieval method: {retrieval_method}")


def answer_query(
    question: str,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    top_k: int = 3,
    retrieval_method: RetrievalMethod = "dense",
    bm25_index: Bm25ChunkIndex | None = None,
) -> RagAnswer:
    """Retrieve context, build a grounded prompt, and generate an answer."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")

    hits = retrieve_hits(
        question,
        retrieval_method=retrieval_method,
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=top_k,
    )
    prompt = build_rag_prompt(question, hits)
    answer = llm.generate(prompt.user, system=prompt.system)

    return RagAnswer(
        answer=answer,
        citations=citations_from_hits(hits),
        hits=hits,
        retrieval_method=retrieval_method,
    )
