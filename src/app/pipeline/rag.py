"""RAG orchestration: retrieve, prompt, generate."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

from app.generation.llm import LLMProvider, iter_llm_tokens
from app.generation.prompt import build_rag_prompt
from app.ingestion.schemas import SearchHit
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.hybrid import hybrid_search_text
from app.retrieval.qdrant_store import QdrantChunkStore
from app.retrieval.reranker import RerankerProvider, resolve_rerank_candidate_limit

RetrievalMethod = Literal["dense", "bm25", "hybrid", "weighted_hybrid"]


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
    reranker: str | None = None
    retrieval_candidate_count: int | None = None


class RagGenerationContext(BaseModel):
    """Retrieval + prompt ready for LLM (used by sync and streaming ``/query``)."""

    prompt_user: str
    prompt_system: str
    citations: list[SourceCitation] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    retrieval_method: RetrievalMethod = "dense"
    reranker: str | None = None
    retrieval_candidate_count: int | None = None


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
    fusion_weights: tuple[float, float] | None = None,
    candidate_limit: int | None = None,
) -> list[SearchHit]:
    retrieve_k = candidate_limit if candidate_limit is not None else top_k
    if retrieval_method == "dense":
        return store.search_text(question, embedder, limit=retrieve_k)
    if bm25_index is None:
        raise ValueError("BM25 index is not available; run ingest or check bm25_index_dir")
    if retrieval_method == "bm25":
        return bm25_index.search(question, limit=retrieve_k)
    if retrieval_method in ("hybrid", "weighted_hybrid"):
        weights = fusion_weights if retrieval_method == "weighted_hybrid" else None
        if retrieval_method == "weighted_hybrid" and fusion_weights is None:
            raise ValueError("fusion_weights are required for weighted_hybrid retrieval")
        return hybrid_search_text(
            question,
            store=store,
            bm25_index=bm25_index,
            embedder=embedder,
            limit=retrieve_k,
            fusion_weights=weights,
        )
    raise ValueError(f"Unsupported retrieval method: {retrieval_method}")


def retrieve_with_optional_rerank(
    question: str,
    *,
    retrieval_method: RetrievalMethod,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int,
    use_rerank: bool = False,
    reranker: RerankerProvider | None = None,
    rerank_candidate_limit: int | None = None,
    fusion_weights: tuple[float, float] | None = None,
) -> tuple[list[SearchHit], str | None, int | None]:
    candidate_limit = None
    if use_rerank:
        if reranker is None:
            raise ValueError("reranker is required when use_rerank is True")
        candidate_limit = resolve_rerank_candidate_limit(
            top_k,
            candidate_limit=rerank_candidate_limit,
        )

    hits = retrieve_hits(
        question,
        retrieval_method=retrieval_method,
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=top_k,
        fusion_weights=fusion_weights,
        candidate_limit=candidate_limit,
    )
    candidate_count: int | None = None
    reranker_name: str | None = None
    if use_rerank and reranker is not None:
        candidate_count = len(hits)
        hits = reranker.rerank(question, hits, top_k=top_k)
        reranker_name = reranker.name
    return hits, reranker_name, candidate_count


def prepare_rag_generation(
    question: str,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    top_k: int = 10,
    retrieval_method: RetrievalMethod = "dense",
    bm25_index: Bm25ChunkIndex | None = None,
    use_rerank: bool = False,
    reranker: RerankerProvider | None = None,
    rerank_candidate_limit: int | None = None,
) -> RagGenerationContext:
    """Retrieve context and build the grounded prompt (no LLM call)."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")

    hits, reranker_name, candidate_count = retrieve_with_optional_rerank(
        question,
        retrieval_method=retrieval_method,
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=top_k,
        use_rerank=use_rerank,
        reranker=reranker,
        rerank_candidate_limit=rerank_candidate_limit,
    )

    prompt = build_rag_prompt(question, hits)
    return RagGenerationContext(
        prompt_user=prompt.user,
        prompt_system=prompt.system,
        citations=citations_from_hits(hits),
        hits=hits,
        retrieval_method=retrieval_method,
        reranker=reranker_name,
        retrieval_candidate_count=candidate_count if use_rerank else None,
    )


def stream_answer_tokens(
    context: RagGenerationContext,
    llm: LLMProvider,
) -> Iterator[str]:
    """Yield answer text deltas from the LLM for a prepared RAG context."""
    yield from iter_llm_tokens(
        llm,
        context.prompt_user,
        system=context.prompt_system,
    )


def answer_query(
    question: str,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    top_k: int = 10,
    retrieval_method: RetrievalMethod = "dense",
    bm25_index: Bm25ChunkIndex | None = None,
    use_rerank: bool = False,
    reranker: RerankerProvider | None = None,
    rerank_candidate_limit: int | None = None,
) -> RagAnswer:
    """Retrieve context, build a grounded prompt, and generate an answer."""
    context = prepare_rag_generation(
        question,
        store=store,
        embedder=embedder,
        top_k=top_k,
        retrieval_method=retrieval_method,
        bm25_index=bm25_index,
        use_rerank=use_rerank,
        reranker=reranker,
        rerank_candidate_limit=rerank_candidate_limit,
    )
    answer = llm.generate(context.prompt_user, system=context.prompt_system)

    return RagAnswer(
        answer=answer,
        citations=context.citations,
        hits=context.hits,
        retrieval_method=context.retrieval_method,
        reranker=context.reranker,
        retrieval_candidate_count=context.retrieval_candidate_count,
    )
