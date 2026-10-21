"""HTTP route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.config.production_retrieval import resolve_production_query
from app.api.resources import AppResources, api_ready, ready_diagnostics
from app.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RetrievalMeta,
)
from app.pipeline.rag import answer_query

router = APIRouter()


def _resources(request: Request) -> AppResources:
    return request.app.state.resources


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: process is up (no dependency checks)."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """
    Readiness for ``/query``.

    When the frozen profile is enforced, requires Qdrant + manifest/dense alignment + LLM.
    BM25 is reported but not required (hybrid/BM25 remain dev/CLI-only). With enforcement
    off, the full ingest bundle (including BM25) must align.
    """
    resources = _resources(request)
    diag = ready_diagnostics(resources)
    if not api_ready(resources):
        raise HTTPException(status_code=503, detail=diag)
    return ReadyResponse(
        qdrant_collection=bool(diag["qdrant_collection"]),
        bm25_index=bool(diag["bm25_index"]),
        retrieval_index_aligned=bool(diag["retrieval_index_aligned"]),
        llm_provider=str(diag["llm_provider"]),
        llm_ready=bool(diag["llm_ready"]),
    )


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """RAG: retrieve chunks, generate an answer, return citations."""
    resources = _resources(request)
    settings = resources.settings
    try:
        resolved = resolve_production_query(
            method=body.method,
            top_k=body.top_k,
            use_rerank=body.use_rerank,
            enforce_frozen=settings.api_enforce_frozen_retrieval,
            frozen_top_k=settings.retrieval_top_k,
        )
        if resolved.use_rerank and resources.reranker is None:
            raise HTTPException(
                status_code=422,
                detail="Reranking requested but RERANKER_PROVIDER=none",
            )
        result = answer_query(
            body.question,
            store=resources.store,
            embedder=resources.embedder,
            llm=resources.llm,
            top_k=resolved.top_k,
            retrieval_method=resolved.method,
            bm25_index=resources.bm25_index,
            use_rerank=resolved.use_rerank,
            reranker=resources.reranker,
            rerank_candidate_limit=settings.rerank_candidate_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieval=RetrievalMeta(
            method=result.retrieval_method,
            top_k=resolved.top_k,
            reranker=result.reranker,
            candidates_retrieved=result.retrieval_candidate_count,
        ),
    )
