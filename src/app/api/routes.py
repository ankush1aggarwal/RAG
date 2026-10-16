"""HTTP route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.resources import AppResources, bm25_ready, llm_ready
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
    """Readiness: Qdrant, BM25 (hybrid), and configured LLM backend are usable."""
    resources = _resources(request)
    settings = resources.settings
    qdrant_ok = resources.store.collection_exists()
    bm25_ok = bm25_ready(resources)
    llm_ok = llm_ready(settings)
    if not (qdrant_ok and bm25_ok and llm_ok):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "qdrant_collection": qdrant_ok,
                "bm25_index": bm25_ok,
                "llm_provider": settings.llm_provider,
                "llm_ready": llm_ok,
            },
        )
    return ReadyResponse(
        qdrant_collection=qdrant_ok,
        bm25_index=bm25_ok,
        llm_provider=settings.llm_provider,
        llm_ready=llm_ok,
    )


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """RAG: retrieve chunks, generate an answer, return citations."""
    resources = _resources(request)
    try:
        result = answer_query(
            body.question,
            store=resources.store,
            embedder=resources.embedder,
            llm=resources.llm,
            top_k=body.top_k,
            retrieval_method=body.method,
            bm25_index=resources.bm25_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieval=RetrievalMeta(method=result.retrieval_method, top_k=body.top_k),
    )
