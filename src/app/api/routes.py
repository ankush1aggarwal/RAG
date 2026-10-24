"""HTTP route handlers."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.errors import (
    ERROR_INTERNAL,
    ERROR_INVALID_REQUEST,
    ERROR_NOT_READY,
    ERROR_RERANK_UNAVAILABLE,
    ERROR_TIMEOUT,
    api_error,
    is_upstream_timeout,
    map_value_error_to_api,
)
from app.config.production_retrieval import ResolvedQueryRetrieval, resolve_production_query
from app.api.resources import AppResources, is_ready_from_diagnostics, ready_diagnostics
from app.api.sse import format_sse
from app.cooperative_cancel import QueryCancelledError, QueryScope, invoke_in_scope
from app.api.streaming import iter_stream_queue, produce_llm_tokens
from app.api.thread_tasks import run_bounded_sync
from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RetrievalMeta,
)
from app.pipeline.rag import (
    RagAnswer,
    RagGenerationContext,
    answer_query,
    prepare_rag_generation,
)

_QUERY_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    422: {"model": ErrorResponse, "description": "Validation, frozen profile, or business rule error"},
    504: {"model": ErrorResponse, "description": "Query or upstream HTTP timeout"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
}

router = APIRouter()


def _resources(request: Request) -> AppResources:
    return request.app.state.resources


def _run_answer_query(
    resources: AppResources,
    question: str,
    resolved: ResolvedQueryRetrieval,
) -> RagAnswer:
    return answer_query(
        question,
        store=resources.store,
        embedder=resources.embedder,
        llm=resources.llm,
        top_k=resolved.top_k,
        retrieval_method=resolved.method,
        bm25_index=resources.bm25_index,
        use_rerank=resolved.use_rerank,
        reranker=resources.reranker,
        rerank_candidate_limit=resources.settings.rerank_candidate_limit,
    )


def _resolve_query_request(
    body: QueryRequest,
    resources: AppResources,
) -> tuple[ResolvedQueryRetrieval, float]:
    settings = resources.settings
    resolved = resolve_production_query(
        method=body.method,
        top_k=body.top_k,
        use_rerank=body.use_rerank,
        enforce_frozen=settings.api_enforce_frozen_retrieval,
        frozen_top_k=settings.retrieval_top_k,
    )
    if resolved.use_rerank and resources.reranker is None:
        api_error(
            422,
            error=ERROR_RERANK_UNAVAILABLE,
            message="Reranking requested but RERANKER_PROVIDER=none",
        )
    timeout = settings.query_timeout_seconds
    if timeout <= 0:
        api_error(
            422,
            error=ERROR_INVALID_REQUEST,
            message="query_timeout_seconds must be positive.",
        )
    return resolved, timeout


def _run_prepare_generation(
    resources: AppResources,
    question: str,
    resolved: ResolvedQueryRetrieval,
) -> RagGenerationContext:
    return prepare_rag_generation(
        question,
        store=resources.store,
        embedder=resources.embedder,
        top_k=resolved.top_k,
        retrieval_method=resolved.method,
        bm25_index=resources.bm25_index,
        use_rerank=resolved.use_rerank,
        reranker=resources.reranker,
        rerank_candidate_limit=resources.settings.rerank_candidate_limit,
    )


def _query_response(result: RagAnswer, resolved_top_k: int) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieval=RetrievalMeta(
            method=result.retrieval_method,
            top_k=resolved_top_k,
            reranker=result.reranker,
            candidates_retrieved=result.retrieval_candidate_count,
        ),
    )


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Liveness: process is up (no dependency checks)."""
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["health"],
    responses={
        503: {"model": ErrorResponse, "description": "Dependencies not ready (`detail` = diagnostics)"},
    },
)
def ready(request: Request) -> ReadyResponse:
    """
    Readiness for ``/query``.

    When the frozen profile is enforced, requires Qdrant + manifest/dense alignment + LLM.
    BM25 is reported but not required (hybrid/BM25 remain dev/CLI-only). With enforcement
    off, the full ingest bundle (including BM25) must align.
    """
    resources = _resources(request)
    diag = ready_diagnostics(resources)
    if not is_ready_from_diagnostics(diag):
        api_error(
            503,
            error=ERROR_NOT_READY,
            message="Service dependencies are not ready.",
            detail=diag,
        )
    return ReadyResponse(
        qdrant_collection=bool(diag["qdrant_collection"]),
        bm25_index=bool(diag["bm25_index"]),
        retrieval_index_aligned=bool(diag["retrieval_index_aligned"]),
        llm_provider=str(diag["llm_provider"]),
        llm_ready=bool(diag["llm_ready"]),
    )


def _map_query_exception(exc: BaseException, *, timeout: float) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, QueryCancelledError):
        api_error(
            504,
            error=ERROR_TIMEOUT,
            message=f"Query exceeded the {timeout:g}s wall-clock limit.",
        )
    if isinstance(exc, asyncio.TimeoutError):
        api_error(
            504,
            error=ERROR_TIMEOUT,
            message=f"Query exceeded the {timeout:g}s wall-clock limit.",
        )
    if isinstance(exc, ValueError):
        code, message = map_value_error_to_api(exc)
        api_error(422, error=code, message=message)
    if is_upstream_timeout(exc):
        api_error(
            504,
            error=ERROR_TIMEOUT,
            message="An upstream dependency timed out (LLM or HTTP client).",
        )
    api_error(
        500,
        error=ERROR_INTERNAL,
        message="An unexpected error occurred.",
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    responses=_QUERY_ERROR_RESPONSES,
)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """RAG: retrieve chunks, generate an answer, return citations."""
    resources = _resources(request)
    timeout = resources.settings.query_timeout_seconds
    resolved: ResolvedQueryRetrieval | None = None
    result: RagAnswer | None = None
    scope: QueryScope | None = None
    try:
        resolved, timeout = _resolve_query_request(body, resources)
        scope = QueryScope(timeout)
        result = await run_bounded_sync(
            scope,
            _run_answer_query,
            resources,
            body.question,
            resolved,
        )
    except BaseException as exc:
        if scope is not None:
            scope.cancel()
        _map_query_exception(exc, timeout=timeout)

    assert resolved is not None and result is not None
    return _query_response(result, resolved.top_k)


@router.post(
    "/query/stream",
    tags=["query"],
    responses={
        422: _QUERY_ERROR_RESPONSES[422],
        504: _QUERY_ERROR_RESPONSES[504],
        500: _QUERY_ERROR_RESPONSES[500],
    },
)
async def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    """RAG with Server-Sent Events: ``token`` deltas then a ``done`` payload."""
    resources = _resources(request)

    async def event_stream() -> AsyncIterator[str]:
        resolved: ResolvedQueryRetrieval | None = None
        timeout = resources.settings.query_timeout_seconds
        context: RagGenerationContext | None = None
        token_queue: queue.Queue | None = None
        producer_task: asyncio.Task[None] | None = None
        scope: QueryScope | None = None
        try:
            resolved, timeout = _resolve_query_request(body, resources)
            scope = QueryScope(timeout)
            context = await run_bounded_sync(
                scope,
                _run_prepare_generation,
                resources,
                body.question,
                resolved,
            )
            token_queue = queue.Queue()
            producer_task = asyncio.create_task(
                asyncio.to_thread(
                    invoke_in_scope,
                    scope,
                    produce_llm_tokens,
                    resources,
                    context,
                    token_queue,
                ),
            )
            parts: list[str] = []
            async for kind, payload in iter_stream_queue(
                token_queue,
                wall_clock_timeout=max(scope.remaining(), 0.001),
                scope=scope,
            ):
                if kind == "token" and isinstance(payload, str):
                    parts.append(payload)
                    yield format_sse("token", {"delta": payload})
            if producer_task is not None:
                await producer_task
        except BaseException as exc:
            if scope is not None:
                scope.cancel()
            if producer_task is not None and not producer_task.done():
                producer_task.cancel()
            try:
                _map_query_exception(exc, timeout=timeout)
            except HTTPException as http_exc:
                detail = http_exc.detail
                if isinstance(detail, dict):
                    yield format_sse("error", detail)
                else:
                    yield format_sse(
                        "error",
                        {"error": ERROR_INTERNAL, "message": str(detail), "detail": None},
                    )
            return

        assert resolved is not None and context is not None
        answer = "".join(parts)
        done = _query_response(
            RagAnswer(
                answer=answer,
                citations=context.citations,
                hits=context.hits,
                retrieval_method=context.retrieval_method,
                reranker=context.reranker,
                retrieval_candidate_count=context.retrieval_candidate_count,
            ),
            resolved.top_k,
        )
        yield format_sse("done", done.model_dump())

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
