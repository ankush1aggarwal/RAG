"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestIdMiddleware
from app.api.resources import AppResources, build_resources, close_resources
from app.api.routes import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: open clients; shutdown: release them."""
    injected = getattr(app.state, "_inject_resources", None)
    if injected is not None:
        app.state.resources = injected
    else:
        app.state.resources = build_resources(get_settings())
    try:
        yield
    finally:
        close_resources(app.state.resources)


def create_app(resources: AppResources | None = None) -> FastAPI:
    """Build the API app. Pass ``resources`` in tests to avoid real Qdrant/Ollama."""
    app = FastAPI(
        title="FiQA Production RAG",
        description=(
            "Production RAG over BEIR FiQA-2018. "
            "When `API_ENFORCE_FROZEN_RETRIEVAL=true` (default), `/query` accepts only the "
            "frozen E8 profile: dense retrieval, `top_k=10`, no rerank. "
            "See `docs/api_contract.md` for error codes and SSE streaming."
        ),
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness probes"},
            {"name": "query", "description": "RAG query (JSON and SSE)"},
        ],
    )
    app.state._inject_resources = resources
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()
