"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state._inject_resources = resources
    app.include_router(router)
    return app


app = create_app()
