"""Long-lived dependencies (Qdrant, embedder, LLM) for the API process."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from app.config.settings import Settings
from app.generation.llm import LLMMethod, LLMProvider, OllamaLLM, create_llm_provider
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import EmbeddingProvider


@dataclass
class AppResources:
    settings: Settings
    store: QdrantChunkStore
    embedder: EmbeddingProvider
    llm: LLMProvider
    bm25_index: Bm25ChunkIndex | None = None


def build_resources(settings: Settings) -> AppResources:
    embedder = create_embedding_provider(
        settings.embedding_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )
    store = QdrantChunkStore.from_path(
        settings.resolved_qdrant_path(),
        settings.collection_name,
    )
    if settings.llm_provider not in ("ollama", "cohere"):
        raise ValueError(f"Unsupported llm_provider for API: {settings.llm_provider}")
    method: LLMMethod = settings.llm_provider  # type: ignore[assignment]
    llm = create_llm_provider(
        method,
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        cohere_api_key=settings.cohere_api_key,
        cohere_model=settings.cohere_chat_model,
    )
    bm25_path = settings.resolved_bm25_index_path()
    bm25_index = Bm25ChunkIndex.load(bm25_path) if bm25_path.is_file() else None
    return AppResources(
        settings=settings,
        store=store,
        embedder=embedder,
        llm=llm,
        bm25_index=bm25_index,
    )


def close_resources(resources: AppResources) -> None:
    resources.store.close()
    llm = resources.llm
    if isinstance(llm, OllamaLLM):
        llm.close()


def ollama_reachable(base_url: str, timeout_seconds: float = 3.0) -> bool:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def llm_ready(settings: Settings) -> bool:
    if settings.llm_provider == "ollama":
        return ollama_reachable(settings.ollama_base_url)
    if settings.llm_provider == "cohere":
        return bool(settings.cohere_api_key or os.getenv("COHERE_API_KEY"))
    return False


def bm25_ready(resources: AppResources) -> bool:
    """True when the on-disk BM25 index was loaded at startup (required for hybrid/bm25)."""
    return resources.bm25_index is not None
