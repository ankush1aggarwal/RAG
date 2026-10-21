"""Long-lived dependencies (Qdrant, embedder, LLM) for the API process."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from app.config.production_retrieval import validate_frozen_retrieval_settings
from app.config.settings import Settings
from app.generation.llm import LLMMethod, LLMProvider, OllamaLLM, create_llm_provider
from app.ingestion.ingest_manifest import (
    IndexAlignmentError,
    IngestManifest,
    load_manifest,
    manifest_path_for_collection,
    validate_dense_retrieval_index_alignment,
    validate_manifest_runtime_config,
    validate_retrieval_index_alignment,
)
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import (
    EmbeddingProvider,
    embedding_dimension,
    embedding_model_for_method,
)
from app.retrieval.reranker import RerankerProvider, create_reranker_provider


@dataclass
class AppResources:
    settings: Settings
    store: QdrantChunkStore
    embedder: EmbeddingProvider
    llm: LLMProvider
    bm25_index: Bm25ChunkIndex | None = None
    ingest_manifest: IngestManifest | None = None
    reranker: RerankerProvider | None = None


def build_resources(settings: Settings) -> AppResources:
    validate_frozen_retrieval_settings(settings)
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
    manifest_path = manifest_path_for_collection(bm25_path)
    bm25_index = Bm25ChunkIndex.load(bm25_path) if bm25_path.is_file() else None
    ingest_manifest = load_manifest(manifest_path)
    if settings.api_enforce_frozen_retrieval:
        reranker = None
    else:
        reranker = create_reranker_provider(
            settings.reranker_provider,  # type: ignore[arg-type]
            cross_encoder_model=settings.cross_encoder_rerank_model,
            cohere_api_key=settings.cohere_api_key,
            cohere_rerank_model=settings.cohere_rerank_model,
        )
    return AppResources(
        settings=settings,
        store=store,
        embedder=embedder,
        llm=llm,
        bm25_index=bm25_index,
        ingest_manifest=ingest_manifest,
        reranker=reranker,
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


def _validate_manifest_runtime(resources: AppResources) -> None:
    settings = resources.settings
    if resources.ingest_manifest is None:
        raise IndexAlignmentError("Ingest manifest is not loaded.")
    method = settings.embedding_method
    if method not in ("sentence_transformer", "cohere"):
        raise IndexAlignmentError(f"Unsupported embedding_method for API: {method!r}")
    validate_manifest_runtime_config(
        resources.ingest_manifest,
        corpus_limit=settings.resolved_fiqa_corpus_limit(),
        chunk_size=settings.chunk_size,
        overlap_fraction=settings.chunk_overlap_fraction,
        embedding_method=method,
        embedding_model=embedding_model_for_method(
            method,  # type: ignore[arg-type]
            sentence_transformer_model=settings.embedding_model,
            cohere_embed_model=settings.cohere_embed_model,
        ),
        embedding_dimension=embedding_dimension(resources.embedder),
        collection_name=settings.collection_name,
    )


def dense_retrieval_index_aligned(resources: AppResources) -> bool:
    """Manifest + Qdrant + embedder settings (dense ``/query`` path; BM25 optional)."""
    if resources.ingest_manifest is None:
        return False
    if not resources.store.collection_exists():
        return False
    try:
        _validate_manifest_runtime(resources)
        validate_dense_retrieval_index_alignment(
            resources.ingest_manifest,
            collection_name=resources.settings.collection_name,
            qdrant_point_count=resources.store.point_count(),
            qdrant_vector_size=resources.store.vector_size(),
        )
    except IndexAlignmentError:
        return False
    return True


def retrieval_index_aligned(resources: AppResources) -> bool:
    """True when manifest, Qdrant, BM25, and embedder settings all align."""
    if resources.bm25_index is None:
        return False
    if not dense_retrieval_index_aligned(resources):
        return False
    try:
        validate_retrieval_index_alignment(
            resources.ingest_manifest,
            collection_name=resources.settings.collection_name,
            qdrant_point_count=resources.store.point_count(),
            bm25_chunk_count=resources.bm25_index.num_chunks,
            qdrant_vector_size=resources.store.vector_size(),
            bm25_ingest_generation_id=resources.bm25_index.ingest_generation_id,
        )
    except IndexAlignmentError:
        return False
    return True


def bm25_ready(resources: AppResources) -> bool:
    """True when lexical index is loaded and aligned with Qdrant + manifest."""
    return retrieval_index_aligned(resources)


def retrieval_ready_for_api(resources: AppResources) -> bool:
    """Readiness gate for serving ``/query`` (dense-only when frozen profile enforced)."""
    if resources.settings.api_enforce_frozen_retrieval:
        return dense_retrieval_index_aligned(resources)
    return retrieval_index_aligned(resources)


def ready_diagnostics(resources: AppResources) -> dict[str, object]:
    """Fields for ``/ready`` JSON (200 or 503 detail)."""
    settings = resources.settings
    qdrant_ok = resources.store.collection_exists()
    bm25_loaded = resources.bm25_index is not None
    index_aligned = retrieval_ready_for_api(resources)
    full_bundle_aligned = retrieval_index_aligned(resources) if bm25_loaded else False
    return {
        "status": "ready" if (qdrant_ok and index_aligned and llm_ready(settings)) else "not_ready",
        "qdrant_collection": qdrant_ok,
        "bm25_index": bm25_ready(resources) if bm25_loaded else False,
        "retrieval_index_aligned": index_aligned,
        "full_ingest_bundle_aligned": full_bundle_aligned,
        "frozen_retrieval_enforced": settings.api_enforce_frozen_retrieval,
        "llm_provider": settings.llm_provider,
        "llm_ready": llm_ready(settings),
    }


def api_ready(resources: AppResources) -> bool:
    diag = ready_diagnostics(resources)
    return (
        bool(diag["qdrant_collection"])
        and bool(diag["retrieval_index_aligned"])
        and bool(diag["llm_ready"])
    )
