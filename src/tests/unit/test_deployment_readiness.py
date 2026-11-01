"""Local vs remote Qdrant wiring and frozen /ready gates (Cloud Run deployment)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.api.resources import (
    AppResources,
    dense_retrieval_index_aligned,
    ready_diagnostics,
    retrieval_ready_for_api,
)
from app.config.production_retrieval import FROZEN_COLLECTION_NAME, FROZEN_EMBEDDING_MODEL
from app.config.settings import Settings
from app.ingestion.ingest_manifest import load_manifest, manifest_path_for_collection
from app.retrieval.qdrant_connection import open_qdrant_chunk_store
from app.retrieval.qdrant_store import QdrantChunkStore


def _packaged_manifest():
    settings = Settings(
        _env_file=None,
        bm25_index_dir="deploy/index_metadata",
        collection_name=FROZEN_COLLECTION_NAME,
    )
    path = manifest_path_for_collection(settings.resolved_bm25_index_path())
    manifest = load_manifest(path)
    assert manifest is not None
    return manifest, settings


def test_local_qdrant_in_memory_store_still_searches():
    store = QdrantChunkStore.from_path(None, "local_memory_smoke")
    store.prepare_collection(3, recreate=True)
    assert store.collection_exists()
    store.close()


@patch("app.retrieval.qdrant_connection.QdrantChunkStore.from_connection")
def test_open_qdrant_chunk_store_remote_when_url_set(mock_from_connection: MagicMock):
    settings = Settings(
        _env_file=None,
        qdrant_url="https://cluster.cloud.qdrant.io:6333",
        qdrant_api_key="secret",
        qdrant_path="data/qdrant_local",
    )
    open_qdrant_chunk_store(settings, FROZEN_COLLECTION_NAME)
    mock_from_connection.assert_called_once_with(
        collection_name=FROZEN_COLLECTION_NAME,
        url="https://cluster.cloud.qdrant.io:6333",
        api_key="secret",
    )


@patch("app.retrieval.qdrant_connection.QdrantChunkStore.from_connection")
def test_open_qdrant_chunk_store_local_path_when_no_url(mock_from_connection: MagicMock):
    settings = Settings(_env_file=None, qdrant_url=None, qdrant_path="data/qdrant_local")
    open_qdrant_chunk_store(settings, "fiqa_sample", path_override="data/qdrant_local")
    kwargs = mock_from_connection.call_args.kwargs
    assert kwargs["collection_name"] == "fiqa_sample"
    assert kwargs.get("path") is not None


def test_frozen_ready_with_packaged_manifest_and_aligned_qdrant():
    manifest, settings = _packaged_manifest()
    store = MagicMock()
    store.collection_exists.return_value = True
    store.point_count.return_value = manifest.num_chunks
    store.vector_size.return_value = manifest.embedding_dimension
    embedder = MagicMock()
    embedder.dimension = manifest.embedding_dimension
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=embedder,
        llm=MagicMock(),
        bm25_index=None,
        ingest_manifest=manifest,
        reranker=None,
    )
    assert dense_retrieval_index_aligned(resources) is True
    assert retrieval_ready_for_api(resources) is True
    diag = ready_diagnostics(resources)
    assert diag["retrieval_index_aligned"] is True
    assert diag["bm25_index"] is False
    assert diag["frozen_retrieval_enforced"] is True


def test_frozen_not_ready_when_qdrant_point_count_differs_from_manifest():
    manifest, settings = _packaged_manifest()
    store = MagicMock()
    store.collection_exists.return_value = True
    store.point_count.return_value = manifest.num_chunks - 1
    store.vector_size.return_value = manifest.embedding_dimension
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(dimension=manifest.embedding_dimension),
        llm=MagicMock(),
        bm25_index=None,
        ingest_manifest=manifest,
        reranker=None,
    )
    assert dense_retrieval_index_aligned(resources) is False
    assert retrieval_ready_for_api(resources) is False
