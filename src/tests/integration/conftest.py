"""Shared fixtures for HTTP-level API integration tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.resources import AppResources
from app.config.production_retrieval import FROZEN_COLLECTION_NAME, FROZEN_EMBEDDING_MODEL
from app.config.settings import Settings
from app.ingestion.ingest_manifest import IngestManifest
from app.ingestion.schemas import SearchHit


class FakeStreamLLM:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "integration answer."

    def stream_generate(self, prompt: str, *, system: str | None = None, cancel=None):
        yield "integration "
        yield "answer."


@pytest.fixture
def integration_client() -> TestClient:
    store = MagicMock()
    store.collection_exists.return_value = True
    store.point_count.return_value = 1
    store.vector_size.return_value = 768
    store.search_text.return_value = [
        SearchHit(
            doc_id="doc-1",
            doc_title="Title",
            chunk_id=0,
            chunk_text="context",
            score=0.9,
        )
    ]
    settings = Settings(_env_file=None, api_enforce_frozen_retrieval=False)
    manifest = IngestManifest(
        ingest_generation_id="gen-integration",
        corpus_limit=None,
        chunk_size=200,
        overlap_fraction=0.2,
        embedding_method="sentence_transformer",
        embedding_model=FROZEN_EMBEDDING_MODEL,
        embedding_dimension=768,
        collection_name=FROZEN_COLLECTION_NAME,
        num_documents=57000,
        num_chunks=1,
    )
    bm25_index = MagicMock()
    bm25_index.num_chunks = 1
    bm25_index.ingest_generation_id = "gen-integration"
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(dimension=768),
        llm=FakeStreamLLM(),
        bm25_index=bm25_index,
        ingest_manifest=manifest,
        reranker=None,
    )
    app = create_app(resources=resources)
    with TestClient(app) as client:
        yield client
