from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.resources import AppResources
from app.config.settings import Settings
from app.ingestion.ingest_manifest import IngestManifest
from app.ingestion.schemas import SearchHit


class _FakeLLM:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "API answer."


@pytest.fixture
def client() -> TestClient:
    store = MagicMock()
    store.collection_exists.return_value = True
    store.point_count.return_value = 1
    store.vector_size.return_value = 384
    store.search_text.return_value = [
        SearchHit(
            doc_id="1",
            doc_title="T",
            chunk_id=0,
            chunk_text="ctx",
            score=0.5,
        )
    ]
    settings = Settings(_env_file=None)
    manifest = IngestManifest(
        corpus_limit=1000,
        chunk_size=100,
        overlap_fraction=0.2,
        embedding_method="sentence_transformer",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        collection_name="fiqa_sample",
        num_documents=1000,
        num_chunks=1,
    )
    bm25_index = MagicMock()
    bm25_index.num_chunks = 1
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(),
        llm=_FakeLLM(),
        bm25_index=bm25_index,
        ingest_manifest=manifest,
    )
    app = create_app(resources=resources)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_success(client: TestClient):
    with patch("app.api.routes.llm_ready", return_value=True):
        response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["qdrant_collection"] is True
    assert body["bm25_index"] is True
    assert body["retrieval_index_aligned"] is True
    assert body["llm_provider"] == "ollama"
    assert body["llm_ready"] is True


def test_ready_returns_503_when_dependencies_down(client: TestClient):
    client.app.state.resources.store.collection_exists.return_value = False
    with patch("app.api.routes.llm_ready", return_value=False):
        response = client.get("/ready")
    assert response.status_code == 503


def test_ready_returns_503_when_bm25_missing(client: TestClient):
    client.app.state.resources.bm25_index = None
    client.app.state.resources.ingest_manifest = None
    with patch("app.api.routes.llm_ready", return_value=True):
        response = client.get("/ready")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["bm25_index"] is False
    assert detail["retrieval_index_aligned"] is False
    assert detail["qdrant_collection"] is True


def test_ready_returns_503_when_manifest_misaligned(client: TestClient):
    client.app.state.resources.store.point_count.return_value = 99
    with patch("app.api.routes.llm_ready", return_value=True):
        response = client.get("/ready")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["retrieval_index_aligned"] is False


def test_query_success(client: TestClient):
    response = client.post("/query", json={"question": "What is APR?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "API answer."
    assert body["citations"][0]["document_id"] == "1"
    assert body["retrieval"] == {"method": "dense", "top_k": 2}


def test_query_validation_empty_question(client: TestClient):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_hybrid_uses_bm25_and_reports_method(client: TestClient):
    bm25 = MagicMock()
    bm25.search.return_value = [
        SearchHit(
            doc_id="2",
            doc_title="Lex",
            chunk_id=1,
            chunk_text="bm25 ctx",
            score=1.0,
        )
    ]
    client.app.state.resources.bm25_index = bm25

    with patch("app.pipeline.rag.hybrid_search_text") as hybrid:
        hybrid.return_value = [
            SearchHit(
                doc_id="2",
                doc_title="Lex",
                chunk_id=1,
                chunk_text="bm25 ctx",
                score=0.03,
            )
        ]
        response = client.post(
            "/query",
            json={"question": "fees", "top_k": 2, "method": "hybrid"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval"] == {"method": "hybrid", "top_k": 2}
    hybrid.assert_called_once()


def test_query_bm25_returns_422_when_index_missing(client: TestClient):
    client.app.state.resources.bm25_index = None
    response = client.post(
        "/query",
        json={"question": "fees", "method": "bm25"},
    )
    assert response.status_code == 422
    assert "BM25 index is not available" in response.json()["detail"]
