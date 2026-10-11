from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.resources import AppResources
from app.config.settings import Settings
from app.ingestion.schemas import SearchHit
class _FakeLLM:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "API answer."


@pytest.fixture
def client() -> TestClient:
    store = MagicMock()
    store.collection_exists.return_value = True
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
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(),
        llm=_FakeLLM(),
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
    assert body["llm_provider"] == "ollama"
    assert body["llm_ready"] is True


def test_ready_returns_503_when_dependencies_down(client: TestClient):
    client.app.state.resources.store.collection_exists.return_value = False
    with patch("app.api.routes.llm_ready", return_value=False):
        response = client.get("/ready")
    assert response.status_code == 503


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
