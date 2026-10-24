"""FastAPI integration tests: middleware, handlers, routes, and lifecycle together."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.resources import AppResources
from app.config.settings import Settings
from app.ingestion.schemas import SearchHit


def _parse_sse_events(body: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = "message"
        data_line = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_line = line[5:].strip()
        if data_line:
            events.append((event_name, data_line))
    return events


def test_openapi_lists_health_ready_query_and_stream(integration_client: TestClient):
    schema = integration_client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/health" in paths
    assert "/ready" in paths
    assert "/query" in paths
    assert "/query/stream" in paths
    error_schema = schema["components"]["schemas"]["ErrorResponse"]
    assert "error" in error_schema["properties"]
    assert "request_id" in error_schema["properties"]


def test_http_stack_carries_request_id_on_success_path(integration_client: TestClient):
    with patch("app.api.resources.llm_ready", return_value=True):
        ready = integration_client.get("/ready", headers={"X-Request-ID": "trace-abc"})
    assert ready.headers["X-Request-ID"] == "trace-abc"

    query = integration_client.post(
        "/query",
        json={"question": "What is APR?", "top_k": 2},
        headers={"X-Request-ID": "trace-abc"},
    )
    assert query.status_code == 200
    assert query.headers["X-Request-ID"] == "trace-abc"
    assert query.json()["answer"] == "integration answer."


def test_readiness_503_detail_includes_operator_fields(integration_client: TestClient):
    integration_client.app.state.resources.store.collection_exists.return_value = False
    with patch("app.api.resources.llm_ready", return_value=False):
        response = integration_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "not_ready"
    detail = body["detail"]
    assert detail["frozen_retrieval_enforced"] is False
    assert "full_ingest_bundle_aligned" in detail
    assert detail["qdrant_collection"] is False
    assert detail["llm_ready"] is False


def test_query_stream_sse_token_then_done(integration_client: TestClient):
    with integration_client.stream(
        "POST",
        "/query/stream",
        json={"question": "fees?", "top_k": 2},
    ) as response:
        assert response.status_code == 200
        raw = response.read().decode()
    events = _parse_sse_events(raw)
    assert [name for name, _ in events] == ["token", "token", "done"]
    assert '"delta": "integration "' in events[0][1] or '"delta":"integration "' in events[0][1]
    assert "integration answer." in events[-1][1]


def test_stream_mid_pipeline_failure_emits_sse_error_event(integration_client: TestClient):
    with patch(
        "app.api.routes.prepare_rag_generation",
        side_effect=RuntimeError("retrieval blew up"),
    ):
        with integration_client.stream(
            "POST",
            "/query/stream",
            json={"question": "fees?", "top_k": 2},
        ) as response:
            raw = response.read().decode()
    assert "event: error" in raw or re.search(r"event:\s*error", raw)
    assert "internal_error" in raw


def test_frozen_profile_rejection_through_full_app_stack():
    store = MagicMock()
    store.collection_exists.return_value = True
    settings = Settings(_env_file=None, api_enforce_frozen_retrieval=True)
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(),
        llm=MagicMock(),
        bm25_index=None,
        ingest_manifest=MagicMock(),
        reranker=None,
    )
    app = create_app(resources=resources)
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"question": "APR?", "top_k": 3},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "frozen_profile_violation"
    assert body["request_id"]


def test_lifespan_shutdown_closes_resources():
    store = MagicMock()
    store.collection_exists.return_value = True
    resources = AppResources(
        settings=Settings(_env_file=None),
        store=store,
        embedder=MagicMock(),
        llm=MagicMock(),
        bm25_index=None,
        ingest_manifest=None,
        reranker=None,
    )
    with patch("app.api.main.close_resources") as close_resources:
        with TestClient(create_app(resources=resources)) as client:
            assert client.get("/health").status_code == 200
        close_resources.assert_called_once()
