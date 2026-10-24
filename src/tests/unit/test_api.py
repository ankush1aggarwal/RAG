import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.resources import AppResources
from app.config.production_retrieval import FROZEN_COLLECTION_NAME, FROZEN_EMBEDDING_MODEL
from app.config.settings import Settings
from app.ingestion.ingest_manifest import IngestManifest
from app.ingestion.schemas import SearchHit


class _FakeLLM:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "API answer."

    def stream_generate(self, prompt: str, *, system: str | None = None, cancel=None):
        yield "API "
        yield "answer."


@pytest.fixture
def client() -> TestClient:
    store = MagicMock()
    store.collection_exists.return_value = True
    store.point_count.return_value = 1
    store.vector_size.return_value = 768
    store.search_text.return_value = [
        SearchHit(
            doc_id="1",
            doc_title="T",
            chunk_id=0,
            chunk_text="ctx",
            score=0.5,
        )
    ]
    settings = Settings(_env_file=None, api_enforce_frozen_retrieval=False)
    manifest = IngestManifest(
        ingest_generation_id="gen-api-test",
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
    bm25_index.ingest_generation_id = "gen-api-test"
    embedder = MagicMock()
    embedder.dimension = 768
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=embedder,
        llm=_FakeLLM(),
        bm25_index=bm25_index,
        ingest_manifest=manifest,
        reranker=None,
    )
    app = create_app(resources=resources)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_id_echoed_from_client_header(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "corr-week6-test"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "corr-week6-test"


def test_request_id_generated_when_header_missing(client: TestClient):
    response = client.get("/health")
    rid = response.headers.get("X-Request-ID")
    assert rid
    assert len(rid) >= 32


def test_validation_error_includes_request_id(client: TestClient):
    response = client.post(
        "/query",
        json={"question": ""},
        headers={"X-Request-ID": "bad-body-id"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["request_id"] == "bad-body-id"
    assert body["error"] == "validation_error"


def test_ready_success(client: TestClient):
    with patch("app.api.resources.llm_ready", return_value=True):
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
    with patch("app.api.resources.llm_ready", return_value=False):
        response = client.get("/ready")
    assert response.status_code == 503


def test_ready_returns_503_when_bm25_missing_and_hybrid_dev_mode(client: TestClient):
    client.app.state.resources.bm25_index = None
    client.app.state.resources.ingest_manifest = None
    with patch("app.api.resources.llm_ready", return_value=True):
        response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "not_ready"
    diag = body["detail"]
    assert diag["bm25_index"] is False
    assert diag["retrieval_index_aligned"] is False
    assert diag["qdrant_collection"] is True


def test_ready_succeeds_without_bm25_when_frozen_profile_enforced():
    store = MagicMock()
    store.collection_exists.return_value = True
    store.point_count.return_value = 1
    store.vector_size.return_value = 768
    settings = Settings(_env_file=None, api_enforce_frozen_retrieval=True)
    manifest = IngestManifest(
        ingest_generation_id="gen-api-test",
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
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(dimension=768),
        llm=_FakeLLM(),
        bm25_index=None,
        ingest_manifest=manifest,
        reranker=None,
    )
    app = create_app(resources=resources)
    with patch("app.api.resources.llm_ready", return_value=True):
        with TestClient(app) as test_client:
            response = test_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_index_aligned"] is True
    assert body["bm25_index"] is False


def test_ready_returns_503_when_manifest_misaligned(client: TestClient):
    client.app.state.resources.store.point_count.return_value = 99
    with patch("app.api.resources.llm_ready", return_value=True):
        response = client.get("/ready")
    assert response.status_code == 503
    diag = response.json()["detail"]
    assert diag["retrieval_index_aligned"] is False


def test_query_success(client: TestClient):
    response = client.post("/query", json={"question": "What is APR?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "API answer."
    assert body["citations"][0]["document_id"] == "1"


def test_query_stream_emits_sse_token_and_done(client: TestClient):
    with client.stream(
        "POST",
        "/query/stream",
        json={"question": "What is APR?", "top_k": 2},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        body = response.read().decode()
    assert "event: token" in body
    assert '"delta": "API "' in body or '"delta":"API "' in body
    assert "event: done" in body
    assert '"answer": "API answer."' in body or '"answer":"API answer."' in body


def test_query_async_cancellation_invokes_cleanup_and_reraises(client: TestClient):
    from app.api.routes import query
    from app.api.schemas import QueryRequest

    body = QueryRequest(question="fees?", top_k=2)
    request = MagicMock()
    request.app.state.resources = client.app.state.resources

    async def _run() -> None:
        with patch(
            "app.api.routes.run_bounded_sync",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with patch(
                "app.api.routes._cleanup_query_cancellation",
                new_callable=AsyncMock,
            ) as cleanup:
                with pytest.raises(asyncio.CancelledError):
                    await query(body, request)
                cleanup.assert_awaited_once()

    asyncio.run(_run())


def test_query_genuine_failure_still_maps_to_http_error(client: TestClient):
    with patch("app.api.routes.run_bounded_sync", new=AsyncMock(side_effect=RuntimeError("boom"))):
        response = client.post("/query", json={"question": "fees?", "top_k": 2})
    assert response.status_code == 500
    assert response.json()["error"] == "internal_error"


def test_query_stream_validation_error_is_json_not_sse(client: TestClient):
    response = client.post("/query/stream", json={"question": ""})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_query_validation_empty_question(client: TestClient):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["message"]


def test_query_internal_error_returns_structured_500(client: TestClient):
    with patch("app.api.routes.answer_query", side_effect=RuntimeError("boom")):
        response = client.post("/query", json={"question": "fees?", "top_k": 2})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert body["detail"] is None


def test_query_wall_clock_timeout_returns_504(client: TestClient):
    def slow_answer(*_args, **_kwargs):
        time.sleep(0.05)
        return MagicMock(
            answer="late",
            citations=[],
            retrieval_method="dense",
            reranker=None,
            retrieval_candidate_count=None,
        )

    settings = client.app.state.resources.settings.model_copy(
        update={"query_timeout_seconds": 0.01},
    )
    client.app.state.resources.settings = settings
    with patch("app.api.routes.answer_query", side_effect=slow_answer):
        response = client.post("/query", json={"question": "fees?", "top_k": 2})
    assert response.status_code == 504
    assert response.json()["error"] == "upstream_timeout"


def test_query_llm_http_timeout_returns_504(client: TestClient):
    with patch(
        "app.api.routes.answer_query",
        side_effect=httpx.ReadTimeout("llm slow"),
    ):
        response = client.post("/query", json={"question": "fees?", "top_k": 2})
    assert response.status_code == 504
    assert response.json()["error"] == "upstream_timeout"


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
    assert body["retrieval"] == {
        "method": "hybrid",
        "top_k": 2,
        "reranker": None,
        "candidates_retrieved": None,
    }
    hybrid.assert_called_once()


def test_query_rerank_422_when_provider_disabled(client: TestClient):
    client.app.state.resources.reranker = None
    response = client.post(
        "/query",
        json={"question": "fees", "use_rerank": True},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "rerank_unavailable"


def test_query_with_rerank(client: TestClient):
    reranker = MagicMock()
    reranker.name = "cross_encoder:test"
    reranker.rerank.return_value = [
        SearchHit(
            doc_id="1",
            doc_title="T",
            chunk_id=0,
            chunk_text="ctx",
            score=9.0,
        )
    ]
    client.app.state.resources.reranker = reranker
    response = client.post(
        "/query",
        json={"question": "What is APR?", "top_k": 1, "use_rerank": True},
    )
    assert response.status_code == 200
    assert response.json()["retrieval"]["reranker"] == "cross_encoder:test"
    reranker.rerank.assert_called_once()


def test_query_bm25_returns_422_when_index_missing(client: TestClient):
    client.app.state.resources.bm25_index = None
    response = client.post(
        "/query",
        json={"question": "fees", "method": "bm25"},
    )
    assert response.status_code == 422
    assert "BM25" in response.json()["message"]


def test_query_rejects_hybrid_when_frozen_profile_enforced():
    store = MagicMock()
    store.collection_exists.return_value = True
    settings = Settings(_env_file=None, api_enforce_frozen_retrieval=True)
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(),
        llm=_FakeLLM(),
        bm25_index=MagicMock(),
        ingest_manifest=MagicMock(),
        reranker=None,
    )
    app = create_app(resources=resources)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/query",
            json={"question": "fees", "method": "hybrid", "top_k": 10},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "frozen_profile_violation"
    assert "dense" in body["message"]


def test_query_default_top_k_is_frozen_when_enforced():
    store = MagicMock()
    store.collection_exists.return_value = True
    settings = Settings(_env_file=None, api_enforce_frozen_retrieval=True)
    resources = AppResources(
        settings=settings,
        store=store,
        embedder=MagicMock(),
        llm=_FakeLLM(),
        bm25_index=MagicMock(),
        ingest_manifest=MagicMock(),
        reranker=None,
    )
    app = create_app(resources=resources)
    with patch("app.api.routes.answer_query") as answer:
        from app.pipeline.rag import RagAnswer

        answer.return_value = RagAnswer(
            answer="ok",
            citations=[],
            retrieval_method="dense",
            reranker=None,
            retrieval_candidate_count=None,
        )
        with TestClient(app) as test_client:
            response = test_client.post("/query", json={"question": "APR?"})
    assert response.status_code == 200
    assert response.json()["retrieval"]["top_k"] == 10
    assert answer.call_args.kwargs["top_k"] == 10
