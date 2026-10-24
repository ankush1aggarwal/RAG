# Week 6 sprint — Production API

**Branch:** `week6_production_api` (or your Week 6 branch name)  
**Prerequisite:** Week 5 frozen retrieval; `.env` aligned with E8 (see `.env.example`).

## Block 0 — Baseline ✅

- [x] Document sprint scope (`docs/week6_sprint.md`, `RAG_PROJECT_ROADMAP.md`)
- [x] HTTP contract reference (`docs/api_contract.md`)
- [x] Local smoke script (`src/scripts/smoke_api.py`)
- [ ] You: branch from latest `main`, confirm `.env` + Qdrant index `fiqa_cs200_bge_base_en_v1_5`, Ollama up
- [ ] You: run smoke script against a live server

## Block 1 — API contract & errors ✅

- [x] `ErrorResponse` schema + stable `error` codes (`src/app/api/errors.py`)
- [x] Exception handlers registered in `create_app()`
- [x] `/ready` and `/query` use structured errors
- [x] Unit tests for error shape

## Block 2 — Request ID ✅

- [x] `RequestIdMiddleware` + `X-Request-ID` on every response
- [x] `request_id` on `ErrorResponse`
- [x] Tests

## Block 3 — Timeouts ✅

- [x] `QUERY_TIMEOUT_SECONDS` wall-clock on `/query` → `504` / `upstream_timeout`
- [x] `httpx.TimeoutException` from LLM → `504`
- [x] Documented in `api_contract.md`

## Block 4 — Readiness polish ✅

- [x] `/ready` field reference in `docs/api_contract.md` + README §8
- [x] Single-pass `ready_diagnostics()` (no duplicate dense/BM25 alignment work)

## Block 5 — Streaming ✅

- [x] `POST /query/stream` (SSE: `token`, `done`, `error`)
- [x] Ollama `stream_generate`; other providers fall back to one chunk via `generate`
- [x] Unit tests

## Block 6 — Integration tests ✅

- [x] `src/tests/integration/` — OpenAPI, middleware + errors + stream + frozen + lifespan

## Block 7 — Docs / OpenAPI polish ✅

- [x] Pydantic examples on `QueryRequest` / `QueryResponse` / `ErrorResponse`
- [x] FastAPI description + tags; `docs/api_contract.md` curl + error samples
- [x] README quick-start API section updated

## Block 8 — Lifecycle & smoke ✅

- [x] Lifespan shutdown calls `close_resources` (integration test)
- [x] `smoke_api.py`: `X-Request-ID` check + `--stream`
