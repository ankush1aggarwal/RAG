# HTTP API contract (Week 6)

Base URL (local): `http://127.0.0.1:8000`

**Public demo (Cloud Run):** https://fiqa-rag-api-729436472430.asia-south1.run.app — example requests, demo quotas (413/429), and release process: [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md).

OpenAPI: `/docs` (FastAPI auto-generated from Pydantic models).

## Endpoints

| Method | Path | Success | Purpose |
|--------|------|---------|---------|
| `GET` | `/health` | `200` | Liveness (no dependency checks) |
| `GET` | `/ready` | `200` / `503` | Readiness for `/query` |
| `POST` | `/query` | `200` | RAG: retrieve + generate + citations |
| `POST` | `/query/stream` | `200` (`text/event-stream`) | Same RAG path; SSE token stream + final payload |

## Success bodies

- **`GET /health`:** `{"status": "ok"}`
- **`GET /ready`:** `ReadyResponse` — `status`, `qdrant_collection`, `bm25_index`, `retrieval_index_aligned`, `llm_provider`, `llm_ready`
- **`POST /query`:** `QueryResponse` — `answer`, `citations`, `retrieval`
- **`POST /query/stream`:** Server-Sent Events (same request body as `/query`):
  - `event: token` — `{"delta": "<text chunk>"}` (one or more)
  - `event: done` — full `QueryResponse` JSON (includes assembled `answer`, `citations`, `retrieval`)
  - `event: error` — same shape as `ErrorResponse` when failure occurs **after** the stream has started (e.g. timeout, pipeline error)

**Before** the response is committed to `text/event-stream`, the same validation as `POST /query` applies: frozen profile, rerank availability, and `query_timeout_seconds` → **422 JSON** (not an SSE `error` under HTTP 200). Pydantic body validation is also JSON **422**.

### Request protection

- Maximum request body: **16 KiB** (413 `payload_too_large`).
- Maximum question length: **2,000 characters** (422 `validation_error`).
- `POST /query` and `POST /query/stream`: **10 requests per 60 seconds per running instance** (429 `rate_limit_exceeded`, with `Retry-After`).
- `GET /health` and `GET /ready` are excluded from the query quota.

The quota is process-local, not per-user or per-IP. It resets when the Cloud Run instance restarts. It is a lightweight demo guard, not distributed abuse protection. Keep Cloud Run maximum instances at one while using it.

### Frozen production profile (default)

When `API_ENFORCE_FROZEN_RETRIEVAL=true`:

- **Settings** must match E8 (`production_retrieval.py`) at startup.
- **`POST /query`:** `method=dense`, `use_rerank=false`, `top_k=10` only.
- **`GET /ready`:** Qdrant + manifest/dense alignment + LLM; BM25 reported but not required. Manifest path: `{BM25_INDEX_DIR}/{collection}.manifest.json` (Cloud Run: copy from local ingest into `deploy/index_metadata/` after re-index when counts/settings change — see [`cloud_run.md`](cloud_run.md)). Qdrant Cloud migration: [`cloud_run.md#migrate-local-collection--qdrant-cloud`](cloud_run.md#migrate-local-collection--qdrant-cloud). **Hosted example:** [`public_demo_operations_runbook.md` §4–§5](public_demo_operations_runbook.md#4-api-endpoints) — `bm25_index: false` is expected for frozen dense-only.

### `GET /ready` fields

| Field | Meaning |
|--------|---------|
| `status` | `"ready"` or `"not_ready"` (informational; use HTTP status for gating) |
| `qdrant_collection` | Named collection exists in Qdrant |
| `retrieval_index_aligned` | **Gate for `/query`:** dense manifest+Qdrant+embedder match when frozen; full bundle (incl. BM25) when enforcement is off |
| `bm25_index` | Lexical index loaded **and** aligned with Qdrant+manifest (`false` if pickle missing or misaligned) |
| `llm_provider` | `ollama` or `cohere` from settings |
| `llm_ready` | Ollama: `GET /api/tags` succeeds; Cohere: API key configured |

On **503**, the same keys appear under `detail`, plus `full_ingest_bundle_aligned` and `frozen_retrieval_enforced` for operators.

## Error body (`ErrorResponse`)

All non-2xx responses handled by the API exception layer use:

```json
{
  "error": "<machine_code>",
  "message": "<human-readable summary>",
  "detail": <optional object>,
  "request_id": "<uuid or client-supplied id>"
}
```

Every response includes header **`X-Request-ID`** (client may send one; otherwise the server generates a UUID). Client values must match **`[A-Za-z0-9._:-]`**, length **1–128** after trim; invalid or overlong values are replaced with a server UUID. Error bodies echo the same id in `request_id` when available.

| HTTP | `error` code | When |
|------|----------------|------|
| `422` | `validation_error` | Invalid JSON or Pydantic request validation |
| `422` | `frozen_profile_violation` | Frozen profile: wrong `method`, `top_k`, or `use_rerank` |
| `422` | `invalid_request` | Business rule (e.g. BM25/hybrid without index, empty question via pipeline) |
| `422` | `rerank_unavailable` | `use_rerank=true` but no reranker provider |
| `503` | `not_ready` | `/ready` — dependencies missing or misaligned (`detail` = diagnostics dict) |
| `413` | `payload_too_large` | Request body exceeds the 16 KiB limit |
| `429` | `rate_limit_exceeded` | Query request quota exceeded |
| `504` | `upstream_timeout` | `/query` wall-clock limit (`QUERY_TIMEOUT_SECONDS`) or LLM HTTP timeout |
| `500` | `internal_error` | Unhandled server exception (no stack trace in response) |

**Timeouts:** `QUERY_TIMEOUT_SECONDS` (default `120`) bounds each `/query` and `/query/stream` request. On expiry the API sets a cooperative **cancel flag** and closes in-flight Ollama streams; sync thread work is not force-killed. Non-streaming LLM calls also use `OLLAMA_TIMEOUT_SECONDS` on the HTTP client. Breaches map to `504` / `upstream_timeout`. Client/task **cancellation** (`asyncio.CancelledError`) is re-raised after cleanup (not mapped to JSON errors).

### Examples

**Validation (empty question):**

```http
POST /query
{"question": ""}
```

```json
{
  "error": "validation_error",
  "message": "Request validation failed.",
  "detail": [ ... ]
}
```

**Frozen profile:**

```json
{
  "error": "frozen_profile_violation",
  "message": "Production API serves dense retrieval only; got method='hybrid'.",
  "detail": null
}
```

**Not ready:**

```json
{
  "error": "not_ready",
  "message": "Service dependencies are not ready.",
  "detail": {
    "status": "not_ready",
    "qdrant_collection": false,
    "bm25_index": false,
    "retrieval_index_aligned": false,
    "full_ingest_bundle_aligned": false,
    "frozen_retrieval_enforced": true,
    "llm_provider": "ollama",
    "llm_ready": false
  }
}
```

## Local smoke

With the API running (`python src/scripts/serve_api.py`):

```bash
python src/scripts/smoke_api.py
python src/scripts/smoke_api.py --stream   # includes SSE /query/stream
```

Or manually (local):

```bash
# Liveness + correlation id
curl -s -H "X-Request-ID: local-dev-1" http://127.0.0.1:8000/health -D -

curl -s http://127.0.0.1:8000/ready

# RAG (frozen profile: omit top_k/method or use dense + top_k=10)
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: local-dev-1" \
  -d "{\"question\": \"What is APR?\"}"

# Validation error (structured ErrorResponse)
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"\"}"

# Frozen violation (when enforcement on)
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"APR?\", \"method\": \"hybrid\"}"

# Streaming
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is APR?\"}"
```

## Public demo smoke

Hosted curls (no auth): [`public_demo_operations_runbook.md` §4](public_demo_operations_runbook.md#4-api-endpoints). Demo quotas (413 / 422 / 429 on `/query`): runbook §3 — may differ from local dev.

Example **validation** body (`422`):

```json
{
  "error": "validation_error",
  "message": "Request validation failed.",
  "detail": [],
  "request_id": "local-dev-1"
}
```

OpenAPI examples also appear under `/docs` (from Pydantic `examples` on request/response models).
