# Observability guide (Week 7 ✅)

Structured **JSON logs** on **stderr**. No OpenTelemetry or Langfuse in this project.

## Enable / configure

Environment variables (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOG_LEVEL` | `INFO` | Root log level |
| `STRUCTURED_LOGGING_ENABLED` | `true` | JSON lines vs plain text |
| `LOG_MAX_FIELD_CHARS` | `120` | Truncate long strings (non-sensitive) |

API startup calls `configure_structured_logging()` from the app lifespan.

## Correlation

- HTTP header **`X-Request-ID`** is copied into every log event as **`request_id`** when present. Allowed characters: letters, digits, `_`, `.`, `:`, `-` (max **128** chars after trim); invalid values become a server-generated UUID.
- Worker threads (RAG pipeline) inherit the same id via `run_bounded_sync(..., request_id=...)`.

## Event types

| `event` | When |
|---------|------|
| `query_start` | `/query` or `/query/stream` accepted (after validation) |
| `query_complete` | Success; includes `duration_ms`, `citation_count`, `retrieval_method`, `top_k`; optional `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated_cost_usd`, `provider` when the LLM returns usage |
| `query_failed` | Handler failure before HTTP mapping (sync `/query` and in-stream `/query/stream`) |
| `llm_usage` | Token/cost snapshot after generation (same fields as on `query_complete`) |
| `stage_start` / `stage_end` | RAG sub-stage timing (`retrieval`, `rerank`, `prompt_build`, `generation`) |
| `stage_error` | Stage raised an exception |
| `retrieval_complete` | Hits ready; `hit_count`, optional `reranker` |

Each line is one JSON object, e.g.:

```json
{"timestamp":"2026-11-01T12:00:00+00:00","level":"INFO","logger":"fiqa.rag","message":"stage_end","event":"stage_end","stage":"retrieval","duration_ms":42.1,"request_id":"abc","retrieval_method":"dense","top_k":10}
```

## Privacy

By default we **do not** log:

- Raw questions, prompts, answers, or chunk text
- API keys or credentials

Sensitive keys are replaced with `[REDACTED]`; omitted text fields show `[omitted len=N]`.

## Local inspection

```powershell
# From repo root — use the script path (NOT ``python -m src/scripts/serve_api``):
python src/scripts/serve_api.py 2> api.log
# In another terminal:
curl -s -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" `
  -H "X-Request-ID: demo-1" -d "{\"question\": \"What is APR?\"}"
Select-String -Path api.log -Pattern "demo-1"
```

Filter on `request_id` and sum `duration_ms` by `stage` to approximate retrieval vs generation latency.
