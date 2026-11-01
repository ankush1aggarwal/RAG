# Week 7 sprint — Observability (structured logs) ✅

**Prerequisite:** Week 6 production API merged or on branch with `/query` + streaming.

**Approach:** Stdlib **JSON logs to stderr** only — **no OpenTelemetry, no Langfuse.**

**Status:** Code, tests, and ops guide complete. Log aggregation in deployment deferred to Week 8+.

## Block 0 — Scope ✅

- [x] Sprint doc (`docs/week7_sprint.md`)
- [x] Ops guide (`docs/observability_guide.md`)
- [x] Roadmap/README updated (no OTel/Langfuse)

## Block 1 — Core module ✅

- [x] `src/app/observability/` — context, redaction, JSON formatter, `emit`, `stage_timer`
- [x] Settings: `LOG_LEVEL`, `STRUCTURED_LOGGING_ENABLED`, `LOG_MAX_FIELD_CHARS`
- [x] Configure logging in API lifespan

## Block 2 — Instrumentation ✅

- [x] `X-Request-ID` → log context (middleware + thread pool)
- [x] RAG stages: `retrieval`, `rerank`, `prompt_build`, `generation`
- [x] API: `query_start` / `query_complete` / `query_failed`
- [x] Token/cost fields when provider returns usage (Ollama/Cohere; optional Cohere USD rates in settings)

## Block 3 — Tests & docs ✅

- [x] Redaction + stage timer unit tests (`test_observability.py`, `test_usage.py`)
- [x] Caplog/integration test on `/query` (`test_api_http.py::test_query_emits_correlated_structured_logs`)

## Block 4 — You

- [x] Run API locally with `LOG_LEVEL=INFO`; confirm one request produces correlated JSON lines
- [ ] Wire log aggregation in deployment (**Week 8+**)

**Definition of done:** met — trace a request by `request_id` through stage timings and optional token fields without logging secrets.

**Next:** Week 8 — Docker + CI ([`RAG_PROJECT_ROADMAP.md`](RAG_PROJECT_ROADMAP.md)).
