# Week 9 sprint — Deployment + public demo ✅

**Status:** Complete (API-only portfolio demo; no separate `frontend/` UI).

**Deployed Cloud Run image (GHCR):** `4f5ddf3` on `main` — see [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md).

## Done

- [x] Qdrant Cloud + ingest manifest in image (`deploy/index_metadata/`)
- [x] Cloud Run `fiqa-rag-api` (public URL in runbook §1)
- [x] Secret Manager (`cohere-api-key`, `qdrant-api-key`)
- [x] Frozen dense profile + Cohere generation on hosted service
- [x] Public invoke (`allUsers` + `run.invoker`) + in-app demo request limits
- [x] Hosted validation `/health`, `/ready`, `/query` (runbook §4–§5)
- [x] Manual release: GHCR on `main` → `gcloud run deploy` with commit SHA (runbook §6–§7)
- [x] README demo screenshots (`live_fiqa_rag_api.png`, `live_api_query_repsonse.png`)

## Explicitly out of scope / parked

- **Demo UI** — not building; use public API + [`/docs`](https://fiqa-rag-api-729436472430.asia-south1.run.app/docs) OpenAPI.
- **E10** citation rubric evaluation — parked.
- **Automated Cloud Run deploy** (WIF workflow) — parked; manual release only.
- **Load / traffic benchmarking** — parked (portfolio demo app).

## Related

- [`cloud_run.md`](cloud_run.md) — app/env, Qdrant migration
- [`cloud_run_deployment.md`](cloud_run_deployment.md) — first GCP deploy history
- [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) — current ops
- [`RAG_PROJECT_ROADMAP.md`](RAG_PROJECT_ROADMAP.md) — Final polish checklist

**Next:** Final polish (blog, LinkedIn, clean-environment repro) per roadmap.
