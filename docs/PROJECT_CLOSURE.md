# Project closure — Production-Grade FiQA RAG

**Closed:** November 2026  
**Repository:** [ankush1aggarwal/RAG](https://github.com/ankush1aggarwal/RAG)

## One-line summary

**End-to-end FiQA RAG built from scratch:** retrieval experiments and qrels-driven metrics informed a **frozen dense BGE production profile**; the same codebase delivers CLI evaluation, a tested **FastAPI** service, and a **public Cloud Run** demo—combining **applied IR / data science** with **production AI engineering**.

## Problem and approach

Financial question answering over **BEIR FiQA-2018** using a **production-minded RAG stack** (Qdrant, BM25, hybrid fusion, optional rerank, LLM + citations)—implemented **independently** in `src/app/`, not by cloning a reference repository.

Development followed a **measure → decide → ship** loop: explore retrieval variants and chunking in Weeks 3–5, **freeze** the served design, then harden API, observability, containers, and hosted deployment (Weeks 6–9).

## Key evidence

| Area | Where to look |
|------|----------------|
| Retrieval quality + experiments | [`reports/rag_experiment_results.md`](reports/rag_experiment_results.md), [`reports/rag_retrieval_lessons.md`](reports/rag_retrieval_lessons.md) |
| Frozen production profile | `fiqa_cs200_bge_base_en_v1_5` — dense, BGE-base, 200w / 20%, top-k 10; test NDCG@10 **0.394** (see results report) |
| Live demo | https://fiqa-rag-api-729436472430.asia-south1.run.app — OpenAPI `/docs`; screenshots in [README](../README.md#live-demo-hosted) |
| Operations | [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) (manual GHCR → Cloud Run release) |
| Engineering topology | [`architecture.md`](architecture.md) |

## Delivered vs parked

**Delivered:** full ingest/index pipeline; hybrid/rerank in **CLI/eval**; frozen **dense** production API; Cohere on hosted demo; Secret Manager; CI + GHCR; public API with demo request limits; structured logs; integration tests; experiment and deployment documentation.

**Parked (documented, not gaps):** separate demo **web UI**; **E10** citation rubric eval; **E11** formal load/concurrency benchmark; **automated** Cloud Run deploy (WIF); Langfuse/OTel (stdlib JSON logs only).

## Skills demonstrated

- **Data science / applied IR:** qrels evaluation, ablation design, metric interpretation, failure analysis, production freeze rationale.  
- **Engineering:** API contracts, readiness gates, manifest alignment, Docker/Actions, cloud env/secrets, rate limiting, reproducible scripts, runbooks.

## Next (outside this repo)

**Phase 2 — Agentic RAG** on the same baseline (planning only): multi-step retrieval, tools, evidence checks — see [`RAG_PROJECT_ROADMAP.md`](RAG_PROJECT_ROADMAP.md) Phase 2 section.
