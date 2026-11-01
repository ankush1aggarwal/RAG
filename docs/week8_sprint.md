# Week 8 sprint — Docker + CI ✅

**Prerequisite:** Week 7 observability merged.

**Validation:** **GitHub Actions** — no Docker Desktop required locally.

**Status:** GHCR public image; CI + container workflows on `main`.

## Block 0 — Container & runtime ✅

- [x] `Dockerfile`, `.dockerignore`, optional `docker-compose.yml`
- [x] [`docs/docker.md`](docker.md), `ci_container_smoke.sh`
- [x] Container workflow green on `main`

## Block 1 — GHCR publish ✅

- [x] `.github/workflows/container.yml`
- [x] Public package `ghcr.io/<owner>/fiqa-production-rag`

## Block 2 — CI ✅

- [x] `.github/workflows/ci.yml` — Ruff (`src/app`, `src/tests`) + pytest

## Block 3 — Reproducibility ✅

- [x] `requirements-git.txt` pinned to commit SHA (bump on release)

## Docs ✅

- [x] [`docs/architecture.md`](architecture.md) — dev / Actions / GHCR / Week 9 topology
- [x] README, roadmap, `PROJECT_SPEC` updated for Week 8 complete

**Definition of done:** met.

**Next:** Week 9 — Qdrant Cloud + hosted API ([`RAG_PROJECT_ROADMAP.md`](RAG_PROJECT_ROADMAP.md)).
