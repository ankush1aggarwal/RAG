# Week 8 sprint — Docker + CI

**Prerequisite:** Week 7 observability merged.

**Validation:** **GitHub Actions** builds the image and runs container smoke — **Docker Desktop is not required** on your laptop.

**End state:** Image on **GHCR**; pytest + Ruff CI (Block 2).

## Block 0 — Container & runtime 🟡

- [x] `Dockerfile` (non-root `app` user, `0.0.0.0` bind, healthcheck)
- [x] `.dockerignore` (no `.env`, no `data/`, no notebooks)
- [x] Optional `docker-compose.yml` (operators with Docker Engine + mounted index)
- [x] [`docs/docker.md`](docker.md)
- [x] `src/scripts/ci_container_smoke.sh` + workflow job **Container smoke**
- [ ] **CI green** on PR (build + `/health` in container)

## Block 1 — GHCR publish 🟡

- [x] `.github/workflows/container.yml` — smoke on PR; push on `main` / tags / manual
- [ ] **You:** merge to `main`; set GHCR package visibility
- [ ] **You (optional):** pull image on a Linux VM and run full `smoke_api.py` with Qdrant + Ollama mounts

## Block 2 — CI (next)

- [ ] GitHub Actions: Ruff + `pytest src/tests` (separate or combined workflow)

## Block 3 — Reproducibility (next)

- [ ] Pin `requirements-git.txt` to commit SHA when using remote editable install

**Definition of done (Week 8):** GHCR image from `main`; Actions build + container smoke + unit CI; documented pull/run for production-like smoke with volumes.
