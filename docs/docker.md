# Docker runtime (Week 8)

Run the **FiQA RAG API** in a container without baking secrets or local Qdrant data into the image.

**Validation strategy:** You do **not** need Docker Desktop locally. **GitHub Actions** builds the image, runs an in-runner **container smoke** (`/health`), and pushes to **GHCR** on `main` / tags (see [`.github/workflows/container.yml`](../.github/workflows/container.yml)). **Cloud Run is not updated automatically** — see [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md).

## What the image contains

- Python 3.12 slim + app install: `pip install ".[api,embeddings,hosted]"` (FastAPI, Uvicorn, sentence-transformers, **cohere** for Cloud Run chat, Qdrant client, etc.)
- **Baked at build:** `BAAI/bge-base-en-v1.5` weights under `/app/.cache/huggingface` (`src/scripts/prefetch_embedding_model.py` during `docker build`) — avoids Hub downloads on Cloud Run cold start
- **Not included:** `.env`, Qdrant data, BM25 pickles
- **Not included:** optional **`datasets`** (ingest/eval scripts only). The API imports `app.ingestion.schemas` only; `app/ingestion/__init__.py` must not import `corpus`/`normalize` (see `test_api_startup_imports.py`).

See also: [`architecture.md`](architecture.md), [`cloud_run.md`](cloud_run.md) (app/env/migration), [`cloud_run_deployment.md`](cloud_run_deployment.md) (first deploy), [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) (**public demo + manual Cloud Run release**).

### Production LLM vs local Compose

| Runtime | `LLM_PROVIDER` | Notes |
|---------|----------------|--------|
| **Cloud Run** (intended prod) | `cohere` | Set `COHERE_API_KEY` in Secret Manager; image includes **`hosted`** extra |
| **Local `.venv` / Compose** | `ollama` (default) | Point `OLLAMA_BASE_URL` at host Ollama; override in `.env` if needed |

## Qdrant: local path vs Qdrant Cloud

- **Local / Compose:** `QDRANT_PATH=data/qdrant_local` (default); leave `QDRANT_URL` unset. Index on disk: `data/qdrant_local/collection/<COLLECTION_NAME>/`.
- **Cloud Run:** set `QDRANT_URL` and `QDRANT_API_KEY`; do not rely on container disk for the vector index.

**Migrate local → cloud:** re-ingest with `QDRANT_URL` set (not a folder copy). **After migration:** toggle local vs cloud by setting or unsetting `QDRANT_URL` in `.env`; see [`cloud_run.md` — Migrate local collection → Qdrant Cloud](cloud_run.md#migrate-local-collection--qdrant-cloud) and [Local vs cloud after migration](cloud_run.md#local-vs-cloud-after-migration).

## CI container smoke (primary)

On every PR and push to `main`:

1. `docker build` → tag `fiqa-production-rag:ci`
2. `src/scripts/ci_container_smoke.sh` — starts the container, waits for **`GET /health`** (allows time for the embedding model download), logs **`/ready`** status (503 is expected without a production index and Ollama)
3. On non-PR events, push to **`ghcr.io/<github-owner>/fiqa-production-rag`**

Reproduce the smoke script on any Linux host with Docker Engine (optional — e.g. cloud VM, WSL2 + `docker.io`, not Docker Desktop):

```bash
docker build -t fiqa-production-rag:ci .
bash src/scripts/ci_container_smoke.sh fiqa-production-rag:ci
```

## Optional local run (Docker Engine + Compose)

Use this only if you have a Docker CLI and want to hit **`/query`** against a **mounted** Qdrant index and host Ollama. [`docker-compose.yml`](../docker-compose.yml) is optional convenience, not required for Week 8 sign-off.

1. Copy [`.env.example`](../.env.example) to `.env`.
2. Mount your Qdrant store (e.g. `./data/qdrant_local` with collection `fiqa_cs200_bge_base_en_v1_5`).
3. Ollama on the host or Cohere via env.

```bash
docker compose up --build
python src/scripts/smoke_api.py --base-url http://127.0.0.1:8000
```

On Linux, `OLLAMA_BASE_URL=http://host.docker.internal:11434` uses `extra_hosts: host-gateway` in Compose.

### Readiness vs liveness

- **`/health`** — liveness once Uvicorn and startup (embedder load) complete; **this is what CI asserts**.
- **`/ready`** — needs aligned Qdrant collection, embedder, and LLM; uses the ingest manifest under `BM25_INDEX_DIR` (image default: `deploy/index_metadata/`). Full RAG smoke remains a **manual** step with mounted data + Ollama/Cohere.

**After re-ingest:** copy `data/bm25_indexes/<collection>.manifest.json` → `deploy/index_metadata/`, commit, rebuild the image — see [`deploy/index_metadata/README.md`](../deploy/index_metadata/README.md).

## GitHub Container Registry (GHCR)

| Event | Behavior |
|--------|----------|
| Pull request | Build + container smoke; **no** GHCR push |
| Push to `main` | Build + smoke + push `latest` and SHA tags; **no Cloud Run deploy** |
| Tag `v*` | Semver tags |
| `workflow_dispatch` | Manual build, smoke, push |

After a successful push to `main`, redeploy Cloud Run manually with the merge commit's GHCR SHA tag — [`public_demo_operations_runbook.md` §7](public_demo_operations_runbook.md#7-manual-release-procedure-after-a-merge-to-main).

**Image:** `ghcr.io/<github-owner>/fiqa-production-rag`

After the first push to `main`:

1. GitHub → **Packages** → **fiqa-production-rag** → **Package settings** → visibility (**Public** for portfolio demos).
2. Pull on a machine with Docker:  
   `docker pull ghcr.io/<owner>/fiqa-production-rag:latest`

## Operational notes

- Mount Qdrant read-only for query-only deployments.
- Plan disk for `hf_cache` and Qdrant separately from image layers.
- Logs on **stderr** → `docker logs` or your aggregator ([`observability_guide.md`](observability_guide.md)).

See also: [`api_contract.md`](api_contract.md), [`week8_sprint.md`](week8_sprint.md).
