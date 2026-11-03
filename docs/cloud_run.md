# Google Cloud Run deployment (Week 9)

Deploy the **GHCR API image** on **Cloud Run** with **Qdrant Cloud** for vectors. Local development keeps using `QDRANT_PATH`; production uses `QDRANT_URL`.

| Doc | Use when |
|-----|----------|
| [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) | **Live public demo**, curl examples, request limits, **manual release after `main`**, rollback |
| [`cloud_run_deployment.md`](cloud_run_deployment.md) | First-deploy history (IAM, Secret Manager, initial authenticated validation) |
| **This file** | App/env prep, Qdrant migration, manifest + BGE in the image |

**Automation boundary:** GitHub Actions publishes images to **GHCR** on `main`; **Cloud Run deploy is manual** (see public demo runbook §6–§7).

## Qdrant connection

| Variable | Local dev | Cloud Run + Qdrant Cloud |
|----------|-----------|---------------------------|
| `QDRANT_URL` | unset | `https://<cluster>.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | unset | cluster API key (Secret Manager → env) |
| `QDRANT_PATH` | `data/qdrant_local` | ignored when `QDRANT_URL` is set |

Connection logic (API, ingest, query, eval scripts):

1. If **`QDRANT_URL`** is set → remote `QdrantClient` (Cloud).
2. Else → existing **local path** or in-memory behaviour via `QDRANT_PATH`.

Implementation: `open_qdrant_chunk_store()` in `src/app/retrieval/qdrant_connection.py`.

**Local on-disk index (reference):** embedded Qdrant data lives under `data/qdrant_local/` (gitignored), collection folder `collection/fiqa_cs200_bge_base_en_v1_5/`. That directory is **not** copied to Qdrant Cloud; use re-ingest below.

## Migrate local collection → Qdrant Cloud

Supported path: **re-run ingest** with cloud env so chunks are embedded on your machine and **uploaded** via the Qdrant client (`upload_points`). Do **not** copy `data/qdrant_local/` to the cloud — layout and hosting differ.

### 1. Provision Qdrant Cloud

1. Create a cluster at [cloud.qdrant.io](https://cloud.qdrant.io).
2. Note **cluster URL** (`https://<cluster-id>.cloud.qdrant.io:6333`) and an **API key**.
3. Size for the frozen E8 profile: **~72,851** points, **768**-dim vectors (`fiqa_cs200_bge_base_en_v1_5`).

### 2. Configure `.env` for cloud ingest

Use the same frozen settings as production (defaults in `src/app/config/production_retrieval.py`):

```env
QDRANT_URL=https://<cluster-id>.cloud.qdrant.io:6333
QDRANT_API_KEY=<secret>
COLLECTION_NAME=fiqa_cs200_bge_base_en_v1_5
EMBEDDING_METHOD=sentence_transformer
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
FIQA_CORPUS_LIMIT=0
CHUNK_SIZE=200
CHUNK_OVERLAP_FRACTION=0.2
```

When `QDRANT_URL` is set, **`QDRANT_PATH` is ignored** for API, ingest, query, and eval scripts.

### 3. Run ingest

From the repo root (venv with embeddings extra installed):

```powershell
$env:PYTHONPATH = "src"
python src/scripts/ingest.py --full-corpus --collection fiqa_cs200_bge_base_en_v1_5
```

```bash
PYTHONPATH=src python src/scripts/ingest.py --full-corpus --collection fiqa_cs200_bge_base_en_v1_5
```

Default **`--recreate`** drops and rebuilds the cloud collection if it already exists. Expect a long run (corpus load + ~72k BGE embeddings + upload).

**Verify:** Qdrant Cloud UI → collection **`fiqa_cs200_bge_base_en_v1_5`** → **72,851** points; or:

```powershell
$env:PYTHONPATH = "src"
python src/scripts/query.py --method dense --query "What is an ETF expense ratio?" --top-k 3
```

BM25 pickles and the ingest manifest are still written under **`data/bm25_indexes/`** on your laptop. Hybrid `/ready` on Cloud Run needs a separate BM25 strategy; the **dense-only** frozen profile does not require BM25 in the container.

### 4. Manifest and image

If point counts or chunk/embedding settings change vs the baked manifest, copy the new manifest into `deploy/index_metadata/` (see below). If cloud ingest matches the committed manifest (**72,851** chunks, **768** dims), you can skip that step.

Optional: Qdrant **snapshots** (local server → restore on cloud) are documented by Qdrant but **not** wired in this repo; re-ingest stays the repeatable Week 9 path.

## Local vs cloud after migration

You can keep **both** a local disk index and a Qdrant Cloud collection. Only **`.env`** (or process env) selects which backend the app uses:

| Goal | `.env` |
|------|--------|
| **Local vectors** (default dev) | Leave **`QDRANT_URL` unset** (or comment it out). Use **`QDRANT_PATH=data/qdrant_local`**. Reads/writes `data/qdrant_local/collection/<COLLECTION_NAME>/`. |
| **Cloud vectors** (prod API, cloud smoke, re-ingest) | Set **`QDRANT_URL`** + **`QDRANT_API_KEY`**. **`QDRANT_PATH`** is ignored. |
| **Ingest to local only** | Unset `QDRANT_URL`; run `ingest.py` — updates local Qdrant + `data/bm25_indexes/`. |
| **Ingest to cloud only** | Set `QDRANT_URL` + key; run `ingest.py` — updates cloud collection; BM25/manifest still local unless you add storage later. |

**Cloud Run / GHCR container:** set **`QDRANT_URL`**, **`QDRANT_API_KEY`**, and **`COLLECTION_NAME`** on the service (Secret Manager). Do not mount `data/qdrant_local` for production.

**Switching back to local** after testing cloud: remove or comment **`QDRANT_URL`** and **`QDRANT_API_KEY`** in `.env`, restart the API or shell. Your existing **`data/qdrant_local`** index is unchanged while you were pointed at cloud.

**Do not run destructive ingest against both at once** without meaning to: with `QDRANT_URL` set, `--recreate` affects **cloud** only; with URL unset, `--recreate` affects **local** only.

## Ingest manifest (frozen dense `/ready`)

`/ready` compares Qdrant to the **ingest manifest** (embedding model, chunk settings, point counts). The manifest path is derived from `BM25_INDEX_DIR` + `COLLECTION_NAME`:

`{BM25_INDEX_DIR}/{collection}.manifest.json`

The **`.pkl` is optional** when `API_ENFORCE_FROZEN_RETRIEVAL=true` (dense-only production profile).

The image ships a pinned manifest at:

```text
deploy/index_metadata/fiqa_cs200_bge_base_en_v1_5.manifest.json
```

Default container env: **`BM25_INDEX_DIR=deploy/index_metadata`**.

**After a new ingest** (local or into Qdrant Cloud), refresh the committed manifest before rebuilding/pushing the image:

```powershell
Copy-Item data\bm25_indexes\fiqa_cs200_bge_base_en_v1_5.manifest.json deploy\index_metadata\
```

```bash
cp data/bm25_indexes/fiqa_cs200_bge_base_en_v1_5.manifest.json deploy/index_metadata/
```

Commit `deploy/index_metadata/*.manifest.json`. See [`deploy/index_metadata/README.md`](../deploy/index_metadata/README.md).

## Embedding model cache (cold start)

Cloud Run’s writable disk is **ephemeral** — caching only in `/tmp` or `/data` is lost on scale-to-zero.

The Docker build runs:

```bash
PYTHONPATH=src python src/scripts/prefetch_embedding_model.py
```

with `HF_HOME=/app/.cache/huggingface`, baking **BGE-base** into the image. Runtime uses the same `HF_HOME` (see `Dockerfile`). Rebuild the image when `EMBEDDING_MODEL` / frozen profile changes.

## LLM (production: Cohere)

Cloud Run does **not** run Ollama in the same container. **Production generation** uses **Cohere chat**:

| Variable | Cloud Run (prod) | Local dev |
|----------|------------------|-----------|
| `LLM_PROVIDER` | **`cohere`** | `ollama` (default in `settings.py`) |
| `COHERE_API_KEY` | Required (secret) | Optional |
| `COHERE_CHAT_MODEL` | `command-r-08-2024` (default) | Same if using Cohere |
| `OLLAMA_*` | Unused | Ollama on localhost / Compose `host.docker.internal` |

The GHCR image installs **`pip install ".[api,embeddings,hosted]"`** so `LLM_PROVIDER=cohere` works without an extra install step.

**Retrieval** stays **local sentence-transformers** + baked BGE (`EMBEDDING_METHOD=sentence_transformer`); do not switch prod embed to Cohere unless you re-ingest and change the manifest.

`/ready` with Cohere checks that `COHERE_API_KEY` is set; Qdrant + manifest alignment still apply.

## Cloud Run service (summary)

| Topic | Where to look |
|--------|----------------|
| Env vars, secrets, frozen profile | This file + [`.env.example`](../.env.example) |
| **Public URL**, smoke curls, **manual image release** | [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) |
| First GCP deploy (IAM, secrets, authenticated validation) | [`cloud_run_deployment.md`](cloud_run_deployment.md) |
| Image build and GHCR (no auto-deploy) | [`docker.md`](docker.md) |

**Runtime checklist:**

1. **Image:** prefer an immutable GHCR tag (`ghcr.io/<owner>/fiqa-production-rag:<git-sha>`). **Current demo:** `4f5ddf3`.
2. **Port:** Cloud Run sets **`PORT`** (typically `8080`). The container image defaults `PORT=8080`; `serve_api.py` binds to `PORT` unless you pass `--port`. Local dev without `PORT` uses **8000**.
3. **Secrets / env:** `QDRANT_URL`, `QDRANT_API_KEY`, **`LLM_PROVIDER=cohere`**, **`COHERE_API_KEY`**, `COLLECTION_NAME`, `API_ENFORCE_FROZEN_RETRIEVAL=true`, `BM25_INDEX_DIR=deploy/index_metadata`.
4. **Memory / CPU:** sized for loading baked BGE from disk (successful first deploy used **2 CPU / 4Gi** — details in deployment record).
5. **Health checks:** liveness → `GET /health`; readiness → `GET /ready` once Qdrant + LLM are configured. Public smoke examples: [`public_demo_operations_runbook.md` §4](public_demo_operations_runbook.md#4-api-endpoints).

**Production service (current):** public Cloud Run API **`fiqa-rag-api`** — https://fiqa-rag-api-729436472430.asia-south1.run.app (`allUsers` + `roles/run.invoker`). Lightweight in-app quotas and release steps: [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md).

## Related

- [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) — **current** public demo + manual release
- [`cloud_run_deployment.md`](cloud_run_deployment.md) — first-deploy record (IAM, initial validation)
- [`docker.md`](docker.md) — image contents and GHCR
- [`architecture.md`](architecture.md) — laptop → Actions → GHCR → cloud
- [`api_contract.md`](api_contract.md) — `/health`, `/ready`, `/query`


## Lightweight request protection for public demo

The API includes a process-local query quota and input bounds:

- Maximum request body: 16 KiB.
- Maximum question: 2,000 characters.
- `POST /query` and `POST /query/stream`: 10 requests per 60 seconds per running instance; excess requests receive HTTP 429 and `Retry-After`.
- `/health` and `/ready` are excluded from the query quota.

This is a simple guard for the single-instance portfolio demo, not a distributed or per-client rate limiter. It resets on instance restart. Keep Cloud Run `--max=1` and `--concurrency=1`. Add Cloud Armor / a shared rate-limit store if the service later attracts sustained public traffic.

### Enable unauthenticated invocation

Deploy the request-protection image and test HTTP 413, 422 and 429 while the service is still private. Then enable public invocation:

```bash
gcloud run services add-iam-policy-binding fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag \
  --member=allUsers \
  --role=roles/run.invoker
```

To revoke public invocation:

```bash
gcloud run services remove-iam-policy-binding fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag \
  --member=allUsers \
  --role=roles/run.invoker
```

Do not enable `allUsers` until the protected revision is deployed and its limit tests pass.
