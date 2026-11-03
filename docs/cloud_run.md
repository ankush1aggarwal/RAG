# Google Cloud Run deployment (Week 9)

Deploy the **GHCR API image** on **Cloud Run** with **Qdrant Cloud** for vectors. Local development keeps using `QDRANT_PATH`; production uses `QDRANT_URL`.

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

### Ingest into Qdrant Cloud

From your laptop (or a Cloud Run Job later), with the same env as production:

```bash
# .env: QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME=...
python src/scripts/ingest.py --full-corpus --collection fiqa_cs200_bge_base_en_v1_5
```

BM25 pickles still land under `data/bm25_indexes/` locally unless you add object storage in a later step. Hybrid `/ready` on Cloud Run requires a BM25 artifact strategy (volume, GCS, or dense-only demo).

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

## Cloud Run service (outline)

1. **Image:** `ghcr.io/<owner>/fiqa-production-rag:latest` (or mirror to Artifact Registry).
2. **Port:** Cloud Run sets **`PORT`** (typically `8080`). The container image defaults `PORT=8080`; `serve_api.py` binds to `PORT` unless you pass `--port`. Local dev without `PORT` uses **8000**.
3. **Secrets / env:** `QDRANT_URL`, `QDRANT_API_KEY`, **`LLM_PROVIDER=cohere`**, **`COHERE_API_KEY`**, `COLLECTION_NAME`, `API_ENFORCE_FROZEN_RETRIEVAL=true`, etc. — see [`.env.example`](../.env.example).
4. **Memory / CPU:** sized for loading baked BGE from disk (not Hub download on cold start).
5. **Health checks:** liveness → `GET /health`; readiness → `GET /ready` once Qdrant + LLM are configured.

## Related

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
