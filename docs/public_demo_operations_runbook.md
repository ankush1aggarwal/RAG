# FiQA RAG Public Demo — Operations and Release Runbook

This document records the currently deployed public demo, its lightweight request controls, the GitHub Actions release process, and the manual Cloud Run deployment procedure.

## 1. Current deployment

| Item | Current value |
|---|---|
| Cloud Run service | `fiqa-rag-api` |
| Project | `fiqa-production-rag` |
| Region | `asia-south1` (Mumbai) |
| Public URL | https://fiqa-rag-api-729436472430.asia-south1.run.app |
| Container registry | `ghcr.io/ankush1aggarwal/fiqa-production-rag` |
| **Deployed image tag (GHCR)** | **`4f5ddf3`** (commit on `main`; prefer this SHA over `latest` for releases) |
| Runtime service account | `fiqa-rag-cloud-run@fiqa-production-rag.iam.gserviceaccount.com` |
| Vector database | Qdrant Cloud |
| LLM provider | Cohere |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Collection | `fiqa_cs200_bge_base_en_v1_5` |
| Retrieval profile | Dense, top-k 10, no reranker |
| Public invocation | Enabled through `allUsers` with `roles/run.invoker` |

The service was deployed in Cloud Run and subsequently validated through the public API. The public service URL is a production-like portfolio demo endpoint; it is not intended to provide a service-level availability guarantee.

## 2. Cloud Run runtime configuration

The intended deployment settings are:

| Setting | Value |
|---|---:|
| CPU | 2 vCPU |
| Memory | 4 GiB |
| Request concurrency | 1 |
| Minimum instances | 0 |
| Maximum instances | 1 |
| Request timeout | 300 seconds |
| Container port | 8080 |
| Billing | Request-based |

The application-level `QUERY_TIMEOUT_SECONDS` defaults to 120 seconds. Cloud Run's request timeout is longer so the platform does not terminate a request before the application timeout is reached.

Confirm the live service configuration rather than assuming it has not changed:

```bash
gcloud run services describe fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag \
  --format=export
```

## 3. Application request protection

The public demo has deliberately lightweight, process-local controls:

- Maximum request body: **16 KiB**. Oversized payloads return HTTP `413` (`payload_too_large`).
- Maximum question length: **2,000 characters**. Overlong questions return HTTP `422` (`validation_error`).
- `POST /query` and `POST /query/stream`: **10 requests per 60 seconds per running application process**. Excess requests return HTTP `429` (`rate_limit_exceeded`) with a `Retry-After` header.
- `GET /health` and `GET /ready` are not subject to the query quota.

Important limitations:

- The quota is **global to the running process**. It is not per user, API key, or source IP; all visitors share it.
- The quota is stored in process memory and resets when the Cloud Run instance restarts or is replaced.
- It is a basic guard for a low-traffic portfolio demo, not distributed abuse protection.
- Keep the Cloud Run maximum instance count at one while relying on this limiter. If the service is scaled out later, each process would have a separate quota.
- Cloud Armor or a shared rate-limit store can be considered if the service receives sustained public traffic.

## 4. API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/ready` | Qdrant, manifest alignment and LLM configuration check |
| POST | `/query` | RAG answer with citations |
| POST | `/query/stream` | Server-Sent Events response |
| GET | `/docs` | FastAPI OpenAPI documentation |

A basic public health check:

```bash
SERVICE_URL="https://fiqa-rag-api-729436472430.asia-south1.run.app"
curl -i "$SERVICE_URL/health"
curl -i "$SERVICE_URL/ready"
```

A public query example:

```bash
curl -i -X POST "$SERVICE_URL/query" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: public-demo-manual-test" \
  -d '{"question":"What is APR on a credit card?"}'
```

A successful query should return an answer, citations, and retrieval metadata. The production retrieval profile is frozen to dense retrieval with `top_k=10` and no reranking.

## 5. Validation history and initial resource baseline

The initial hosted validation returned HTTP 200 for all three checks:

- `/health`: `200`
- `/ready`: `200`, with `qdrant_collection=true`, `retrieval_index_aligned=true`, `llm_provider=cohere`, `llm_ready=true`. `bm25_index=false` is expected for the frozen dense-only API profile.
- `/query`: `200`, answer generated successfully with 10 citations.

Initial Cloud Run metrics reported during the small controlled assessment:

| Metric | Reported result |
|---|---:|
| Request latency | P50 0.45 s; P95 2.0 s |
| Container startup latency | P50 1.02 min; P95 1.07 min |
| Container CPU utilization | P50 0.5%; P95 0.95% |
| Container memory utilization | P50 30.5%; P95 31% |
| Instance count | 1 |
| Maximum concurrent requests metric | P50 1.5; P95 1.99 |

These are the values observed in the initial dashboard review, not a formal capacity benchmark. Cloud Run request latency and container startup latency are different measurements; the roughly one-minute startup is a cold-start characteristic. The reported concurrency metric should be rechecked against the live service's configured concurrency before changing that setting.

The initial sample was intentionally small and sequential. Do not treat the P95 readings as statistically stable under significant traffic.

## 6. What happens when code is merged into `main`

The current repository has two GitHub Actions workflows:

### CI workflow — `.github/workflows/ci.yml`

Runs on pull requests targeting `main`, pushes to `main`, and manual workflow dispatch. It:

1. Checks out the code.
2. Sets up Python 3.12.
3. Installs project and development dependencies.
4. Runs Ruff lint checks.
5. Runs the pytest suite.

### Container workflow — `.github/workflows/container.yml`

Runs on pull requests targeting `main`, pushes to `main`, version tags (`v*`), and manual workflow dispatch. It:

1. Builds the Docker image.
2. Runs the container smoke test against `/health`.
3. On a push to `main` or a version tag, logs in to GHCR and publishes the image.
4. Publishes a `latest` tag for the default branch and a commit-SHA tag for traceability.

For pull requests, the container is built and smoke-tested but is not pushed to GHCR.

### Important: Cloud Run deployment is currently manual

**Merging code into `main` does not automatically deploy it to Cloud Run.**

After a merge:

- CI runs lint and tests.
- The Container workflow builds and smoke-tests the Docker image.
- If successful, a new image is pushed to GHCR (including `latest` and the commit-SHA tag).
- The currently running Cloud Run revision continues serving its already deployed image until a new `gcloud run deploy` is executed.

Cloud Run does not continuously watch the mutable GHCR `latest` tag. A new image in GHCR does not itself change the running Cloud Run service.

## 7. Manual release procedure after a merge to `main`

### Initial deploy or env/secret change (full command)

Use when creating the service or changing non-image settings. Replace **`https://<YOUR_QDRANT_CLUSTER_HOST>:6333`** with your Qdrant Cloud URL.

```bash
gcloud run deploy fiqa-rag-api \
  --project=fiqa-production-rag \
  --image=ghcr.io/ankush1aggarwal/fiqa-production-rag:latest \
  --region=asia-south1 \
  --port=8080 \
  --cpu=2 \
  --memory=4Gi \
  --concurrency=1 \
  --min=0 \
  --max=1 \
  --timeout=300 \
  --service-account=fiqa-rag-cloud-run@fiqa-production-rag.iam.gserviceaccount.com \
  --set-env-vars="LLM_PROVIDER=cohere,QDRANT_URL=https://<YOUR_QDRANT_CLUSTER_HOST>:6333,COLLECTION_NAME=fiqa_cs200_bge_base_en_v1_5,API_ENFORCE_FROZEN_RETRIEVAL=true,BM25_INDEX_DIR=deploy/index_metadata" \
  --set-secrets="COHERE_API_KEY=cohere-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest" \
  --no-allow-unauthenticated
```

After deploy, grant public invoke if needed (§8). Historical first deploy used `--no-allow-unauthenticated`; the demo is public via IAM `allUsers` + `roles/run.invoker`.

For **image-only** releases on an existing service, follow the steps below (prefer a **commit-SHA** image tag over `:latest`).

### Step 1 — Wait for GitHub Actions

Open the repository's Actions tab and confirm that both **CI** and **Container** workflows for the merge commit have succeeded.

Do not deploy an image from a failed or incomplete workflow.

### Step 2 — Identify the merge commit's image tag

The Container workflow publishes a commit-SHA tag. Use that immutable commit-specific tag rather than relying on `latest` for a controlled release.

From a local clone:

```bash
git fetch origin main
IMAGE_TAG=$(git rev-parse --short=7 origin/main)
echo "$IMAGE_TAG"
```

Confirm that the corresponding image tag exists in the GHCR package before deploying it:

```text
ghcr.io/ankush1aggarwal/fiqa-production-rag:<IMAGE_TAG>
```

### Step 3 — Deploy the new image to Cloud Run (routine release)

Run from Google Cloud Shell. This updates the **container image** while retaining existing service-level environment variables and secrets (no `--set-env-vars` / `--set-secrets` unless you intend to change them):

```bash
IMAGE_TAG="<7-character-merge-commit-sha>"

gcloud run deploy fiqa-rag-api \
  --project=fiqa-production-rag \
  --region=asia-south1 \
  --image="ghcr.io/ankush1aggarwal/fiqa-production-rag:${IMAGE_TAG}" \
  --port=8080 \
  --cpu=2 \
  --memory=4Gi \
  --concurrency=1 \
  --min=0 \
  --max=1 \
  --timeout=300 \
  --service-account=fiqa-rag-cloud-run@fiqa-production-rag.iam.gserviceaccount.com
```

Do not add `--no-allow-unauthenticated` or `--allow-unauthenticated` casually during routine image releases. Public invocation is configured separately through the Cloud Run service IAM policy (§8); verify that policy after deploy.

**Full command** (initial service + env + secrets): see [§7 — Initial deploy](#initial-deploy-or-envsecret-change-full-command) above or [`cloud_run_deployment.md` §5.2](../cloud_run_deployment.md#52-full-deployment-command).

### Step 4 — Verify the new revision

```bash
gcloud run services describe fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag \
  --format="value(status.latestReadyRevisionName,status.url)"
```

Then run the public `/health`, `/ready`, and one `/query` smoke test. Check Cloud Run logs for startup or request errors.

## 8. Check public invocation

The service is intended to be publicly invokable. Verify its IAM policy with:

```bash
gcloud run services get-iam-policy fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag
```

The service-level policy should include:

```yaml
- members:
  - allUsers
  role: roles/run.invoker
```

To revoke public invocation:

```bash
gcloud run services remove-iam-policy-binding fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag \
  --member=allUsers \
  --role=roles/run.invoker
```

To restore public invocation:

```bash
gcloud run services add-iam-policy-binding fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag \
  --member=allUsers \
  --role=roles/run.invoker
```

## 9. Rollback procedure

Cloud Run retains prior revisions. If a new revision has a runtime regression, first list revisions:

```bash
gcloud run revisions list \
  --service=fiqa-rag-api \
  --region=asia-south1 \
  --project=fiqa-production-rag
```

Then direct all traffic to the previous known-good revision (replace the placeholder with its exact revision name):

```bash
gcloud run services update-traffic fiqa-rag-api \
  --to-revisions=PREVIOUS_GOOD_REVISION=100 \
  --region=asia-south1 \
  --project=fiqa-production-rag
```

After rollback, recheck `/health`, `/ready`, and a single `/query` request.

## 10. Operational and cost notes

- Public `/query` calls can incur Cohere usage charges. The application quota is a coarse demo safeguard, not a billing cap.
- Cloud Run minimum instances are configured as zero, so cold starts are expected after idle periods.
- Maximum instances is one to constrain scaling and keep the process-local limiter coherent.
- Review Cloud Run request count, request latency, startup latency, CPU, memory, instance count, and logs periodically.
- Review Cohere and Qdrant usage separately; they may bill independently of Cloud Run.
- Never commit `COHERE_API_KEY`, `QDRANT_API_KEY`, or other secret values. They should remain in Secret Manager.

## 11. Current automation boundary and future option

Current automation ends at **image publication to GHCR**. Cloud Run deployment is a deliberate manual step, allowing the image and tests to be checked before production changes.

A future enhancement could add a dedicated GitHub Actions deployment workflow using Google Workload Identity Federation. **Parked** — Cloud Run release stays **manual** by design for this portfolio demo.

## 12. Related project documentation

| Document | Role |
|----------|------|
| **This file** | Public demo URL, quotas, manual release, rollback, CI automation boundary |
| [`cloud_run.md`](cloud_run.md) | Qdrant migration, env vars, app/image prep |
| [`cloud_run_deployment.md`](cloud_run_deployment.md) | First Cloud Run deploy (IAM, Secret Manager, initial private validation) |
| [`docker.md`](docker.md) | Dockerfile, GHCR workflow |
| [`architecture.md`](architecture.md) | Dev → CI → GHCR → manual Cloud Run |
| [`api_contract.md`](api_contract.md) | HTTP contract and error shapes |
| [`week9_sprint.md`](week9_sprint.md) | Week 9 closure checklist (complete) |
| [`PROJECT_CLOSURE.md`](PROJECT_CLOSURE.md) | Portfolio closure — DS + engineering narrative |
| [`RAG_PROJECT_ROADMAP.md`](RAG_PROJECT_ROADMAP.md) | Optional final polish checklist |
