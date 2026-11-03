# Production-Grade RAG — Project Roadmap & Progress Tracker

**Repository:** https://github.com/ankush1aggarwal/RAG  
**Planning baseline:** 19 October 2026  
**Status convention:** ✅ Done · 🟡 Assumed done / verify evidence · ⬜ Remaining · ⭐ Optional/stretch

> **Production vs experiments:** The served application uses **one frozen retrieval design** (quality + latency). All other retrieval/rerank/chunking options remain available via **CLI and evaluation scripts** for Week 5+ experiments. See [`docs/reports/rag_experiment_program_report.md`](reports/rag_experiment_program_report.md).

## Executive summary

The intended outcome is a portfolio-quality RAG application for FiQA financial-information questions that demonstrates retrieval, reranking, measurable quality, generation, API design, tests, operational reliability, deployment, and technical communication. Build this in small, reviewable increments. Keep `src/app/` canonical and notebooks experimental.

| Phase | Status | Goal |
|---|---|---|
| Week 0 — Foundation | ✅ Done | Repository, environment, package structure, settings and project docs |
| Week 1 — Ingestion + dense retrieval | ✅ Done | FiQA chunks, embeddings, Qdrant, dense search and CLI |
| Week 2 — Generation + API | ✅ Done | Grounded answer generation, citations, FastAPI health/readiness/query |
| Week 3 — Hybrid retrieval | ✅ Done | BM25, dense + BM25 RRF, CLI/API and timing/overlap report |
| Week 4 — Reranking + relevance evaluation | ✅ Done | Qrels, IR metrics, reranker, experiment program 1–5, error analysis |
| Week 5 — **Experimentation sprint** | ✅ Done | E3–E9 complete; frozen retrieval: **200w / BGE / dense** (see lessons report) |
| Week 6 — Production API engineering | ✅ Complete | Frozen profile, errors, streaming, integration tests, docs |
| Week 7 — Observability | ✅ | Structured JSON logs, stage timings, correlation ID, token/cost when available |
| Week 8 — Container + CI | ✅ | GHCR, Actions CI + container smoke, [`architecture.md`](architecture.md) |
| Week 9 — Deployment + public demo | ✅ **Done** | Public Cloud Run + Qdrant Cloud; API-only demo ([`public_demo_operations_runbook.md`](public_demo_operations_runbook.md)); image **`4f5ddf3`** |
| Final polish — Portfolio | 🟡 **In progress** | README demo screenshots ✅; blog/LinkedIn/repro pass still open |
| Phase 2 — Agentic RAG | ⬜ Future project | Multi-step retrieval/tools/evidence checks built on the proven RAG baseline |

---

## Completed foundation: Weeks 0–3

### Week 0 — Foundation ✅
- [x] Public GitHub repository and commit/PR workflow established.
- [x] Python package layout (`src/app`, scripts, tests), dependency files and settings.
- [x] `.env` secret pattern and `.gitignore`; local Qdrant/BM25 artifacts are not committed.
- [x] `README.md` and `PROJECT_SPEC.md` established as living documentation.

### Week 1 — Ingestion and dense retrieval ✅
- [x] FiQA corpus loader and document normalization.
- [x] Chunking with stable source document IDs and chunk IDs.
- [x] Embedding provider abstraction and dimension validation.
- [x] Qdrant local persistence, collection creation/recreation and upsert.
- [x] Dense retrieval and CLI query path.

### Week 2 — Generation and API ✅
- [x] Ollama and Cohere LLM provider abstractions.
- [x] Grounding-focused prompt builder and answer pipeline.
- [x] Source citations returned from retrieved chunks.
- [x] FastAPI `/health`, `/ready`, and `POST /query`.
- [x] Request schemas and baseline unit tests.

### Week 3 — Hybrid retrieval ✅
- [x] BM25 lexical index with common tokenizer and no-overlap filtering.
- [x] Dense + BM25 hybrid retrieval using RRF.
- [x] Dense/BM25/hybrid options in CLI and API.
- [x] Fixed query-set benchmark, latency protocol and comparison report.
- [x] BM25 overlap precomputation microbenchmark (clearly labelled as an isolated stage, not end-to-end speedup).
- [x] Manifest/runtime configuration checks and BM25/manifest generation-ID alignment.
- [x] BM25 index pickle removed from Git and documented as locally generated/trusted-only.

**Week 3 operational caveat to retain:** for local persistent Qdrant, stop the API before running a destructive re-ingest. Before concurrent or cloud rebuilds, strengthen collection-generation validation/swap behavior.

---

## Week 4 — Reranking and labeled evaluation ✅

- [x] FiQA qrels loader, metrics (NDCG/MRR/MAP/P/R), eval runners and full-corpus experiments 1–5.
- [x] Reranker (local CE + optional Cohere); API/CLI optional rerank.
- [x] Master summaries: [`docs/reports/week4_retrieval_program_report.md`](reports/week4_retrieval_program_report.md), [`docs/reports/rag_experiment_program_report.md`](reports/rag_experiment_program_report.md).

**Outcome:** **Dense retrieval without rerank** selected as the initial **production profile** (best test NDCG@10 + latency). Other methods remain for experiments only.

**Week 4 acceptance rule:** reranking did **not** improve dense NDCG@10 on test; BM25 benefited from CE but stays below dense.

---

## Remaining implementation plan — work in this order

## Experiment program — run alongside the roadmap, not after it

The project should demonstrate not only a working RAG system, but also disciplined experiments with hypotheses, controls, metrics and decisions. Keep experiments in [`docs/reports/experiments/README.md`](reports/experiments/README.md) (one sheet per experiment) plus the consolidated [`docs/reports/rag_experiment_results.md`](reports/rag_experiment_results.md) from [`RAG_EXPERIMENT_REPORT_TEMPLATE.md`](RAG_EXPERIMENT_REPORT_TEMPLATE.md). Keep implementation in `src/app/`; use notebooks for exploration only.

### Experiment status and schedule

| ID | Experiment | Timing | Priority | State / deliverable |
|---|---|---|---|---|
| E0 | Dense vs BM25 vs hybrid RRF latency and result overlap | Week 3 | Core | ✅ Existing `docs/reports/week3_retrieval_comparison.md`; unlabeled quality proxy only |
| E1 | Dense vs BM25 vs hybrid using FiQA qrels | Week 4 | Core | ✅ [`week4_exp2`](reports/week4_exp2_full_corpus_validation.md), Exp 4 test |
| E2 | Methods + **local CE rerank** | Week 4 | Core | ✅ [`week4_exp5`](reports/week4_exp5_test_rerank.md) + error analysis |
| E3 | **Candidate diversity / duplicate-chunk crowding** | **Week 5** | **High** | ✅ Complete — control pool frozen |
| E5 | Reranker candidate depth (10 / 20 / 30 / 50) | Week 5 | Core | **Skipped** — E2 rejected CE; no new reranker |
| E6 | Chunk size sensitivity | Week 5 | High | ✅ 200w adopted |
| E7 | Chunk overlap sensitivity | Week 5 | High | ✅ 20% @ 200w |
| E8 | Embedding model comparison | Week 5 | High | ✅ BGE-base |
| E9 | **Paragraph-aware chunking** (vs E8 BGE 200w) | Week 5 | Medium | ✅ Complete (tie → keep E8) |
| E10 | Generation + citation grounding | Week 6–7 | Core | ⭐ **Parked** — out of scope for portfolio demo (no citation rubric eval) |
| E11 | E2E performance / concurrency | Week 8–9 | Core | ⬜ |

### New priority experiment — E3: duplicate-chunk crowding / candidate diversity

**Why this is separate from the planned candidate-depth sweep:** candidate depth asks how many candidates to score; E3 asks how to spend that fixed reranker budget across documents. The Week 4 findings suggest candidate coverage is already strong (no `missing_candidates` cases), while multiple chunks from the same `doc_id` can consume slots in a document-level evaluation. This mechanism is not fully tested by varying depth alone.

**Hypothesis:** controlling the number of chunks contributed by any one document will reduce redundant candidates and can improve document-level NDCG@10 / MRR@10 without hurting candidate Recall@50. This is a hypothesis to test, not an assumed outcome.

**Keep fixed:** corpus/index snapshot, query split, dense retriever, reranker model, scoring settings and final document-level metric implementation. Use development queries for policy selection.

**Compare these arms:**

1. **Control — current:** rerank the current top 50 dense chunks, with no per-document cap.
2. **Diagnostic — deduplicate top 50:** from those same 50 chunks, keep only the highest dense-ranked chunk per `doc_id`, then rerank the remaining unique chunks. This directly shows what duplicates are contributing, although the reranker receives fewer candidates.
3. **Treatment — overfetch + cap 1:** retrieve a deeper ordered pool (start with top 100 or 150 chunks), scan in rank order and take up to 50 candidates with at most one chunk per `doc_id`.
4. **Treatment — overfetch + cap 2:** use the same deeper pool and budget of 50 candidates, but allow at most two chunks per `doc_id`.

If the deeper pool cannot fill the intended budget, record the actual candidate count rather than silently comparing unequal pools. Preserve original chunk IDs/text and score candidates at the chunk level; aggregate/deduplicate results by source `doc_id` for qrels metrics. Avoid assuming one chunk always represents a document's best evidence; the cap-2 arm tests that trade-off.

**Report:** unique documents and duplicate-slot fraction entering the reranker; relevant-document candidate Recall@50; NDCG@10, MRR@10 and Recall@10 after reranking; reranking latency; per-query gains/losses and whether lost queries reflect candidate coverage or ordering. Include both the no-backfill diagnostic and overfetch treatments so the report distinguishes “removing duplicates” from “replacing duplicate slots with candidates from deeper ranks.”

**Decision rule:** select the policy on development data using quality and latency together. Do not pick a cap just because it increases the number of unique documents; confirm that ranked relevance improves and that no material regression is hidden by the aggregate. Save the chosen policy, exact fetch depth, cap, metrics and commit in the experiment log. Use the frozen pool for any **local** rerank eval (E2/E5); production remains dense-only unless a reranker beats dense on **test**.

### How to run each experiment

For every run, write down:

1. **Question / hypothesis:** what change do we expect, and why?
2. **Control:** the known-good baseline and exact configuration.
3. **One main variable:** avoid changing chunking, model and fusion settings together.
4. **Fixed evaluation:** same qrels split, query set, corpus snapshot, cutoff and metric implementation for both control and treatment.
5. **Results:** aggregate metrics plus per-query changes, p50/p95 latency, index/build cost where relevant, and run timestamp/commit.
6. **Decision:** adopt, reject or defer the change, with a short rationale and limitations.

### Recommended execution order after Week 4

Run the retrieval experiment sprint in **Week 5** (dedicated experimentation week), **before Week 6+ production API and deployment**:

1. E1/E2 baselines are saved in [`rag_experiment_program_report.md`](reports/rag_experiment_program_report.md).
2. Run E3 candidate-diversity / duplicate-crowding ablation first (Week 4 hurt cases). ✅
3. **Skip E5** depth sweep unless a **local** reranker beats dense on test (E2 already rejected CE @50).
4. Run a small chunk-size sweep (**E6**), then overlap sweep (**E7**) using the best-supported chunk size.
5. Compare one alternative embedding model (**E8**) only if compute/time budget allows. Rebuild each index with the selected model; never compare models against mismatched embeddings/indexes.
6. **E9** — paragraph-aware chunking on BGE (vs E8 fixed 200w). Hybrid/RRF sweeps remain optional/deferred (E1).
7. **Freeze** the production retrieval profile; document in the experiment report. **Week 6** finishes API hardening (streaming, tests); the served pipeline is **already locked** to that profile when enforcement is on (other methods remain CLI/eval-only).
8. **E10** parked for this portfolio (retrieval frozen; production uses live Cohere + structural citations). **E11** uses observability from Week 7 where possible.

For a constrained schedule, the minimum valuable additional experiments are **E3 candidate diversity (done), E6 chunk size, and E8 one alternate embedding model**. **E5** is skipped while rerank stays off; **E7** and **E9** are follow-up/stretch. Do not run a full Cartesian grid across every setting; it multiplies cost and makes the source of improvement hard to explain.

### Data and interpretation guardrails

- Use development queries for tuning. Keep the final test split untouched until the selected configuration is ready for a final evaluation.
- Report corpus coverage: qrels may reference documents missing from a 1,000-document sample. A reranker cannot retrieve relevant documents that are absent from its candidate pool.
- Evaluate at the same relevance unit as the qrels (FiQA judgments are document-level); do not count overlapping chunks as independent relevant documents.
- Do not claim an improvement from latency alone or from Jaccard overlap alone. Show the quality/latency trade-off and per-query cases.
- Record a negative or neutral result too; this is useful evidence of experimental discipline.

**Experiment-program definition of done:** the report shows a stable labeled baseline; at least three controlled retrieval experiments have reproducible results; each has an explicit decision; and the chosen configuration is frozen before deployment work. The generation experiment and load/performance test are additional later milestones.

---

## Week 5 — Experimentation sprint ✅

**Goal:** Run E3–E9 on full FiQA; adopt/reject with qrels + latency — **done**.

**Frozen retrieval (eval + served API):** dense, no rerank, **`top_k=10`**, **200w / 20% overlap**, **`BAAI/bge-base-en-v1.5`**, index **`fiqa_cs200_bge_base_en_v1_5`** (test NDCG@10 **0.394**). Enforced when `API_ENFORCE_FROZEN_RETRIEVAL=true` (default).

**Reports:** [`rag_experiment_results.md`](reports/rag_experiment_results.md) · [`rag_retrieval_lessons.md`](reports/rag_retrieval_lessons.md) · [`rag_experiment_program_report.md`](reports/rag_experiment_program_report.md).

**Indexes on disk:** `fiqa` + `fiqa_cs200_bge_base_en_v1_5` only (`src/scripts/cleanup_local_indexes.py`).

---

## Week 6 — Production API engineering ✅

**Goal:** Harden the API and **serve only the frozen production retrieval design** (experimental methods remain CLI/eval-only when enforcement is on).

**Delivered:**

1. ✅ API contract, `ErrorResponse`, exception handlers (`docs/api_contract.md`)
2. ✅ `POST /query/stream` (SSE); incremental tokens; pre-stream **422 JSON** for config/validation errors
3. ✅ `QUERY_TIMEOUT_SECONDS`, cooperative cancel (`QueryScope`), Ollama stream abort on timeout
4. ✅ Lifespan `close_resources`; smoke script with `X-Request-ID` and `--stream`
5. ✅ Single-pass `/ready` diagnostics; field docs in README + contract
6. ✅ `src/tests/integration/` + **251** tests under `src/tests/`
7. ✅ OpenAPI examples and route `responses`
8. ✅ Frozen profile enforced in settings + `/query` (`production_retrieval.py`)

**Definition of done:** met for code/tests/docs; **live smoke + PR merge** remain operator steps ([`docs/week6_sprint.md`](week6_sprint.md) Block 0).

## Week 7 — Observability ✅

**Goal:** Explain what a query did and where latency/errors/cost came from without exposing secrets or unnecessarily logging user data.

1. ✅ **Structured JSON logs to stderr** (stdlib only — no OpenTelemetry or Langfuse).
2. ✅ Propagate **`X-Request-ID`** into RAG stage logs (middleware + thread pool).
3. ✅ Log method, candidate counts, reranker use, **timings by stage**, errors and provider/model IDs. Avoid logging API keys or full prompts/context by default.
4. ✅ Token usage on `llm_usage` / `query_complete` when Ollama or Cohere returns counts; optional **`estimated_cost_usd`** via Cohere pricing env vars. p50/p95 in log aggregator — **Week 8+** deployment.
5. ✅ Tests for log fields/redaction + integration caplog on `/query`; [`docs/observability_guide.md`](observability_guide.md).
6. Eval reports can split latency using `duration_ms` on `stage_*` events.

**Deliverables:** `src/app/observability/`, `src/app/generation/usage.py`, config switches, tests, operational guide.  
**Definition of done:** met — sample request traceable through main stages; latency/failure observable without leaking secrets.

## Week 8 — Docker + CI ✅

**Goal:** Recreate the runtime in a clean environment and automatically reject regressions.

1. ✅ Dockerfile, `.dockerignore`, optional Compose; data/models via volumes/env (not baked in).
2. ✅ GitHub Actions **CI** ([`ci.yml`](../.github/workflows/ci.yml): Ruff + pytest) and **Container** (build + `/health` smoke + GHCR).
3. ✅ [`docs/docker.md`](docker.md), [`docs/architecture.md`](architecture.md), [`week8_sprint.md`](week8_sprint.md).
4. ✅ `requirements-git.txt` pinned to commit SHA.

**Definition of done:** met.

## Week 9 — Cloud deployment + demo ✅ **(complete)**

**Goal:** Give a reviewer a repeatable way to try the system without running your laptop.

1. ✅ Qdrant Cloud + Secret Manager — [`cloud_run.md`](cloud_run.md#migrate-local-collection--qdrant-cloud).
2. ✅ Repeatable ingest; manifest in `deploy/index_metadata/`.
3. ✅ Cloud Run deployed (GHCR image, env/secrets) — first deploy: [`cloud_run_deployment.md`](cloud_run_deployment.md); **current image:** `4f5ddf3`.
4. ✅ Frozen dense `/ready` on hosted service.
5. ✅ Public hosted smoke: `/health`, `/ready`, `/query` — [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) §4–§5.
6. ✅ **Public API** (`allUsers` invoker) with lightweight in-app request limits — runbook §3.
7. ✅ Demo evidence: README screenshots (`docs/live_fiqa_rag_api.png`, `docs/live_api_query_repsonse.png`). Initial cost/latency notes in runbook §5 (formal load testing **parked** — portfolio demo app).

**Out of scope (explicit):** separate demo UI (`frontend/`), **E10** citation evaluation, automated Cloud Run deploy (WIF **parked**), production-scale traffic testing.

**Release process:** merge to `main` → Actions build/push **GHCR only** → **manual** `gcloud run deploy` with commit SHA tag (runbook §6–§7).

**Deliverables:** ✅ public URL + ops runbook; ✅ API curl examples + README screenshots; ✅ manual release documented.  
**Definition of done:** met — anonymous reviewers can call the public API and get citations; secrets not in git.

## Final polish — portfolio deliverables 🟡

1. 🟡 Update README with architecture, setup, API examples, evaluation tables and links to reports (live demo section + screenshots added).
2. ✅ Engineering topology — [`architecture.md`](architecture.md) (+ [`architecture-diagram.png`](architecture-diagram.png) in repo).
3. ✅ Public demo screenshots in README (`docs/live_fiqa_rag_api.png`, `docs/live_api_query_repsonse.png`).
4. ⬜ Publish a technical blog that explains the engineering decisions and measured trade-offs, not just a tutorial-style list of features.
5. ⬜ Publish a LinkedIn summary that links to the blog and repo, includes one meaningful result, and makes limitations clear.
6. ✅ Agentic RAG direction documented under Phase 2 (below) — not implemented.
7. ⬜ Final reproducibility pass from a clean environment (links, tests, CI, public API smoke).

**Deliverables:** portfolio-ready README, architecture diagram, public demo evidence, evaluation report, blog and LinkedIn post.  
**Definition of done:** an interviewer can understand, run or demo the project, see measured evidence and follow the main design choices without needing your verbal explanation.

---

## Phase 2 — Agentic RAG (separate follow-on; start only after production RAG is stable)

Reuse the existing ingestion, Qdrant, BM25, RRF, reranker, evaluation, API and observability rather than rebuilding them.

1. Establish a baseline agentic use case, e.g. a multi-part financial question requiring evidence from multiple FiQA documents.
2. Add query decomposition or a planner that creates bounded sub-questions.
3. Implement controlled tools: retrieve from the current RAG system first; add a calculation tool only for appropriate arithmetic tasks. Add web search only if the problem genuinely needs external/current information.
4. Collect evidence per sub-question and preserve source IDs through the final answer.
5. Add evidence verification / answer-grounding checks and explicit fallback when retrieval cannot support a claim.
6. Use a graph framework such as LangGraph only if state, branching and retries justify the complexity; begin with the simplest explicit orchestration that is testable.
7. Evaluate the agent against a fixed task set: answer correctness/grounding, retrieval recall, tool-call success, latency and cost. Compare with the non-agentic RAG baseline.
8. Add tracing for tool calls, intermediate steps and failures. Make iteration limits/timeouts explicit to avoid runaway loops.

**Future deliverable:** a separate Agentic RAG case study showing what multi-step orchestration improves compared with the current RAG baseline and what extra latency/cost it introduces.

---

## Work tracking checklist

Use this as the live checklist in commits/PR descriptions. Complete the current step before starting the next unless there is a clear dependency reason to parallelize.

### Immediate next action — Final polish
- [x] Week 9 complete — [`week9_sprint.md`](week9_sprint.md), [`public_demo_operations_runbook.md`](public_demo_operations_runbook.md) (deployed image **`4f5ddf3`**).
- [ ] Blog + LinkedIn write-up.
- [ ] Clean-environment reproduction pass + public API smoke from docs.

### Week 6 (API)
- [x] Block 0 baseline (`docs/week6_sprint.md`, `smoke_api.py`)
- [x] API contract/error handling (`docs/api_contract.md`, `api/errors.py`)
- [x] Request ID middleware (`X-Request-ID`)
- [x] Query wall-clock + upstream timeouts → `504`
- [x] `POST /query/stream` (SSE)
- [x] Readiness docs + single-pass diagnostics
- [x] API integration tests (`src/tests/integration/`)
- [x] API docs/examples + OpenAPI polish
- [x] Lifecycle smoke (`close_resources`, `smoke_api.py --stream`)

### Week 7
- [x] Structured JSON logs (`src/app/observability/`, `docs/observability_guide.md`)
- [x] Correlation ID in pipeline + stage latency events
- [x] Token/cost accounting when provider returns usage (`src/app/generation/usage.py`)
- [x] Redaction tests + log-based integration test on `/query`

### Week 8
- [x] Dockerfile and `.dockerignore`
- [x] Optional `docker-compose.yml` + [`docs/docker.md`](docker.md)
- [x] Actions: build + container smoke + GHCR push (`.github/workflows/container.yml`)
- [x] GitHub Actions Ruff + pytest (`.github/workflows/ci.yml`)
- [x] Pinned `requirements-git.txt` + [`docs/architecture.md`](architecture.md)

### Week 9 ✅
- [x] Qdrant Cloud provisioned and secured (API key in Secret Manager)
- [x] Deployment configuration and persistent-index strategy (Qdrant Cloud + manifest in image)
- [x] Hosted API smoke tests ([`public_demo_operations_runbook.md`](public_demo_operations_runbook.md))
- [x] Public demo API (`allUsers` invoker + in-app limits — runbook §3); **no separate UI** (OpenAPI `/docs` + curl)
- [x] Deployment / release runbook; manual Cloud Run only (WIF automation **parked**)
- [x] README demo screenshots

### Portfolio
- [x] Engineering topology ([`architecture.md`](architecture.md), [`architecture-diagram.png`](architecture-diagram.png))
- [x] Evaluation report + error analysis ([`docs/reports/rag_experiment_results.md`](reports/rag_experiment_results.md))
- [x] Live demo screenshots in README
- [ ] README final polish pass (ongoing)
- [ ] Blog published
- [ ] LinkedIn post published
- [ ] Clean-environment reproduction pass
- [x] Agentic RAG direction (Phase 2 — not implemented)

### Agentic RAG — only after the above is stable
- [ ] Define an evaluation task set and non-agentic baseline
- [ ] Query decomposition / planning
- [ ] Bounded tools and evidence collection
- [ ] Evidence verification and fallback
- [ ] Agentic evaluation, tracing and cost/latency comparison

---

## Operating rules for this project

- Keep `src/app/` as canonical implementation; notebooks are for exploration.
- Make one small logical change per branch/PR; include tests and update docs when behavior changes.
- Preserve a known-good baseline and compare quality, latency and cost before adopting optimizations.
- Keep retrieval evaluation separate from generation evaluation.
- Never report a metric without its dataset/split, corpus coverage, configuration and limitations.
- Do not commit secrets, local vector stores, or untrusted pickle artifacts.
- Keep local development inexpensive; move to hosted services only when deployment is the task at hand.
