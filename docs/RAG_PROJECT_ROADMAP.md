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
| Week 6 — Production API engineering | ⬜ | Hardening, streaming, errors, integration tests; **serve frozen retrieval profile only** |
| Week 7 — Observability | ⬜ | Structured logs, tracing, latency/error/cost visibility |
| Week 8 — Container + CI | ⬜ | Reproducible Docker runtime and automated checks on GitHub |
| Week 9 — Deployment + public demo | ⬜ | Hosted API, Qdrant Cloud, smoke tests and demo UI |
| Final polish — Portfolio | ⬜ | Final docs, architecture image, demo evidence, blog/LinkedIn write-up |
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
| E4 | ~~Jev reranker (hosted)~~ | — | — | **Removed** — cost; not in program |
| E4b | ~~Jev relevance filtering~~ | — | — | **Removed** — depended on E4 |
| E5 | Reranker candidate depth (10 / 20 / 30 / 50) | Week 5 | Core | **Skipped** — E2 rejected CE; no new reranker |
| E6 | Chunk size sensitivity | Week 5 | High | ✅ 200w adopted |
| E7 | Chunk overlap sensitivity | Week 5 | High | ✅ 20% @ 200w |
| E8 | Embedding model comparison | Week 5 | High | ✅ BGE-base |
| E9 | **Paragraph-aware chunking** (vs E8 BGE 200w) | Week 5 | Medium | ✅ Complete (tie → keep E8) |
| E10 | Generation + citation grounding | Week 6–7 | Core | ⬜ After retrieval freeze |
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

### Hosted Jev reranker (E4 / E4b) — removed from program

**E4** (Jev `rerank()`) and **E4b** (relevance filtering) were planned as optional hosted-API trials. They are **out of scope** due to API cost. Rerank evidence stays **Week 4 E2** (local cross-encoder). Sheets retained for history: [`E4_jev_rerank.md`](reports/experiments/E4_jev_rerank.md), [`E4b_jev_relevance_filtering.md`](reports/experiments/E4b_jev_relevance_filtering.md).

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
7. **Freeze** the production retrieval profile; document in the experiment report. **Week 6** implements API hardening and **locks the served pipeline** to that profile (other methods remain CLI/eval-only).
8. **E10** after retrieval freeze; **E11** uses observability from Week 7 where possible.

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

**Frozen retrieval (eval / post–Week 6 target):** dense, no rerank, **`top_k=10`**, **200w / 20% overlap**, **`BAAI/bge-base-en-v1.5`**, index **`fiqa_cs200_bge_base_en_v1_5`** (test NDCG@10 **0.394**). **Served API** still uses legacy **`fiqa`** (100w MiniLM) until Week 6 wiring.

**Reports:** [`rag_experiment_results.md`](reports/rag_experiment_results.md) · [`rag_retrieval_lessons.md`](reports/rag_retrieval_lessons.md) · [`rag_experiment_program_report.md`](reports/rag_experiment_program_report.md).

**Indexes on disk:** `fiqa` + `fiqa_cs200_bge_base_en_v1_5` only (`src/scripts/cleanup_local_indexes.py`).

---

## Week 6 — Production API engineering ⬜

**Goal:** Harden the API and **serve only the frozen production retrieval design** (experimental methods not exposed on default `/query` unless explicitly scoped for dev).

**Includes:**

1. **Review API contracts and errors.** Define expected status codes and response bodies for invalid requests, missing BM25, unavailable LLM/provider errors, retrieval failures and timeouts.
2. **Add streaming** if still part of the agreed scope (`POST /query/stream`, likely Server-Sent Events). Keep non-streaming `/query` working as the compatibility path.
3. **Add timeout and resource controls.** Ensure model/API calls are bounded; configure appropriate client timeouts and safe error mapping. Do not add blind retries to non-idempotent work.
4. **Check lifecycle/concurrency.** Ensure clients are closed on shutdown, avoid unsafe concurrent access to embedded local Qdrant, and keep expensive startup/model loading out of each request.
5. **Strengthen readiness.** Avoid duplicate alignment checks per `/ready` request; turn provider/network failures into controlled not-ready diagnostics. Do not treat a configured API key as proof that the provider is reachable unless that is explicitly the chosen check.
6. **Add API-level integration tests.** Test success, invalid input, retrieval/provider failure, timeout behavior, streaming termination/errors, and readiness when dependencies are missing or misaligned.
7. **Update OpenAPI/examples and docs.** Include request/response examples, error semantics and run commands.
8. **Enforce the frozen production retrieval profile in code** (`method`, `use_rerank`, `top_k` from settings — today target: dense, no rerank, `top_k=10`). Do not rely on documentation alone; client overrides should be rejected or ignored per chosen API policy.

**Deliverables:** hardened routes/schemas, optional stream route, API integration tests, updated API documentation.  
**Definition of done:** `/query` serves only the frozen profile; errors are predictable; streaming works if committed to; tests cover critical success/failure paths.

## Week 7 — Observability ⬜

**Goal:** Explain what a query did and where latency/errors/cost came from without exposing secrets or unnecessarily logging user data.

1. Choose one primary approach: lightweight structured JSON logs first; add OpenTelemetry or Langfuse only where it gives concrete value.
2. Assign a request/correlation ID and propagate it through retrieval, reranking and generation.
3. Log method, candidate counts, reranker use, timings by stage, errors and provider/model IDs. Avoid logging API keys or full sensitive prompts/context by default.
4. Track p50/p95 and error counts; record token usage and estimated LLM cost when the provider returns the needed usage data.
5. Add tests for log fields/redaction and failure reporting; document how to enable/disable telemetry.
6. Update the benchmark/report so total query latency can be separated into retrieval, reranking and generation.

**Deliverables:** `src/app/observability/` implementation, config switches, tests and a short operational guide.  
**Definition of done:** a sample request can be traced through the main stages and its latency/failure cause is observable without leaking secrets.

## Week 8 — Docker + CI ⬜

**Goal:** Recreate the runtime in a clean environment and automatically reject regressions.

1. Create a production-appropriate Dockerfile; use a non-root user, a sensible working directory and only required dependencies where feasible.
2. Decide how models and data are supplied. Do not bake secrets or local Qdrant/BM25 artifacts into the image. Account for the size of sentence-transformer models and native dependencies.
3. Add a local Docker run path and persistent-volume/environment instructions. A compose file is optional if it meaningfully simplifies local testing.
4. Add GitHub Actions for dependency install, Ruff, unit tests and any fast integration checks.
5. Keep costly downloads/API calls out of ordinary CI tests; use mocks or cached test fixtures where appropriate.
6. Add container smoke tests for startup, `/health` and readiness behavior. Validate the container build from a clean checkout.
7. Pin deployment-critical dependencies / model identifiers deliberately; fix `requirements-git.txt` to use a commit SHA when a reproducible remote checkout is intended instead of floating `@main`.

**Deliverables:** Dockerfile, `.dockerignore`, CI workflow(s), updated run instructions.  
**Definition of done:** a clean CI run builds the image, lint and tests pass, and the container starts using documented configuration.

## Week 9 — Cloud deployment + demo ⬜

**Goal:** Give a reviewer a repeatable way to try the system without running your laptop.

1. Provision Qdrant Cloud and configure connection URL/API key through the hosting platform's secrets mechanism.
2. Decide where the corpus/embeddings are built and how they are loaded. Make the ingestion job repeatable; do not assume ephemeral application filesystem storage preserves indexes.
3. Deploy the API (Render is the current plan unless cost/constraints change) and configure start command, runtime variables, health checks and resource limits.
4. Make readiness checks match the deployment architecture. Use a persistent, explicitly managed index generation and avoid destructive ingest at API startup.
5. Run smoke tests against the deployed URL: health, readiness, dense/BM25/hybrid query, reranked query (if enabled), citations and controlled failure behavior.
6. Add a lightweight user interface or hosted API demo. Keep it simple: question input, retrieval choice, answer and source citations. Do not add UI features before the API is stable.
7. Document cost, cold starts, model downloads, rate limits, persistence and operational limitations honestly.

**Deliverables:** deployed API URL, Qdrant Cloud setup notes, demo UI/API examples, smoke-test record and deployment runbook.  
**Definition of done:** a person outside your laptop can run a documented query and inspect citations; service failures are diagnosable; secrets and data are not committed.

## Final polish — portfolio deliverables ⬜

1. Update README with architecture, setup, API examples, evaluation tables and links to reports.
2. Add a clear architecture diagram: ingest → chunks/embeddings → Qdrant + BM25 → RRF → reranker → prompt/LLM → answer/citations; include evaluation/observability separately.
3. Add a short demo GIF or screenshots only after the deployed path is reliable.
4. Publish a technical blog that explains the engineering decisions and measured trade-offs, not just a tutorial-style list of features.
5. Publish a LinkedIn summary that links to the blog and repo, includes one meaningful result, and makes limitations clear.
6. Add the Agentic RAG direction to the roadmap without representing future capabilities as implemented.
7. Do one final reproducibility pass from a clean environment and verify links, tests, CI and deployed smoke tests.

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

### Immediate next action — Week 5 experimentation
- [x] Week 4 artifacts and [`rag_experiment_program_report.md`](reports/rag_experiment_program_report.md).
- [x] E3 (candidate diversity) — control pool frozen.
- [ ] **E6–E8** (E7 after E6); optional **E9**. E4/E4b removed; **E5** skipped.
- [ ] Update experiment report with decisions; confirm or revise production profile.

### Week 6 (API)
- [ ] API contract/error handling
- [ ] Streaming route (if in agreed scope)
- [ ] Timeouts, lifecycle and readiness cleanup
- [ ] API integration tests
- [ ] API docs/examples updated

### Week 7
- [ ] Structured logs and correlation IDs
- [ ] Stage-level latency/error instrumentation
- [ ] Token/cost accounting where available
- [ ] Redaction/privacy checks and tests

### Week 8
- [ ] Dockerfile and `.dockerignore`
- [ ] Clean container build/start/smoke tests
- [ ] GitHub Actions lint + tests + build
- [ ] Reproducible dependency/runtime configuration

### Week 9
- [ ] Qdrant Cloud provisioned and secured
- [ ] Deployment configuration and persistent-index strategy
- [ ] Hosted API smoke tests
- [ ] Public demo experience
- [ ] Deployment runbook and limitations/cost documented

### Portfolio
- [ ] Architecture diagram
- [ ] Final evaluation report + error analysis
- [ ] README polish and demo screenshots/GIF
- [ ] Blog published
- [ ] LinkedIn post published
- [ ] Clean-environment reproduction pass

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
