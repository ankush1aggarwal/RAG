# Project specification

**Status:** Week 4 complete; **Week 5** = experimentation sprint; **Week 6** = production API (frozen retrieval profile)  
**Last updated:** 2026-10-20

This document maps the **reference architecture** to **this repository’s independent implementation**. It complements [`README.md`](README.md) and [`docs/production_rag_project_playbook.pdf`](docs/production_rag_project_playbook.pdf).

## Purpose

Build a portfolio-quality, production-minded RAG system for **financial information retrieval** using **BEIR FiQA-2018**, with measurable retrieval and generation quality, a deployable **FastAPI** service, and a path to **Agentic RAG**.

**Rule:** The reference repo informs *what* to build and *how to think about production*; **no copying** of its implementation code ([README §19](README.md)).

## Reference vs this repo

| Item | Reference | This project |
|------|-----------|--------------|
| Inspiration | [Ashok007-cmd/production-grade-rag](https://github.com/Ashok007-cmd/production-grade-rag) | Same engineering themes; **original codebase** |
| Corpus | Domain documents (reference-specific) | **FiQA-2018** (`BeIR/fiqa`) |
| Vector DB | Production RAG patterns | **Qdrant** (local persistent → Qdrant Cloud later) |
| Code location | Monorepo / app package | **`src/`** — `src/app/`, `src/scripts/`, `src/tests/` |
| Learning artifacts | — | **`notebooks/`** — prototypes; **not** the canonical pipeline |
| Config / secrets | Env-based | **`.env`** at repo root (gitignored); no committed `.env.example` |

## Locked implementation decisions

| Area | Choice | Notes |
|------|--------|--------|
| Python | **Workspace-driven** (`pyproject.toml` floor + `requirements.txt` freeze) | Playbook 3.11–3.12 is **reference only** |
| Embeddings (baseline) | **Hugging Face** `sentence-transformers` | Default **`all-MiniLM-L6-v2`** |
| Embeddings (hosted) | **Cohere** `embed-v4.0` | Optional via `EMBEDDING_METHOD=cohere` |
| Vector store | **Qdrant** | `qdrant-client`; local path or `:memory:` |
| Lexical | **BM25** | `rank-bm25` (**core** dep); pickle aligned with chunk IDs |
| Hybrid fusion | **RRF** (equal + **weighted** lists) | `reciprocal_rank_fusion`; `fusion_weights.py` + `weighted_hybrid` |
| Reranking | Cross-encoder and/or **Cohere Rerank** | Optional at query time (`use_rerank`); Week 4 ✅ |
| Local LLM | **Ollama** (`/api/chat`) | `LLMProvider` interface |
| Hosted LLM | **Cohere** `ClientV2.chat` | `LLM_PROVIDER=cohere` |
| API | **FastAPI** + **Uvicorn** | `/health`, `/ready`, `POST /query` with `method`: `dense` \| `bm25` \| `hybrid`; streaming later |
| Observability | Langfuse / OpenTelemetry | **Week 7** |
| Packaging / CI | Docker + **GitHub Actions** | **Week 8** |
| Hosting | Render + Qdrant Cloud | **Week 9** |

## Repository layout (this repo)

```text
RAG/
├── src/
│   ├── app/
│   │   ├── api/           # FastAPI + readiness (Qdrant + BM25 + LLM)
│   │   ├── config/        # Settings
│   │   ├── ingestion/     # Week 1
│   │   ├── retrieval/     # Dense, BM25, fusion, hybrid
│   │   ├── generation/    # Week 2
│   │   ├── pipeline/      # RAG orchestrator + retrieve_hits
│   │   ├── evaluation/    # Benchmark (Week 3); qrels, experiments, error analysis (Week 4)
│   │   └── observability/ # planned Week 6
│   ├── scripts/
│   │   ├── ingest.py      # Qdrant + BM25; --full-corpus, --no-recreate
│   │   ├── query.py       # dense / bm25 / hybrid; --rerank
│   │   ├── evaluate.py, evaluate_compare.py, prepare_fiqa_eval.py
│   │   ├── run_week4_experiments.py, analyze_rerank_dense.py
│   │   └── serve_api.py
│   └── tests/unit/
├── data/
│   └── benchmarks/        # qrels JSON, week4_exp*.json, fusion weights, protocol
├── notebooks/
│   └── retrieval_benchmark.ipynb
├── docs/reports/          # week3–week4 experiment and eval reports
├── pyproject.toml
├── requirements.txt
├── .env
└── PROJECT_SPEC.md
```

## Module status (`src/app/`)

| Module | Responsibility | Status |
|--------|----------------|--------|
| `config` | `pydantic-settings` from `.env` | ✅ |
| `ingestion` | FiQA load, normalize, chunk, schemas | ✅ |
| `retrieval` | Embeddings, Qdrant, BM25, RRF, hybrid, **reranker**, fusion weights | ✅ |
| `generation` | `LLMProvider`, Ollama + Cohere, prompt builder | ✅ |
| `pipeline` | `retrieve_hits`, `answer_query` → `RagAnswer` + citations | ✅ |
| `api` | FastAPI app, `POST /query` retrieval `method`, readiness | ✅ |
| `evaluation` | Benchmark, qrels metrics, Week 4 experiment runner, rerank error analysis | ✅ |
| `observability` | Logging, tracing | ⬜ Week 7 |

## Provider interfaces (design contract)

```text
EmbeddingProvider   → SentenceTransformerEmbedding, CohereEmbedding
LLMProvider         → OllamaLLM, CohereLLM
RerankerProvider    → CrossEncoderReranker, CohereReranker (`create_reranker_provider`)
VectorStore         → QdrantChunkStore (local; cloud adapter later)
LexicalIndex        → Bm25ChunkIndex (in-memory + pickle)
```

Factory functions:

- `create_embedding_provider(method, ...)` — `src/app/retrieval/embeddings.py`
- `create_llm_provider(method, ...)` — `src/app/generation/llm.py`
- `create_reranker_provider(method, ...)` — `src/app/retrieval/reranker.py`

## Feature mapping (reference → this build)

| Capability | This repo implementation | Week | Status |
|------------|-------------------------|------|--------|
| Ingest + chunk | `ingestion/*`, `scripts/ingest.py` (`--recreate` default **true**; `--no-recreate` validates manifest) | 1 | ✅ |
| Dense retrieval | `QdrantChunkStore.search_text`, `scripts/query.py` | 1 | ✅ |
| BM25 + RRF + hybrid | `bm25_store.py`, `fusion.py`, `hybrid.py`, `retrieve_hits` | 3 | ✅ |
| Prompt + grounded generation | `generation/prompt.py`, `pipeline/rag.py` | 2 | ✅ |
| LLM providers | Ollama + Cohere chat | 2 | ✅ |
| HTTP API | `api/*`, `scripts/serve_api.py`; `QueryRequest.method` | 2–3 | ✅ |
| Retrieval benchmark | `evaluation/retrieval_benchmark.py`, notebook + report | 3 | ✅ |
| Rerank + offline eval (qrels) | `reranker.py`, `retrieval_eval.py`, `run_week4_experiments.py` | 4 | ✅ |
| Experiment sprint E3–E9 (E4 Jev) | `docs/RAG_PROJECT_ROADMAP.md` | 5 | ⬜ |
| SSE, API hardening (frozen retrieval) | — | 6 | ⬜ |
| Observability | — | 7 | ⬜ |
| Docker + CI + deploy | — | 8–9 | ⬜ |

## Environment variables

Use **`.env`** at the repository root (gitignored). Pydantic field names in `Settings` accept the usual `SCREAMING_SNAKE` env aliases.

### Secrets

| Variable | Required when | Purpose |
|----------|---------------|---------|
| `COHERE_API_KEY` | `EMBEDDING_METHOD=cohere` or `LLM_PROVIDER=cohere` | Cohere API |

### Embeddings (Week 1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_METHOD` | `sentence_transformer` | `sentence_transformer` or `cohere` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model |
| `COHERE_EMBED_MODEL` | `embed-v4.0` | Cohere embedding model |
| `COHERE_EMBED_BATCH_SIZE` | `96` | Cohere embed batching |
| `COHERE_EMBED_BATCH_SLEEP_SECONDS` | `60.0` | Sleep between Cohere batches (rate limits) |

### LLM (Week 2)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `cohere` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `mistral:latest` | Ollama model tag |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Ollama HTTP timeout |
| `COHERE_CHAT_MODEL` | `command-r-08-2024` | Cohere chat model |

### Ingestion / index (Weeks 1–3)

| Variable | Default | Purpose |
|----------|---------|---------|
| `FIQA_CORPUS_LIMIT` | `1000` | Max documents per ingest (`0` or negative = **full** ~57k corpus) |
| `CHUNK_SIZE` | `100` | Chunk size in words |
| `CHUNK_OVERLAP_FRACTION` | `0.2` | Overlap fraction |
| `QDRANT_PATH` | `data/qdrant_local` | Empty → in-memory Qdrant |
| `COLLECTION_NAME` | `fiqa_sample` | Collection name |
| `BM25_INDEX_DIR` | `data/bm25_indexes` | Pickle path: `<dir>/<collection>.pkl` (gitignored; build via `ingest.py`) |

### Local index artifacts (not in git)

| Path | Produced by | In repo? |
|------|-------------|----------|
| `data/qdrant_local/` | `ingest.py` | No (`.gitignore`) |
| `data/bm25_indexes/*.pkl` | `ingest.py` | No (`.gitignore`) |
| `data/bm25_indexes/*.manifest.json` | `ingest.py` | No (same directory) |

**`ingest.py --recreate` (default):** deletes and rebuilds the Qdrant collection and replaces BM25 + manifest. Use after changing corpus limit, chunking, or embeddings.

**`ingest.py --no-recreate`:** re-embeds and upserts only if the new run matches the saved manifest (corpus limit, chunking, embedding method/model/dimension, document/chunk counts) and Qdrant `point_count` equals `num_chunks`. Otherwise ingest exits with an error — run with `--recreate`.

**Atomic publish:** BM25 is built in memory; Qdrant is recreated/upserted first; pickle + manifest are written to `*.tmp` and `os.replace`d only after a successful upsert (two separate atomic file replaces, not one transaction with Qdrant). **`--recreate`** removes any existing pickle + manifest before the rebuild so a failed run cannot satisfy `/ready` with stale metadata.

**`/ready` alignment:** manifest must match runtime ingest/embedding settings (method, model id, dimension, collection, corpus limit, chunking), Qdrant point count and vector size, BM25 chunk count, and matching `ingest_generation_id` on manifest and BM25 pickle.

Reproducibility comes from **ingest settings** (`FIQA_CORPUS_LIMIT`, chunking, embedding model) and `python src/scripts/ingest.py`, not from committing binary indexes. This project does **not** ship a demo BM25 pickle in git.

### BM25 pickle security

`Bm25ChunkIndex.save` / `load` use Python **pickle**. Pickle can execute arbitrary code when loading malicious files. Only load indexes you created with this project's ingest pipeline (or other sources you fully trust). Do not point `BM25_INDEX_DIR` at user-uploaded or untrusted paths in production without a safer serialization format.

### API retrieval (Weeks 3–4)

`POST /query` body field **`method`**: `dense` (default), `bm25`, or `hybrid`. Optional **`use_rerank`** (requires configured `RerankerProvider` on `AppResources`). Response `retrieval` includes `method`, `top_k`, `reranker`, `candidates_retrieved`.

`GET /ready` requires Qdrant collection, loaded BM25 index, manifest/index alignment, and LLM readiness.

### Reranker (Week 4)

| Variable | Default | Purpose |
|----------|---------|---------|
| `RERANKER_PROVIDER` | `cross_encoder` | `cross_encoder`, `cohere`, or `none` |
| `CROSS_ENCODER_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local CE model |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere Rerank model |
| `RERANK_CANDIDATE_LIMIT` | `50` | Chunk candidates before rerank down to `top_k` |

### Planned (not wired in code yet)

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud |
| `LOG_LEVEL` | Structured logging (Week 6) |
| Default `method=hybrid` via settings | Optional product default (currently client default is `dense`) |

## Week exit criteria

### Week 0 ✅ (foundation)

- [x] Repo structure, `pyproject.toml`, `requirements.txt`, `.gitignore`
- [x] `README.md`, `PROJECT_SPEC.md`
- [x] Local `.env` pattern
- [x] `src/app` package skeleton → grown into Weeks 1–3 modules

### Week 1 ✅ (dense retrieval)

- [x] FiQA sample → chunks with stable `doc_id` / `chunk_id`
- [x] Embeddings + Qdrant upsert + dense query
- [x] CLI returns top-k with scores and source IDs

### Week 2 ✅ (generation + API)

- [x] `LLMProvider` + Ollama + Cohere
- [x] Prompt builder with grounding instructions
- [x] `answer_query` with citations from hits
- [x] FastAPI `/health`, `/ready`, `POST /query`
- [x] `QdrantChunkStore.close()` on API shutdown (and ingest CLI `finally`)

### Chunking behavior (Week 1, updated)

Word windows advance by `chunk_size - overlap_int`. The **first** window is always kept (even if shorter than the overlap width). Only **trailing** windows with word count ≤ overlap width are dropped, to avoid tiny duplicate tail chunks.

### Week 3 ✅ (hybrid retrieval)

- [x] BM25 index on same corpus/chunk IDs (`ingest.py` saves pickle)
- [x] Dense + BM25 + hybrid (`retrieve_hits`, `query.py`, `hybrid_search_text`)
- [x] RRF fusion (`fusion.py`)
- [x] API `method` + `/ready` BM25 check
- [x] Fixed query set + benchmark notebook + comparison report
- [x] Weighted RRF + validation weight tuning (Week 4)
- [x] Unit tests (162+ in `src/tests/unit/`)

### Week 4 ✅ (reranking + evaluation)

- [x] Reranker interface + local cross-encoder (optional Cohere)
- [x] BEIR qrels → NDCG / MRR / MAP / recall@k; full-corpus exp program (see `docs/reports/week4_experiment_plan.md`)
- [x] `evaluate.py`, `run_week4_experiments.py`, reproducible eval protocol + citations in eval path
- [x] Dense rerank error analysis (Exp 4 vs Exp 5 dense; `analyze_rerank_dense.py`)
- [x] Reports: `docs/reports/week4_retrieval_program_report.md` + per-experiment markdown

### Production retrieval profile (frozen after experiments)

| Setting | Current value | Notes |
|---------|---------------|--------|
| Method | `dense` | Test NDCG@10 ≈ 0.37; best latency |
| Rerank | off | CE did not improve dense NDCG@10 |
| `top_k` | 10 | Week 4 protocol |
| Collection | `fiqa` (full corpus) | |

CLI/eval may use `bm25`, `hybrid`, `weighted_hybrid`, `use_rerank` for **Week 5+ experiments**. Week **6** API will expose only this profile on `/query`. Reports: [`rag_experiment_results.md`](docs/reports/rag_experiment_results.md), per-experiment [`experiments/README.md`](docs/reports/experiments/README.md), catalog [`rag_experiment_program_report.md`](docs/reports/rag_experiment_program_report.md).

### Week 5 (next) — experimentation sprint

- [ ] E3–E9 per [`docs/RAG_PROJECT_ROADMAP.md`](docs/RAG_PROJECT_ROADMAP.md) (incl. **E4** Jev ranking-only); update experiment report and production profile if a treatment wins.

### Week 6 — production API engineering

- [ ] Streaming, request IDs, errors, integration tests; **single served retrieval design**

## Evaluation discipline

- **Week 3:** unlabeled comparison — latency, top-k Jaccard overlap, qualitative snippets (`docs/reports/week3_retrieval_comparison.md`).
- **Week 4+:** use **qrels** (`BeIR/fiqa-qrels`) for retrieval metrics; do not tune repeatedly on the test set.
- Record every experiment: subset, chunking, embedding model, top-k, fusion, reranker, timestamp.
- **Separate** retrieval quality from generation quality.
- Keep a **known-good baseline** before each optimization ([README §19](README.md)).

## Dependency groups (`pyproject.toml`)

| Extra | Packages | When to install |
|-------|----------|-----------------|
| (core) | qdrant-client, pydantic, httpx, dotenv, **rank-bm25** | Always (`app.retrieval` imports BM25) |
| `embeddings` | sentence-transformers | Ingest + ST query |
| `hosted` | cohere | Cohere embed or chat |
| `ingestion` | datasets | `BeIR/fiqa` load in ingest |
| `lexical` | rank-bm25 (alias) | Optional extra name; same package as core since Week 3 |
| `api` | fastapi, uvicorn | `serve_api.py` |
| `dev` | pytest, ruff, fastapi (TestClient) | Development |
| `notebooks` | jupyter, pandas | `notebooks/` |

**Install patterns (use together for a full dev venv):**

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned **third-party** deps only (UTF-8); does **not** install `fiqa-production-rag` |
| `requirements-editable.txt` | `-e .[extras]` — **local** `src/` from `pyproject.toml` |
| `requirements-git.txt` | `-e git+https://github.com/…/RAG.git@…` — same package from **GitHub** |

```text
pip install -r requirements.txt
pip install -r requirements-editable.txt
```

Regenerating the freeze: `pip freeze` and **omit** any `-e .` or `-e git+…` lines before saving to `requirements.txt`.

## Out of scope (for now)

- Copying reference repository source code
- `POST /query/stream` until Week 6
- HTTP ingest endpoints (ingest remains CLI)
- LangChain/LangGraph as core orchestrator
- Docker/cloud until Weeks 8–9 (experiment + API hardening first)

## Next implementation steps

**Week 5:** Run E3 → **E4** (Jev `rerank()`, no filtering) → optional E4b → E5–E8; fill sheets under `docs/reports/experiments/` and update `rag_experiment_results.md`.

**Week 6:** API hardening; lock `/query` to frozen production retrieval profile.
