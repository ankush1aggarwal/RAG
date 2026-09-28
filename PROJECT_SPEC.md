# Project specification

**Status:** Week 0 — foundation  
**Last updated:** 2026-09-27

This document maps the **reference architecture** to **this repository’s independent implementation**. It complements [`README.md`](README.md) and [`docs/production_rag_project_playbook.pdf`](docs/production_rag_project_playbook.pdf).

## Purpose

Build a portfolio-quality, production-minded RAG system for **financial information retrieval** using **BEIR FiQA-2018**, with measurable retrieval and generation quality, a deployable **FastAPI** service, and a path to **Agentic RAG**.

**Rule:** The reference repo informs *what* to build and *how to think about production*; **no copying** of its implementation code ([README §19](README.md)).

## Reference vs this repo

| Item | Reference | This project |
|------|-----------|--------------|
| Inspiration | [Ashok007-cmd/production-grade-rag](https://github.com/Ashok007-cmd/production-grade-rag) | Same engineering themes; **original codebase** |
| Corpus | Domain documents (reference-specific) | **FiQA-2018** (BEIR) |
| Vector DB | Production RAG patterns | **Qdrant** (local → Qdrant Cloud); not Chroma/Weaviate in core path |
| Code location | Monorepo / app package | **`src/`** — application code under `src/app/`, tooling under `src/scripts/`, tests under `src/tests/` |
| Learning artifacts | — | **`notebooks/`** — prototypes (FAISS, wine sample, hybrid demos); **not** the canonical production pipeline |
| Config / secrets | Env-based | **`.env`** at repo root (gitignored); local-only, not committed |

## Locked implementation decisions

Aligned with the playbook and README:

| Area | Choice | Notes |
|------|--------|--------|
| Python | **Workspace-driven** (`pyproject.toml` floor + `requirements.txt` freeze) | Playbook 3.11–3.12 is **reference only**; this repo follows the active `.venv` / `requirements.txt` unless we deliberately change them |
| Embeddings (baseline) | **Hugging Face** `sentence-transformers` | Initial model: **`all-MiniLM-L6-v2`** (used in notebooks; formalize in Week 1 config) |
| Vector store | **Qdrant** | `qdrant-client`; local embedded/persistent first |
| Lexical | **BM25** | `rank-bm25`; Week 3 |
| Hybrid fusion | **RRF** first | Weighted fusion = experiment |
| Reranking | Local **cross-encoder** and/or **Cohere Rerank** | Week 4 |
| Local LLM | **Ollama** | Provider interface; Week 2 |
| Hosted LLM | **Cohere** first; OpenAI optional | Rate-limit aware for demo |
| API | **FastAPI** + **Uvicorn** | `/health`, `/ready`, `POST /query`, streaming later |
| Observability | Structured logs → **Langfuse / OpenTelemetry** | Week 6 |
| Packaging / CI | **Docker** after native app works; **GitHub Actions** | Weeks 7–8 |
| Hosting | **Render** + **Qdrant Cloud** | Public demo |

## Repository layout (this repo)

```text
RAG/
├── src/
│   ├── app/           # Production package: api, ingestion, retrieval, generation, …
│   ├── scripts/       # CLI: ingest, query, evaluate, benchmark
│   ├── tests/         # unit / integration / evaluation
│   ├── experiments/   # experiment configs and results artifacts
│   └── data/          # Code-adjacent data helpers (if any); corpus files live at repo data/
├── data/              # FiQA / evaluation corpora (raw, evaluation splits)
├── notebooks/         # Exploratory and course-style prototypes
├── docs/              # Playbook, plans, guides
├── configs/           # (planned) YAML/TOML eval and app configs
├── pyproject.toml     # Package metadata and dependency groups
├── requirements.txt   # Workspace dependency freeze (notebooks + tooling)
├── .env               # Local secrets (gitignored; create/maintain locally)
└── PROJECT_SPEC.md
```

### Planned modules under `src/app/`

Evolve toward README §9 boundaries (names may shift):

| Module | Responsibility |
|--------|----------------|
| `config` | Settings from env + files (`pydantic-settings`) |
| `ingestion` | FiQA load, normalize, chunk, IDs on every chunk |
| `retrieval` | Dense (Qdrant), BM25, RRF, reranker adapters |
| `generation` | Prompt/context builder, citations |
| `pipeline` | RAG orchestrator |
| `api` | FastAPI routes, schemas, errors |
| `evaluation` | qrels, Recall@K, MRR, nDCG, reports |
| `observability` | Logging, tracing, latency/cost fields |

## Provider interfaces (design contract)

Implementations must swap without rewriting the pipeline:

```text
EmbeddingProvider   → HuggingFace (baseline), Cohere (experiment)
LLMProvider         → Ollama (dev), Cohere / OpenAI (hosted)
RerankerProvider    → local cross-encoder, Cohere Rerank
VectorStore         → Qdrant local vs Qdrant Cloud (same client adapter)
```

## Feature mapping (reference → this build)

| Capability | Reference intent | This repo plan | Week (playbook) |
|------------|------------------|----------------|-----------------|
| Ingest + chunk | Document → chunks + metadata | FiQA docs, chunk IDs, normalization | 1 |
| Dense retrieval | Vector search | Qdrant + embedding provider | 1 |
| Generation + citations | Grounded answers | Ollama + prompt template | 2 |
| HTTP API | Serve RAG | FastAPI health/ready/query | 2 |
| BM25 | Lexical branch | rank-bm25 index on same corpus | 3 |
| Hybrid + RRF | Combine rankings | RRF fusion, comparison table | 3 |
| Rerank | Rescore top-N | Cross-encoder / Cohere | 4 |
| Offline eval | Benchmark loop | FiQA qrels, metrics, `experiments/results/` | 4 |
| Hardening | Validation, SSE, tests | Config, limits, streaming, pytest | 5 |
| Observability | Trace request path | Latency breakdown, Langfuse | 6 |
| Docker + CI | Reproducible deploy | Dockerfile, GitHub Actions | 7 |
| Public demo | Hosted API + UI | Render, Qdrant Cloud, simple UI | 8 |

## Environment variables

Use **`.env`** at the repository root (gitignored). Document new keys here when they are introduced in code; do not commit secret values.

| Variable | Required | Purpose |
|----------|----------|---------|
| `COHERE_API_KEY` | Optional (Week 2+) | Cohere LLM, embeddings, or rerank experiments |

Additional variables (add to `.env` locally when implemented):

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | Ollama API base (default `http://localhost:11434`) |
| `QDRANT_URL` / `QDRANT_API_KEY` | Local or Qdrant Cloud |
| `EMBEDDING_MODEL` | Default `all-MiniLM-L6-v2` |
| `LOG_LEVEL` | Logging verbosity |

## Evaluation discipline

- Use **qrels** for retrieval metrics; do not tune on the test set repeatedly.
- Record every experiment: subset, chunking, embedding model, top-k, fusion, reranker, timestamp.
- **Separate** retrieval quality from generation quality.
- Keep a **known-good baseline** before each optimization ([README §19](README.md)).

## Week 0 exit criteria (playbook)

- [ ] Git repository with README, `.gitignore`, local `.env`, `pyproject.toml`
- [ ] `PROJECT_SPEC.md` (this file)
- [ ] Smoke test: Python runs, `qdrant-client` imports, Ollama responds
- [ ] Initial FiQA sample under `data/`
- [ ] Skeleton under `src/` ready for Week 1 code

## Out of scope (for now)

- Copying reference repository source code
- Second vector database in the core stack (Chroma/Weaviate in `requirements.txt` are notebook-era only—trim over time)
- Docker and cloud deploy before native pipeline meets Week 2–4 exit criteria
- LangChain/LangGraph as the core orchestrator (playbook: dependencies must earn their place)

## Next implementation steps (Week 1)

1. FiQA small sample loader → `src/app/ingestion/`
2. `EmbeddingProvider` + Hugging Face adapter
3. Qdrant collection create/upsert/search
4. `src/scripts/query.py` — print top-k with source IDs and scores
