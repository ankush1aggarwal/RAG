# Project specification

**Status:** Week 2 complete — dense retrieval + grounded generation + FastAPI  
**Last updated:** 2026-10-11

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
| Lexical | **BM25** | `rank-bm25`; **Week 3** |
| Hybrid fusion | **RRF** first | **Week 3** |
| Reranking | Cross-encoder and/or **Cohere Rerank** | **Week 4** |
| Local LLM | **Ollama** (`/api/chat`) | `LLMProvider` interface |
| Hosted LLM | **Cohere** `ClientV2.chat` | `LLM_PROVIDER=cohere` |
| API | **FastAPI** + **Uvicorn** | `/health`, `/ready`, `POST /query` (Week 2); streaming later |
| Observability | Langfuse / OpenTelemetry | **Week 6** |
| Packaging / CI | Docker + **GitHub Actions** | **Weeks 7–8** |
| Hosting | Render + Qdrant Cloud | **Week 8** |

## Repository layout (this repo)

```text
RAG/
├── src/
│   ├── app/
│   │   ├── api/           # Week 2 — FastAPI
│   │   ├── config/        # Week 0–1 — Settings
│   │   ├── ingestion/     # Week 1
│   │   ├── retrieval/     # Week 1
│   │   ├── generation/    # Week 2
│   │   ├── pipeline/      # Week 2 — RAG orchestrator
│   │   ├── evaluation/    # planned Week 4
│   │   └── observability/ # planned Week 6
│   ├── scripts/
│   │   ├── ingest.py
│   │   ├── query.py
│   │   └── serve_api.py
│   └── tests/unit/
├── data/
├── notebooks/
├── docs/
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
| `retrieval` | `EmbeddingProvider`, `QdrantChunkStore` | ✅ dense only |
| `generation` | `LLMProvider`, Ollama + Cohere, prompt builder | ✅ |
| `pipeline` | `answer_query` → `RagAnswer` + citations | ✅ |
| `api` | FastAPI app, routes, readiness | ✅ |
| `evaluation` | qrels, metrics, reports | ⬜ Week 4 |
| `observability` | Logging, tracing | ⬜ Week 6 |

## Provider interfaces (design contract)

```text
EmbeddingProvider   → SentenceTransformerEmbedding, CohereEmbedding
LLMProvider         → OllamaLLM, CohereLLM
RerankerProvider    → (Week 4) local cross-encoder, Cohere Rerank
VectorStore         → QdrantChunkStore (local; cloud adapter later)
```

Factory functions:

- `create_embedding_provider(method, ...)` — `src/app/retrieval/embeddings.py`
- `create_llm_provider(method, ...)` — `src/app/generation/llm.py`

## Feature mapping (reference → this build)

| Capability | This repo implementation | Week | Status |
|------------|-------------------------|------|--------|
| Ingest + chunk | `ingestion/*`, `scripts/ingest.py` | 1 | ✅ |
| Dense retrieval | `QdrantChunkStore.search_text`, `scripts/query.py` | 1 | ✅ |
| Prompt + grounded generation | `generation/prompt.py`, `pipeline/rag.py` | 2 | ✅ |
| LLM providers | Ollama + Cohere chat | 2 | ✅ |
| HTTP API | `api/*`, `scripts/serve_api.py` | 2 | ✅ |
| BM25 + hybrid + RRF | — | 3 | ⬜ |
| Rerank + offline eval | — | 4 | ⬜ |
| SSE, hardening | — | 5 | ⬜ |
| Observability | — | 6 | ⬜ |
| Docker + CI + deploy | — | 7–8 | ⬜ |

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

### Ingestion / index (Week 1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `FIQA_CORPUS_LIMIT` | `1000` | Max documents per ingest run |
| `CHUNK_SIZE` | `100` | Chunk size in words |
| `CHUNK_OVERLAP_FRACTION` | `0.2` | Overlap fraction |
| `QDRANT_PATH` | `data/qdrant_local` | Empty → in-memory Qdrant |
| `COLLECTION_NAME` | `fiqa_sample` | Collection name |

### Planned (not wired in code yet)

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant Cloud |
| `LOG_LEVEL` | Structured logging (Week 6) |

## Week exit criteria

### Week 0 ✅ (foundation)

- [x] Repo structure, `pyproject.toml`, `requirements.txt`, `.gitignore`
- [x] `README.md`, `PROJECT_SPEC.md`
- [x] Local `.env` pattern
- [x] `src/app` package skeleton → grown into Weeks 1–2 modules

### Week 1 ✅ (dense retrieval)

- [x] FiQA sample → chunks with stable `doc_id` / `chunk_id`
- [x] Embeddings + Qdrant upsert + dense query
- [x] CLI returns top-k with scores and source IDs

### Week 2 ✅ (generation + API)

- [x] `LLMProvider` + Ollama + Cohere
- [x] Prompt builder with grounding instructions
- [x] `answer_query` with citations from hits
- [x] FastAPI `/health`, `/ready`, `POST /query`
- [x] Unit tests (61 tests in `src/tests/unit/`)

### Week 3 (next)

- [ ] BM25 index on same corpus/chunk IDs
- [ ] Dense + sparse retrieval paths
- [ ] RRF fusion + comparison report

## Evaluation discipline

- Use **qrels** (`BeIR/fiqa-qrels`) for retrieval metrics; do not tune repeatedly on the test set.
- Record every experiment: subset, chunking, embedding model, top-k, fusion, reranker, timestamp.
- **Separate** retrieval quality from generation quality.
- Keep a **known-good baseline** before each optimization ([README §19](README.md)).

## Dependency groups (`pyproject.toml`)

| Extra | Packages | When to install |
|-------|----------|-----------------|
| (core) | qdrant-client, pydantic, httpx, dotenv | Always |
| `embeddings` | sentence-transformers | Ingest + ST query |
| `hosted` | cohere | Cohere embed or chat |
| `ingestion` | datasets | `BeIR/fiqa` load in ingest |
| `api` | fastapi, uvicorn | `serve_api.py` |
| `dev` | pytest, ruff, fastapi (TestClient) | Development |
| `notebooks` | jupyter, pandas | `notebooks/` |

Full notebook workspace: `pip install -r requirements.txt`.

## Out of scope (for now)

- Copying reference repository source code
- `POST /query/stream` until Week 5 hardening
- HTTP ingest endpoints (ingest remains CLI)
- LangChain/LangGraph as core orchestrator
- Docker/cloud before Weeks 3–4 benchmarks justify deploy

## Next implementation steps (Week 3)

1. BM25 index builder aligned with Qdrant `doc_id` / `chunk_id`
2. Parallel dense + BM25 retrieval APIs in `retrieval/`
3. RRF fusion module + experiment script / notebook comparison
4. Update `answer_query` / API `retrieval.method` when hybrid is default
