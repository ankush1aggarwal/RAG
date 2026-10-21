# Production-Grade RAG for Financial Information Retrieval

This project is inspired by the architecture and engineering goals of modern production RAG systems, but the implementation, experiments, dataset configuration, engineering decisions, deployment, evaluation methodology, and application are developed independently.

The project is designed to progress from a simple RAG baseline to a production-ready application with hybrid retrieval, reranking, evaluation, observability, CI/CD, and public deployment — and eventually serve as the foundation for an **Agentic RAG** system.

---

## Why This Project?

The goal is not simply to build "a chatbot with a vector database."

The goal is to understand and implement the complete lifecycle of a modern RAG application:

```text
Document Corpus
      ↓
Ingestion
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Database
      ↓
Dense Retrieval ───────┐
                       │
BM25 / Sparse Retrieval│
                       ↓
                 Hybrid Fusion
                       ↓
                    Reranking
                       ↓
                 Context Builder
                       ↓
                    LLM
                       ↓
              Answer + Citations
                       ↓
        Evaluation + Observability
                       ↓
             API + Web Application
                       ↓
            CI/CD + Public Deployment
```

The project will emphasize **experimentation and measurement**, rather than implementing components without understanding their impact.

---

## Project Goals

By the end of the project, the system should demonstrate:

- A complete RAG ingestion and query pipeline
- A vector-search implementation using **Qdrant**
- Dense retrieval using embedding models
- Lexical retrieval using **BM25**
- Hybrid retrieval using **Reciprocal Rank Fusion (RRF)**
- Optional weighted hybrid retrieval
- Reranking using a cross-encoder and/or Cohere
- Ground-truth retrieval evaluation
- Generation-quality evaluation
- Latency and cost measurement
- FastAPI-based serving
- Streaming responses
- Configuration and provider abstraction
- Logging, tracing, and observability
- Automated tests
- Docker-based packaging
- GitHub Actions CI/CD
- Public deployment
- A simple web demo
- Technical documentation and experiment reports
- A clear path to extending the same system into Agentic RAG

---

# 1. Current Project Status

> **Current phase: Week 5 complete — retrieval experiment program (E0–E9)**  
> **Next phase: Week 6 — Production API** (enforce frozen retrieval on `/query`; streaming, tests).  
> **Lessons:** [`docs/reports/rag_retrieval_lessons.md`](docs/reports/rag_retrieval_lessons.md) · **Results:** [`rag_experiment_results.md`](docs/reports/rag_experiment_results.md) · [roadmap](docs/RAG_PROJECT_ROADMAP.md)

Canonical implementation lives under **`src/app/`**. The notebook [`notebooks/code_dev.ipynb`](notebooks/code_dev.ipynb) is used for experiments before porting patterns into `src/`. Authoritative planning docs: [`PROJECT_SPEC.md`](PROJECT_SPEC.md), [`docs/production_rag_project_playbook.pdf`](docs/production_rag_project_playbook.pdf).

### Progress summary

| Week | Theme | Status | Exit criterion (playbook) |
|------|--------|--------|---------------------------|
| 0 | Foundation | Done (local) | Repo, config, architecture docs, dev environment |
| 1 | Dense retrieval | **Done** | Natural-language query → top-k chunks with IDs and scores from Qdrant |
| 2 | Generation + API | **Done** | Local HTTP `POST /query` → answer + traceable citations; `/health` + `/ready` |
| 3 | Hybrid retrieval | **Done** | BM25 + dense + RRF; API/CLI; benchmark notebook + comparison report |
| 4 | Rerank + qrels eval | **Done** | Experiments 1–5; production default **dense, no rerank** (see experiment report) |
| 5 | Experimentation | **Done** | E3–E9; frozen eval: **BGE 200w** NDCG@10 **0.394** (test) |
| 6 | Production API | **Next** | Hardening + streaming; API serves frozen design only |
| 7–9 | Ops + deploy | Planned | Observability, Docker/CI, public demo |

**Automated tests:** **185** unit tests — `pytest src/tests/unit` (unset `FIQA_CORPUS_LIMIT` in shell if `.env` uses `0` and defaults tests fail).

**Experiment results:** [`docs/reports/rag_experiment_results.md`](docs/reports/rag_experiment_results.md) (full report); per-experiment [`docs/reports/experiments/`](docs/reports/experiments/README.md); catalog [`rag_experiment_program_report.md`](docs/reports/rag_experiment_program_report.md). Week 4 runbook: [`week4_retrieval_program_report.md`](docs/reports/week4_retrieval_program_report.md).

**Production vs experiments:** CLI/eval scripts can run `dense`, `bm25`, `hybrid`, `weighted_hybrid`, and optional rerank. **Frozen retrieval (Week 5):** dense, rerank off, **`top_k=10`**, **200w / 20%**, **`BAAI/bge-base-en-v1.5`**, collection **`fiqa_cs200_bge_base_en_v1_5`**. The **HTTP API** still serves legacy **`fiqa`** (100w MiniLM) and defaults **`top_k=3`** until **Week 6**. Eval rules: [`docs/reports/eval_protocol_reference.md`](docs/reports/eval_protocol_reference.md). Index cleanup: `python src/scripts/cleanup_local_indexes.py` (dry-run; `--execute` to delete).

---

### Week 0 checklist

- [x] Create local project directory and Python virtual environment (`.venv` + `requirements.txt` freeze)
- [x] Configure Cursor workspace (incl. project context rules)
- [x] Initial structure: `src/`, `data/`, `notebooks/`, `docs/`
- [x] `README.md`, `PROJECT_SPEC.md`, `.gitignore`
- [x] Local `.env` for secrets (gitignored); committed **`.env.example`** as a template (copy to `.env` and fill in values)
- [x] `pyproject.toml` (package metadata + optional dependency groups)
- [x] Python **≥ 3.11** (workspace-driven; playbook version pins are reference only)
- [x] Architecture documented
- [ ] GitHub remote / portfolio publish (your timeline)
- [ ] “First commit” / branch hygiene (your timeline)

---

### Week 1 — implemented (dense RAG index + retrieval)

**Pipeline:**

```text
BeIR/fiqa corpus → normalize → chunk → embed → Qdrant → dense search
```

| Component | Location | Notes |
|-----------|----------|--------|
| Corpus loader | `src/app/ingestion/corpus.py` | `BeIR/fiqa`, configurable limit |
| Normalize | `src/app/ingestion/normalize.py` | `_id` → `doc_id`, title/text fields |
| Chunking | `src/app/ingestion/chunking.py` | Word-based size/overlap; **tail-only** skip when a trailing window is ≤ overlap width |
| Schemas | `src/app/ingestion/schemas.py` | `DocumentRecord`, `ChunkRecord`, `SearchHit` |
| Embeddings | `src/app/retrieval/embeddings.py` | `EmbeddingProvider`: Sentence Transformers + Cohere |
| Vector store | `src/app/retrieval/qdrant_store.py` | Local path or in-memory; `close()` on shutdown; payload includes `doc_id`, `chunk_id`, `doc_title`, `chunk_text` |
| Settings | `src/app/config/settings.py` | `pydantic-settings` from `.env` |
| Ingest CLI | `src/scripts/ingest.py` | Default **`--recreate`** (delete/rebuild Qdrant + BM25); `--no-recreate` only if manifest + point count match; writes `*.manifest.json`; closes Qdrant client after run |
| Query CLI | `src/scripts/query.py` | Retrieval-only: dense, `bm25`, or `hybrid` (RRF); prints top-k hits + scores (no LLM) |

**Default embedding model:** `all-MiniLM-L6-v2` (`sentence_transformer`). **Default index:** `data/qdrant_local`, collection `fiqa_sample`, 1k-doc sample (`fiqa_corpus_limit`).

---

### Week 2 — implemented (grounded generation + FastAPI)

**Pipeline:**

```text
Question → retrieve (dense | bm25 | hybrid) → prompt builder → LLM → answer + citations (from hits)
```

| Component | Location | Notes |
|-----------|----------|--------|
| LLM interface | `src/app/generation/llm.py` | `LLMProvider.generate(prompt, system=...)` |
| Ollama | `OllamaLLM` | `POST /api/chat`, non-streaming; context manager + `close()` |
| Cohere chat | `CohereLLM` | `ClientV2.chat`; same interface |
| Factory | `create_llm_provider("ollama" \| "cohere", ...)` | Cohere requires `COHERE_API_KEY` |
| Prompt builder | `src/app/generation/prompt.py` | `build_rag_prompt`; per-source `Citation reference: [doc_id:chunk_id]` in context |
| Orchestrator | `src/app/pipeline/rag.py` | `retrieve_hits()` + `answer_query()` → `RagAnswer` (default retrieval: dense) |
| FastAPI | `src/app/api/` | `create_app()`, lifespan loads Qdrant + BM25 + embedder + LLM |
| Serve CLI | `src/scripts/serve_api.py` | Uvicorn wrapper |

**Not in Week 2 scope (later weeks):** `POST /query/stream`, `POST /ingest` over HTTP, structured request logging, Docker, offline generation metrics.

---

### Week 3 — implemented (BM25 + RRF + hybrid)

**Pipeline:**

```text
Question → dense + BM25 candidates → RRF fuse → prompt → LLM → answer + citations
```

| Component | Location | Notes |
|-----------|----------|--------|
| BM25 index | `src/app/retrieval/bm25_store.py` | Same `ChunkRecord` metadata as Qdrant; pickle under `data/bm25_indexes/<collection>.pkl` |
| RRF fusion | `src/app/retrieval/fusion.py` | `reciprocal_rank_fusion` keyed by `(doc_id, chunk_id)` |
| Hybrid search | `src/app/retrieval/hybrid.py` | `hybrid_search_text` — dense + BM25 candidate pools, then RRF |
| Retrieval router | `src/app/pipeline/rag.py` | `retrieve_hits()` + `answer_query(..., retrieval_method=..., bm25_index=...)` |
| API resources | `src/app/api/resources.py` | Loads BM25 at startup when pickle exists; `bm25_ready()` for readiness |
| Query CLI | `src/scripts/query.py` | `--method sentence_transformer \| cohere \| bm25 \| hybrid` |
| Evaluation | `src/app/evaluation/retrieval_benchmark.py` | Fixed query set in `data/benchmarks/fiqa_sample_query_set.json`; latency + overlap metrics |
| Benchmark notebook | `notebooks/retrieval_benchmark.ipynb` | Latency report (warm-up + repeats, p50/p95) + exploratory snapshot |

**Week 3 stretch:** weighted RRF implemented in Week 4 (`fusion_weights.py`, `weighted_hybrid` retrieval path).

---

### Week 4 — implemented (rerank + qrels evaluation)

**Index:** Full BEIR FiQA (`FIQA_CORPUS_LIMIT=0`, collection `fiqa`, ~111k chunks). Development slice remains `fiqa_sample` (1k docs).

| Component | Location | Notes |
|-----------|----------|--------|
| Reranker | `src/app/retrieval/reranker.py` | Local cross-encoder (default); optional Cohere Rerank |
| Optional rerank path | `src/app/pipeline/rag.py` | `retrieve_with_optional_rerank`; API `use_rerank` |
| Qrels + metrics | `src/app/evaluation/fiqa_qrels.py`, `retrieval_metrics.py`, `retrieval_eval.py` | NDCG, MRR, MAP, P/R @k |
| Eval runners | `src/scripts/evaluate.py`, `evaluate_compare.py`, `prepare_fiqa_eval.py` | Validation/test JSON under `data/benchmarks/` |
| Experiment program | `src/scripts/run_week4_experiments.py` | Exp 1–5; artifacts in `data/benchmarks/week4_exp*.json` |
| Weight tuning | `src/app/evaluation/weight_tuning.py` | Validation grid-search → `week4_fusion_weights.json` |
| Reproducible RAG eval | `src/app/evaluation/eval_llm.py` | Stub LLM; citations from hits in eval runs |
| Error analysis | `src/scripts/analyze_rerank_dense.py` | Exp 4 vs Exp 5 **dense** rerank diagnosis |

**Headline (full-corpus test, Exp 4):** **dense** NDCG@10 ≈ **0.37**; weighted hybrid ≈ dense; BM25 ≈ **0.19**. **Exp 5** local CE rerank: large BM25 gains, ~**5–6×** dense latency; net dense NDCG@10 flat vs Exp 4 (see program report).

---

### Quick start (Weeks 1–4)

**1. Install dependencies** (from repo root):

`requirements.txt` pins **third-party** packages only. Install **this app** in a second step (local `src/` or GitHub).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# A) Frozen workspace (notebooks, Jupyter stack, etc.)
pip install -r requirements.txt

# B) This repo from current checkout (pyproject.toml / src/app)
pip install -r requirements-editable.txt
# same as: pip install -e ".[embeddings,hosted,lexical,api,ingestion,dev,notebooks]"

# C) Optional: app from GitHub instead of local tree (e.g. another machine)
# pip install -r requirements-git.txt
```

Minimal install without the full freeze (CI or lightweight venv):

```powershell
pip install -e ".[ingestion,embeddings,api,hosted,lexical,dev]"
```

**2. Configure environment**

Copy the committed template and edit locally (never commit `.env`):

```powershell
copy .env.example .env   # Windows
# cp .env.example .env     # macOS/Linux
```

`.env.example` lists the variables the app reads via `Settings`; add secrets (e.g. `COHERE_API_KEY`) and overrides in `.env`. Common keys:

```env
# Embeddings (Week 1)
EMBEDDING_METHOD=sentence_transformer
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM (Week 2) — pick one provider
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest

# Optional hosted LLM
# LLM_PROVIDER=cohere
# COHERE_API_KEY=...
# COHERE_CHAT_MODEL=command-r-08-2024

# Qdrant (Week 1)
QDRANT_PATH=data/qdrant_local
COLLECTION_NAME=fiqa_sample

# BM25 pickles (Week 3; written by ingest)
BM25_INDEX_DIR=data/bm25_indexes
```

**3. Build the index** (Ollama not required for ingest):

```powershell
# Default: delete and recreate Qdrant + BM25 (recommended when corpus/chunking/embeddings change):
python src/scripts/ingest.py

# Reuse existing Qdrant collection only when configuration is unchanged:
python src/scripts/ingest.py --no-recreate
```

`--no-recreate` is safe only when **`--limit`**, **chunk size/overlap**, **embedding method/model**, and resulting **document/chunk counts** match the saved `*.manifest.json` next to the BM25 pickle (and Qdrant point count matches). Otherwise use default `--recreate`.

`data/qdrant_local/` and `data/bm25_indexes/` are **gitignored** local indexes. They are not shipped in the repo — run ingest on each machine (same as Qdrant). BM25 files are **pickle**; only load indexes you built locally or otherwise trust (see `Bm25ChunkIndex.load` in `bm25_store.py`).

**4. Retrieval smoke test:**

```powershell
# Dense (default)
python src/scripts/query.py "What are broker fees?" --top-k 3

# Lexical only
python src/scripts/query.py "broker fees" --method bm25 --top-k 3

# Hybrid (RRF over dense + BM25)
python src/scripts/query.py "What are broker fees?" --method hybrid --top-k 3
```

**5. Run API** (Ollama running if `LLM_PROVIDER=ollama`, or Cohere key if `cohere`):

```powershell
python src/scripts/serve_api.py --reload
```

- Liveness: `GET http://127.0.0.1:8000/health`
- Readiness: `GET http://127.0.0.1:8000/ready` (Qdrant collection + **BM25 index loaded** + configured LLM backend)
- Interactive docs: `http://127.0.0.1:8000/docs`
- RAG query: `POST http://127.0.0.1:8000/query` with JSON `{"question": "...", "top_k": 3, "method": "dense"}` — `method` may be `dense`, `bm25`, or `hybrid` (default `dense`)

**6. Run unit tests:**

```powershell
pytest src/tests/unit
# optional: verbose + stop on first failure
pytest src/tests/unit -v --tb=short
```

All tests are **unit** scope (mocked Qdrant/LLM where needed); qrels metrics and eval helpers have unit coverage. Full index integration eval runs via scripts, not CI yet.

---

### Configuration reference (`Settings`)

Environment variables map to fields in `src/app/config/settings.py` (case-insensitive). Common keys:

| Variable | Default | Purpose |
|----------|---------|---------|
| `EMBEDDING_METHOD` | `sentence_transformer` | `sentence_transformer` or `cohere` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | ST model name |
| `COHERE_API_KEY` | — | Cohere embeddings and/or chat |
| `COHERE_EMBED_MODEL` | `embed-v4.0` | Cohere embedding model |
| `LLM_PROVIDER` | `ollama` | `ollama` or `cohere` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API |
| `OLLAMA_MODEL` | `mistral:latest` | Ollama model tag |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | HTTP timeout for Ollama |
| `COHERE_CHAT_MODEL` | `command-r-08-2024` | Cohere chat model |
| `FIQA_CORPUS_LIMIT` | `1000` | Ingest sample size |
| `CHUNK_SIZE` | `100` | Chunk size (words) |
| `CHUNK_OVERLAP_FRACTION` | `0.2` | Overlap as fraction of chunk size |
| `QDRANT_PATH` | `data/qdrant_local` | Empty string → in-memory Qdrant |
| `COLLECTION_NAME` | `fiqa_sample` | Qdrant collection name |
| `BM25_INDEX_DIR` | `data/bm25_indexes` | Directory for `<collection>.pkl` (created by ingest; **not committed**) |

Full list and design notes: [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

---

# 2. Technology Stack

## Development

| Component | Planned Technology |
|---|---|
| IDE | Cursor |
| Language | Python |
| Environment | Python virtual environment |
| LLM - local | Ollama |
| LLM - hosted | Cohere and/or OpenAI |
| Embeddings | Hugging Face / Sentence Transformers initially |
| Vector database | **Qdrant** |
| Lexical retrieval | BM25 |
| Reranking | Local cross-encoder and/or Cohere Rerank |
| API | FastAPI |
| Evaluation | Custom evaluation harness + selected RAG evaluation methods |
| Observability | Langfuse / OpenTelemetry |
| Packaging | Docker |
| CI/CD | GitHub Actions |
| Application hosting | Render |
| Vector DB hosting | Qdrant Cloud |
| Documentation/demo site | GitHub Pages (optional) |

### Initial local architecture

Docker is **not required at the beginning**.

The initial setup is intentionally lightweight:

```text
Cursor
  │
  ├── Python
  ├── Ollama
  └── Qdrant local mode
```

Docker will be introduced later as a packaging and deployment skill, rather than as a prerequisite for learning RAG.

---

# 3. Dataset / Corpus

## Primary Corpus: FiQA-2018

The project will initially use the **FiQA-2018** financial information retrieval benchmark.

FiQA is part of the BEIR family of information-retrieval benchmarks and provides a suitable combination of:

- Financial-domain textual content
- Search queries
- Relevance judgments
- A research-oriented benchmark structure
- A corpus that can be used without building an entire data-cleaning pipeline from raw documents

The benchmark will allow the project to compare retrieval approaches using the same underlying relevance judgments.

### Planned corpus progression

```text
Phase 1
Small local subset
    ↓
Fast development + debugging

Phase 2
Larger corpus
    ↓
Performance / indexing experiments

Phase 3
Full benchmark
    ↓
Final retrieval evaluation
```

### Why finance?

Finance is a useful domain for this project because it creates realistic retrieval challenges:

- Terminology matters
- Exact phrases can matter
- Similar concepts can appear in different contexts
- Numerical and factual questions are common
- Retrieval quality is important before generation quality
- The domain aligns well with future analytical/agentic use cases

---

# 4. Target Architecture

The target system will progressively evolve toward:

```text
                           User
                            │
                            ▼
                    Web / API Client
                            │
                            ▼
                         FastAPI
                            │
                            ▼
                     RAG Orchestrator
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        Dense Retriever             BM25 Retriever
              │                           │
              │                           │
              └─────────────┬─────────────┘
                            ▼
                       RRF Fusion
                            │
                            ▼
                        Reranker
                            │
                            ▼
                     Context Builder
                            │
                            ▼
                       LLM Provider
                            │
                            ▼
                     Answer + Sources
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
           Evaluation              Observability
```

The final hosted architecture is expected to resemble:

```text
                        Internet
                           │
                           ▼
                        Render
                           │
                        FastAPI
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
      Qdrant Cloud     LLM Provider   Observability
```

---

# 5. Design Principles

This project follows a few principles throughout development.

## 5.1 Build primitives before abstractions

Important RAG components will first be implemented and understood directly.

Examples:

```text
Chunking
Embedding
Vector search
BM25 retrieval
RRF
Reranking
Prompt construction
Evaluation
```

Frameworks and convenience libraries can be introduced where useful, but they should not replace understanding of the underlying mechanism.

---

## 5.2 Prefer interfaces over provider-specific code

The application should be able to switch between providers without rewriting the whole pipeline.

Conceptually:

```text
EmbeddingProvider
    ├── HuggingFaceEmbedding
    └── CohereEmbedding

LLMProvider
    ├── OllamaLLM
    ├── CohereLLM
    └── OpenAILLM

RerankerProvider
    ├── LocalCrossEncoder
    └── CohereReranker
```

This enables inexpensive local development while retaining the ability to use hosted models in the public application.

---

## 5.3 Measure before optimizing

Whenever a major RAG component changes, the project should attempt to answer:

> Did this actually improve the system?

Experiments should capture both quality and operational metrics.

---

## 5.4 Production concerns are part of the project

The application is not considered complete when a notebook generates a good answer.

The project will progressively address:

```text
Configuration
Testing
Error handling
Security
Logging
Observability
Latency
Cost
API design
Containerization
CI/CD
Deployment
```

---

# 6. Planned Experiments

Experiments are a core part of the project, not optional documentation.

## Experiment 1 — Chunking

Compare:

- Chunk size
- Chunk overlap
- Chunking strategy

Example matrix:

```text
Chunk size: 256 / 512 / 768 / 1024
Overlap:    0 / 50 / 100 / 150
```

Measure retrieval quality and latency.

---

## Experiment 2 — Embeddings

Compare at least two embedding approaches where practical.

Potential candidates:

```text
Local Hugging Face / Sentence Transformer
Cohere embedding model
```

Measure:

- Recall@K
- MRR
- Retrieval latency
- Indexing cost/time

---

## Experiment 3 — Dense vs BM25 vs Hybrid

Compare:

```text
Dense retrieval
BM25
Dense + BM25
Dense + BM25 + RRF
```

Measure:

- Recall@K
- MRR
- nDCG where appropriate
- Latency

The purpose is to understand where lexical retrieval helps beyond semantic search.

---

## Experiment 4 — Reranking

Compare:

```text
Dense only
Hybrid
Hybrid + reranker
```

Measure:

- Retrieval quality
- End-to-end latency
- Candidate count
- Final context size

---

## Experiment 5 — Corpus Scale

Compare system behavior on increasing corpus sizes.

Potential levels:

```text
5K
10K
25K
Full benchmark
```

Measure:

- Indexing time
- Memory footprint where practical
- Retrieval latency
- Recall
- MRR

---

## Experiment 6 — Generation Models

Compare local and hosted models where API/quota constraints permit.

Potential setup:

```text
Ollama local model
Cohere
OpenAI
```

Measure:

- Answer quality
- Faithfulness
- Relevance
- Latency
- Token usage
- Cost

---

## Experiment 7 — Context Size

Evaluate:

```text
Top 3
Top 5
Top 10
```

and optionally different reranking cutoffs.

The objective is to understand how additional context affects answer quality and latency.

---

## Experiment 8 — Prompt Variations

Compare prompt strategies for:

- Citation behavior
- Refusing unsupported answers
- Concise vs detailed answers
- Handling insufficient context

---

## Experiment 9 — Retrieval Failure Analysis

Create a failure taxonomy.

Examples:

```text
Wrong document
Correct document / wrong chunk
Too little context
Too much context
Lexical mismatch
Semantic mismatch
Ambiguous question
No-answer question
Generation hallucination
```

The goal is to move beyond aggregate metrics and understand **why** the system fails.

---

# 7. Evaluation Framework

The project will separate evaluation into three levels.

## Retrieval

Primary metrics:

```text
Recall@K
MRR
Precision@K where appropriate
nDCG where appropriate
```

## Generation

Potential metrics:

```text
Faithfulness
Answer relevance
Context precision
Context recall
```

## System

Operational metrics:

```text
P50 latency
P95 latency
Error rate
Token usage
Estimated cost
Throughput where measurable
```

The evaluation dataset will eventually contain a reproducible set of benchmark queries and expected/relevant evidence.

---

# 8. HTTP API

## Implemented (Weeks 2–3)

Served by `src/app/api/` (`create_app()`). Local entrypoint: `python src/scripts/serve_api.py`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | **Liveness** — process up; no dependency checks |
| `GET` | `/ready` | **Readiness** — Qdrant + BM25 index + LLM backend (`503` if any check fails) |
| `POST` | `/query` | **RAG** — body: `question`, optional `top_k` (1–20, default 3), optional `method` (`dense` \| `bm25` \| `hybrid`, default `dense`) |

**`GET /ready` response (200):**

```json
{
  "status": "ready",
  "qdrant_collection": true,
  "bm25_index": true,
  "retrieval_index_aligned": true,
  "llm_provider": "ollama",
  "llm_ready": true
}
```

**`GET /ready` failure (503)** includes `qdrant_collection`, `bm25_index`, `retrieval_index_aligned`, `llm_provider`, and `llm_ready` in `detail`. Readiness requires BM25 + manifest + Qdrant to agree (counts, vector dimension, `ingest_generation_id`) and the manifest to match the API’s embedding method/model/dimension and ingest settings. A destructive ingest (`--recreate`) clears on-disk artifacts first so a partial failed run stays not-ready.

For `LLM_PROVIDER=cohere`, `llm_ready` means `COHERE_API_KEY` is configured (no live Cohere ping on readiness).

**`POST /query` response shape (200):**

```json
{
  "answer": "<grounded natural-language answer>",
  "citations": [
    {
      "document_id": "<BEIR doc_id>",
      "chunk_id": 0,
      "title": "<doc_title or empty string>"
    }
  ],
  "retrieval": {
    "method": "dense",
    "top_k": 3
  }
}
```

### Example — `POST /query` (FiQA sample index)

**Request:**

```json
{
  "question": "What is Broker Fees?",
  "top_k": 3,
  "method": "hybrid"
}
```

**Response** (local run; answer text depends on LLM and retrieved chunks):

```json
{
  "answer": "Based on the provided sources, I can explain the concept of broker fees and how they are structured. \n\nBroker fees, in the context of financial transactions, refer to the charges or commissions that retail brokers impose on their clients for various services. These fees are a way for brokers to cover their operational costs and generate profits. \n\nRetail brokers, as mentioned in Source 2, are generally not members of exchanges and route orders to market makers who are exchange members. Discount brokers, which charge lower fees, typically route orders to market makers and have lower operational costs compared to full-service brokers. \n\nThe fees charged by brokers can include:\n- Clearing fees: These are costs associated with processing and settling trades.\n- Regulation and licensing fees: Brokers and market makers must pay registration and licensing fees to regulatory bodies like FINRA.\n- Technology costs: Maintaining and updating technological infrastructure is an expense for both brokers and market makers.\n- Human capital: The cost of employing staff and providing training.\n- Exchange access fees: Market makers may have to pay fees for access to exchanges, although these are often not passed on to routing brokers.\n- Transaction fees: Regulatory bodies like the SEC and FINRA charge transaction fees, which are sometimes passed on to customers.\n\nIt's important to note that the structure and amount of broker fees can vary, and some brokers may offer different fee models or waive certain fees for specific types of trades or clients. The example given in Source 1 about the buying and selling of money illustrates how businesses, including brokers, generate profits by buying at one price and selling at a higher price.",
  "citations": [
    {
      "document_id": "7411",
      "chunk_id": 0,
      "title": ""
    },
    {
      "document_id": "4883",
      "chunk_id": 0,
      "title": ""
    },
    {
      "document_id": "4883",
      "chunk_id": 2,
      "title": ""
    }
  ],
  "retrieval": {
    "method": "hybrid",
    "top_k": 3
  }
}
```

`method: "bm25"` or `"hybrid"` without a loaded BM25 index returns **422** with a clear error (index missing — re-run ingest).

Citations are derived from **retrieval hits** (`doc_id` / `chunk_id` / `doc_title` in Qdrant), not from parsing the LLM output. The model may refer to “Source 1” / “Source 2” in prose; API citations remain the structured list above. Ingest remains a **CLI** (`src/scripts/ingest.py`), not an HTTP route yet.

OpenAPI / Swagger UI: `/docs` (provided by FastAPI).

## Planned (later weeks)

```text
POST /ingest
POST /query/stream          # SSE — Week 6
POST /ingest/async
GET  /ingest/jobs/{job_id}
```

`POST /query` response `retrieval` includes `method`, `top_k`, optional `reranker` name, and `candidates_retrieved` when `use_rerank=true`.

---

# 9. Repository Structure

**As of Week 4** (✅ = present in this repo):

```text
RAG/
│
├── src/
│   ├── app/
│   │   ├── api/              ✅ FastAPI routes, schemas, lifespan resources
│   │   ├── config/           ✅ Settings (pydantic-settings)
│   │   ├── ingestion/        ✅ FiQA load, normalize, chunk, schemas
│   │   ├── retrieval/        ✅ Embeddings, Qdrant, BM25, RRF fusion, hybrid
│   │   ├── generation/       ✅ LLM providers + prompt builder
│   │   ├── pipeline/         ✅ answer_query orchestrator
│   │   ├── evaluation/       ✅ Benchmark (Week 3); qrels metrics, Week 4 experiments, error analysis
│   │   └── observability/    ⬜ Week 7+
│   ├── scripts/
│   │   ├── ingest.py         ✅ Build Qdrant + BM25 indexes (`--full-corpus`, `--no-recreate`)
│   │   ├── query.py          ✅ Dense / BM25 / hybrid CLI; `--rerank`
│   │   ├── serve_api.py      ✅ Uvicorn entrypoint
│   │   ├── evaluate.py       ✅ Single-method qrels eval
│   │   ├── evaluate_compare.py ✅ Multi-method comparison report
│   │   ├── prepare_fiqa_eval.py ✅ Build `fiqa_qrels_{split}.json`
│   │   ├── run_week4_experiments.py ✅ Exp 2–5 runner + reports
│   │   └── analyze_rerank_dense.py ✅ Dense rerank error analysis (Exp 4 vs 5)
│   └── tests/
│       └── unit/             ✅ 162 unit tests (no integration eval dirs yet)
│
├── data/                     # Local Qdrant path, FiQA artifacts (gitignored as needed)
├── notebooks/                # code_dev.ipynb — experiments, not canonical app code
├── docs/reports/               ✅ week3–week4 experiment + eval reports
├── docs/                       # Playbook PDFs, plans
├── configs/                    ⬜ planned
├── frontend/                   ⬜ Week 9 demo UI
├── .github/workflows/          ⬜ Week 8 CI
│
├── pyproject.toml
├── requirements.txt            # Full workspace freeze (incl. FastAPI, notebooks)
├── PROJECT_SPEC.md
├── .env.example                # committed template (copy → .env)
├── .env                        # local secrets/overrides (gitignored)
└── README.md
```

Directory names may change as implementation experience suggests better boundaries.

---

# 10. Development Roadmap

## Week 0 — Foundation

**Goal:** establish the repository and development environment.

Deliverables:

- GitHub repository
- Cursor project
- Python environment
- Repository structure (`src/` skeleton for application code)
- README
- `pyproject.toml` and local `.env` configuration
- `PROJECT_SPEC.md`
- Qdrant local verification
- Ollama verification
- Initial FiQA data setup
- Architecture document

---

## Week 1 — Baseline RAG ✅

```text
Documents
   ↓
Chunking
   ↓
Embedding
   ↓
Qdrant
   ↓
Dense retrieval
```

Deliverables (done — see §1):

- [x] Document loader (`BeIR/fiqa`)
- [x] Chunking implementation
- [x] Embedding interface (Sentence Transformers + Cohere)
- [x] Qdrant abstraction
- [x] `ingest.py` / `query.py` CLIs
- [x] Unit tests for ingestion + retrieval

---

## Week 2 — Generation + API ✅

```text
Question
   ↓
Retriever
   ↓
Prompt
   ↓
LLM
   ↓
Answer + citations
```

Deliverables (done — see §1):

- [x] Prompt builder
- [x] LLM provider interface
- [x] Ollama provider
- [x] Cohere chat provider (hosted)
- [x] FastAPI application + lifespan
- [x] `POST /query`
- [x] `GET /health`, `GET /ready`
- [x] `answer_query` orchestrator
- [x] Unit tests (LLM, prompt, pipeline, API)
- [x] Graceful Qdrant client `close()` on API shutdown

---

## Week 3 — Hybrid Retrieval ✅

Deliverables:

- [x] BM25 implementation (`Bm25ChunkIndex`, ingest persistence)
- [x] Dense + sparse retrieval (`query.py`, `retrieve_hits`)
- [x] RRF (`reciprocal_rank_fusion`)
- [x] Hybrid end-to-end (`hybrid_search_text`, API `method`, `/ready` BM25 check)
- [x] Hybrid benchmark notebook (`notebooks/retrieval_benchmark.ipynb`)
- [x] Retrieval comparison report (`docs/reports/week3_retrieval_comparison.md`; regenerated by notebook)
- [x] Weighted RRF (Week 4 eval path)

---

## Week 4 — Reranking + Evaluation ✅

Deliverables:

- [x] Reranker interface
- [x] Local cross-encoder
- [x] Optional Cohere reranker
- [x] FiQA qrels eval datasets + retrieval metrics
- [x] Retrieval evaluation runner (`evaluate.py`, `run_week4_experiments.py`)
- [x] Full-corpus experiment program + dense rerank error analysis

---

## Week 5 — Experimentation sprint ✅

Deliverables (see [`RAG_PROJECT_ROADMAP.md`](docs/RAG_PROJECT_ROADMAP.md) E3–E9):

- [x] Candidate diversity / duplicate-chunk policy before rerank (E3)
- [x] Chunk size and overlap sweeps (E6–E7) → **200w / 20%**
- [x] Embedding model (E8) → **BGE-base**; paragraph chunking (E9) → reject (tie)
- [x] [`rag_experiment_results.md`](docs/reports/rag_experiment_results.md), sheets, [`rag_retrieval_lessons.md`](docs/reports/rag_retrieval_lessons.md)
- [x] Frozen **retrieval profile** for Week 6 API wiring (eval index: `fiqa_cs200_bge_base_en_v1_5`)

---

## Week 6 — Production API Engineering

Deliverables:

- Serve **frozen production retrieval** on `/query` (experimental methods CLI/eval-only)
- Configuration management, error handling, validation, request IDs
- Authentication approach, streaming (`POST /query/stream`), async ingestion design
- Security hardening and API integration tests

---

## Week 7 — Observability

Deliverables:

- Structured logs
- Request tracing
- Retrieval timing
- Reranking timing
- LLM timing
- Token/cost tracking
- Langfuse/OpenTelemetry integration
- Evaluation reporting

---

## Week 8 — Docker + CI/CD

Deliverables:

- Dockerfile
- Docker Compose setup
- Containerized local application
- GitHub Actions CI
- Linting
- Automated tests
- Automated evaluation
- Docker image build/publish workflow

---

## Week 9 — Deployment + Public Demo

Deliverables:

- Qdrant Cloud
- Render deployment
- Environment variables/secrets
- Production configuration
- Public API
- Simple web UI
- Public demo
- Deployment documentation
- Screenshots/GIF/video for README

---

# 11. Public Demo Goal

The final demo should allow a visitor to:

1. Open the application
2. Ask a financial question
3. See the generated answer
4. See supporting sources/citations
5. Optionally inspect retrieval metadata
6. Experience streaming generation

A simple UI is sufficient.

The emphasis is on demonstrating the **RAG engineering system**, not front-end design.

---

# 12. CI/CD Goal

The intended GitHub workflow is:

```text
Developer
   ↓
git push / Pull Request
   ↓
GitHub Actions
   ├── Lint
   ├── Type checks
   ├── Unit tests
   ├── Integration tests
   └── RAG evaluation
             ↓
       Quality gate
             ↓
        Build Docker
             ↓
          Deploy
```

Exact deployment mechanics may evolve based on the hosting platform.

---

# 13. Documentation Strategy

The repository should contain enough documentation that another developer can:

```text
Clone repository
      ↓
Configure environment
      ↓
Install dependencies
      ↓
Load corpus
      ↓
Run application
      ↓
Run evaluation
      ↓
Run tests
      ↓
Understand architecture
```

Planned documentation:

```text
README.md
PROJECT_SPEC.md
ARCHITECTURE.md
EVALUATION.md
DEPLOYMENT.md
EXPERIMENTS.md
CONTRIBUTING.md  (optional)
```

---

# 14. Blog Series

The project is intended to become a technical writing series.

## Article 1

**I Built a RAG System From Scratch: What Actually Happens Between a Question and an Answer?**

Topics:

- ingestion
- chunking
- embeddings
- Qdrant
- retrieval
- prompt construction
- generation
- citations
- FastAPI

## Article 2

**Dense Retrieval vs BM25 vs Hybrid Search: What My RAG Experiments Found**

Topics:

- dense retrieval
- lexical retrieval
- RRF
- reranking
- retrieval benchmarks
- failure analysis
- latency/quality tradeoffs

## Article 3

**From RAG Prototype to Production: Evaluation, Observability, Docker and CI/CD**

Topics:

- evaluation
- LLM-as-judge
- observability
- Docker
- GitHub Actions
- deployment
- operational considerations

These articles should use the project's **actual measured results**, not fabricated or illustrative benchmark numbers.

---

# 15. LinkedIn Publishing Strategy

The LinkedIn post should focus on the journey and engineering lessons rather than simply announcing a GitHub repository.

Potential content:

```text
What I built
Why I built it
What I learned
Most surprising experiment
Dense vs BM25 vs hybrid
Reranking impact
Production challenges
Public demo
GitHub repository
Technical article
```

A short screen recording/GIF of the application can make the final post substantially easier to understand.

---

# 16. Future Extension — Agentic RAG

This project is deliberately designed to become the foundation for a second project.

The current architecture:

```text
User
 ↓
Retriever
 ↓
LLM
 ↓
Answer
```

can eventually become:

```text
User
 ↓
Agent
 ↓
Query Planning
 ↓
┌─────────────────────────────────────┐
│                                     │
│ Retrieve     Search     Calculate   │
│    │            │            │       │
└────┴────────────┴────────────┴───────┘
                  ↓
             Evidence Fusion
                  ↓
              Final Answer
```

Possible future capabilities:

- Query decomposition
- Multi-step retrieval
- Tool calling
- Financial calculation tool
- Web search
- Evidence verification
- Self-correction
- Follow-up question handling
- Agent memory
- Multi-hop reasoning

The existing:

```text
Qdrant
BM25
Reranker
Evaluation dataset
Observability
FastAPI
```

should be reusable rather than rebuilt from scratch.

---

# 17. What This Project Is Intended to Demonstrate

By completion, the repository should demonstrate capability across:

### RAG / GenAI

- LLMs
- Embeddings
- Vector search
- Hybrid retrieval
- Reranking
- Prompt engineering
- Evaluation

### AI Engineering

- Provider abstraction
- API design
- Production architecture
- Testing
- Observability
- Latency/cost analysis

### MLOps / Platform

- Docker
- CI/CD
- Cloud deployment
- Configuration management
- Monitoring

### Data Science

- Experimental design
- Benchmarking
- Metrics
- Error analysis
- Tradeoff analysis

The project is intentionally positioned at the intersection of these areas.

---

# 18. Definition of Done

The project will be considered complete when all of the following are true.

## Core RAG

- [x] Corpus successfully indexed (local 1k FiQA sample)
- [x] Dense retrieval works
- [x] BM25 works
- [x] Hybrid retrieval works (API + CLI + benchmark report)
- [x] Reranking works (cross-encoder + API `use_rerank`; optional Cohere)
- [x] LLM generation works
- [x] Citations are returned

## Evaluation

- [x] Reproducible retrieval benchmark exists (fixed query set + notebook)
- [x] Retrieval metrics are implemented (NDCG/MRR/MAP with qrels — Week 4)
- [ ] Generation metrics are implemented (stub LLM + citations in retrieval eval only)
- [x] Baseline vs improved pipeline is documented (Week 3 + Week 4 program report)
- [x] Failure analysis is documented (dense rerank error analysis in `week4_exp5_test_rerank.md`)

## Engineering

- [x] FastAPI API works (`/health`, `/ready`, `/query`)
- [x] Request validation (Pydantic) + basic HTTP errors (`422`, `503`)
- [ ] Streaming works (`POST /query/stream` — Week 6)
- [x] Unit tests exist (**162** in `src/tests/unit/`; integration tests later)
- [ ] Structured logging exists (Week 7)
- [x] Configuration is externalized (`Settings` + `.env`)

## Productionization

- [ ] Docker image builds
- [ ] Application runs through Docker
- [ ] GitHub Actions CI works
- [ ] Automated evaluation works
- [ ] Qdrant Cloud works
- [ ] Public deployment works
- [ ] Public demo works

## Portfolio

- [ ] README polished
- [ ] Architecture diagram added
- [x] Experiment results documented (Week 4 `docs/reports/week4_*.md`)
- [ ] Screenshots/demo GIF added
- [ ] Blog articles published
- [ ] LinkedIn post published
- [ ] Future Agentic RAG roadmap documented

---

# 19. Working Rules for the Project

### Rule 1 — Do not clone implementation

The reference repository is a source of architectural ideas and production concerns, not code to copy.

### Rule 2 — Commit frequently

Use small commits that describe one meaningful change.

Example:

```text
feat: add qdrant document store
feat: implement bm25 retriever
feat: add reciprocal rank fusion
test: add retrieval evaluation cases
```

### Rule 3 — Record experiments immediately

Do not postpone experiment notes until the end.

Record:

```text
Experiment
Hypothesis
Configuration
Metric
Result
Conclusion
Next action
```

### Rule 4 — Keep a known-good baseline

Every major optimization should be compared against the previous baseline.

### Rule 5 — Favor understanding over feature count

A smaller system that can be explained end-to-end is more valuable than a large system that depends on opaque abstractions.

---

# 20. How We Will Work Together

This project is expected to be completed in **small, resumable increments** because development time will be interrupted by work and other commitments.

At each stage, the working pattern will be:

```text
1. Identify the next small task
2. Understand the concept
3. Implement
4. Test locally
5. Commit
6. Record what changed
7. Move to the next task
```

For unfamiliar infrastructure — such as:

```text
Qdrant Cloud
Docker
GitHub Actions
Render
Secrets / environment variables
Observability
```

the implementation should be done step-by-step with explicit commands, explanations, expected output, and troubleshooting guidance.

---

# 21. Project Philosophy

> **Build it. Measure it. Explain it. Deploy it. Write about it.**

The final artifact should not merely be a working application.

It should be a project that can be opened by another engineer and understood from:

```text
Architecture
   ↓
Implementation
   ↓
Experiments
   ↓
Evaluation
   ↓
Productionization
   ↓
Deployment
   ↓
Technical writing
```

That is the standard this project is aiming for.

