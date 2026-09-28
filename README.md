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

> **Current phase: Week 0 — Project setup and architecture**

### Week 0 checklist

- [ ] Create GitHub repository
- [ ] Create local project directory
- [ ] Create Python virtual environment
- [ ] Configure Cursor workspace
- [ ] Create initial project structure
- [ ] Add `README.md`
- [ ] Add `.gitignore`
- [ ] Configure local `.env` (gitignored secrets; no committed env template)
- [x] Create initial `pyproject.toml`
- [x] Decide Python version (workspace-driven; playbook pins are reference only)
- [ ] Verify local Qdrant Python setup
- [ ] Verify Ollama installation
- [ ] Select initial embedding model
- [ ] Download/sample FiQA-2018 data
- [x] Document architecture (`README.md`, `PROJECT_SPEC.md`)
- [ ] Make first Git commit

The README will be updated throughout the project as features, experiments, metrics, screenshots, and deployment details become available.

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

# 8. Planned API

The exact API may evolve, but the initial target is:

```text
GET  /health
GET  /ready

POST /ingest

POST /query
POST /query/stream
```

Potential later additions:

```text
POST /ingest/async
GET  /ingest/jobs/{job_id}
```

The API should return useful metadata, including source/citation information.

Example target response:

```json
{
  "answer": "Example answer...",
  "citations": [
    {
      "document_id": "123",
      "title": "Example source",
      "chunk_id": "456"
    }
  ],
  "retrieval": {
    "method": "hybrid",
    "reranker": "cross-encoder"
  }
}
```

---

# 9. Planned Repository Structure

The repository is expected to evolve toward:

```text
production-rag/
│
├── src/
│   ├── app/
│   │   ├── api/
│   │   ├── ingestion/
│   │   ├── retrieval/
│   │   ├── generation/
│   │   ├── evaluation/
│   │   ├── observability/
│   │   ├── config/
│   │   └── pipeline/
│   ├── scripts/
│   │   ├── ingest.py
│   │   ├── query.py
│   │   ├── evaluate.py
│   │   └── benchmark.py
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── evaluation/
│   ├── experiments/
│   └── data/
│
├── data/
│   ├── raw/
│   └── evaluation/
│
├── notebooks/
├── docs/
├── frontend/
├── configs/
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── PROJECT_SPEC.md
├── .env                 # local only (gitignored)
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

## Week 1 — Baseline RAG

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

Deliverables:

- Document loader
- Chunking implementation
- Embedding interface
- Qdrant abstraction
- Basic retrieval script
- Baseline retrieval results

---

## Week 2 — Generation + API

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

Deliverables:

- Prompt builder
- LLM provider interface
- Ollama provider
- Optional hosted provider
- FastAPI application
- Query endpoint
- Health/readiness endpoints

---

## Week 3 — Hybrid Retrieval

Deliverables:

- BM25 implementation
- Dense + sparse retrieval
- RRF
- Optional weighted RRF
- Hybrid benchmark
- Retrieval comparison report

---

## Week 4 — Reranking + Evaluation

Deliverables:

- Reranker interface
- Local cross-encoder
- Optional Cohere reranker
- Golden evaluation dataset
- Retrieval evaluation runner
- Initial generation evaluation

---

## Week 5 — Production API Engineering

Deliverables:

- Configuration management
- Error handling
- Input validation
- Request IDs
- Authentication approach
- Streaming responses
- Async ingestion approach
- Security hardening

---

## Week 6 — Observability

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

## Week 7 — Docker + CI/CD

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

## Week 8 — Deployment + Public Demo

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

- [ ] Corpus successfully indexed
- [ ] Dense retrieval works
- [ ] BM25 works
- [ ] Hybrid retrieval works
- [ ] Reranking works
- [ ] LLM generation works
- [ ] Citations are returned

## Evaluation

- [ ] Reproducible benchmark exists
- [ ] Retrieval metrics are implemented
- [ ] Generation metrics are implemented
- [ ] Baseline vs improved pipeline is documented
- [ ] Failure analysis is documented

## Engineering

- [ ] FastAPI API works
- [ ] Validation/error handling exists
- [ ] Streaming works
- [ ] Tests exist
- [ ] Structured logging exists
- [ ] Configuration is externalized

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
- [ ] Experiment results documented
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

