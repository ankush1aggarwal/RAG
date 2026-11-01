# Engineering topology (Week 8)

How **local development**, **GitHub Actions**, **GHCR**, and **Week 9 cloud** fit together. The RAG pipeline diagram (ingest → Qdrant → LLM) lives in [README §2](../README.md#2-system-architecture) and [`PROJECT_SPEC.md`](../PROJECT_SPEC.md).

## Dev → CI → registry → cloud

```mermaid
flowchart TB
    subgraph LOCAL["Your Windows Laptop — No Docker"]
        A["VS Code + Git"]
        B["Python virtual environment"]
        C["FastAPI application"]
        D["Local Qdrant index + BGE model"]
        E["Ollama / LLM endpoint"]
        A --> B --> C
        C --> D
        C --> E
    end

    A -->|"Git push / Pull Request"| G

    subgraph GITHUB["GitHub Actions — Remote Ubuntu Runner"]
        G["Checkout repository"]
        H["Install Python dependencies"]
        I["Run Ruff + pytest"]
        J["Build Docker image"]
        K["Start container + smoke tests"]
        G --> H --> I --> J --> K
    end

    K -->|"Success / failure"| L["GitHub PR status"]
    K -.->|"Optional: publish image later"| M["GitHub Container Registry"]

    subgraph CLOUD["Week 9 — Cloud Deployment (Cloud Run)"]
        N["Cloud Run service (GHCR image)"]
        O["Persistent Qdrant storage or managed Qdrant"]
        P["LLM endpoint"]
        N --> O
        N --> P
    end

    M -.-> N
```

## Workflows

| Workflow | File | On PR | On `main` |
|----------|------|-------|-----------|
| **CI** | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Ruff + pytest | Same |
| **Container** | [`.github/workflows/container.yml`](../.github/workflows/container.yml) | Build + `/health` smoke | Smoke + push to **GHCR** |

## GHCR image

`ghcr.io/<github-owner>/fiqa-production-rag` — see [`docker.md`](docker.md).

## Related

- [`cloud_run.md`](cloud_run.md) — Qdrant Cloud env + Cloud Run outline  
- [`week8_sprint.md`](week8_sprint.md) — container + CI checklist  
- [`observability_guide.md`](observability_guide.md) — JSON logs on stderr  
- [`api_contract.md`](api_contract.md) — HTTP API
