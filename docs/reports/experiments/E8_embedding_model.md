# E8 — Embedding model comparison

| Field | Value |
|-------|--------|
| **ID** | E8 |
| **Status** | **Planned** (Week 5) |
| **Primary variable** | Embedding model (re-ingest required) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

| Model | Dim | Build time | Index size | NDCG@10 | Recall@10 | p50/p95 retrieval | Decision |
|-------|----:|------------|------------|--------:|----------:|-------------------|----------|
| `all-MiniLM-L6-v2` (baseline) | 384 | Week 4 | full `fiqa` | 0.3698 | 0.4304 | 197.9 / 204.5 | current |
| Alternative (TBD) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Guardrail:** never compare models on mismatched indexes.

**Decision:** TBD
