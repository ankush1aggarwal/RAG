# E6 — Chunk size sensitivity

| Field | Value |
|-------|--------|
| **ID** | E6 |
| **Status** | **Planned** (Week 5) |
| **Primary variable** | `chunk_size` (words) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

**Fixed overlap:** 20% (current baseline). **Baseline chunk size:** 100 words (111,170 chunks on full `fiqa`).

| Chunk size | Overlap | Chunks | NDCG@10 | Recall@10 | Index build | p50/p95 query |
|-----------:|---------|-------:|--------:|----------:|-------------|---------------|
| 100 | 20% | 111,170 (current) | 0.3698 (dense test, E1) | 0.4304 | (Week 4 ingest) | 197.9 / 204.5 |
| 200 | 20% | TBD | TBD | TBD | TBD | TBD |
| 300 | 20% | TBD | TBD | TBD | TBD | TBD |

**Decision:** TBD — separate collection per size; one variable at a time.
