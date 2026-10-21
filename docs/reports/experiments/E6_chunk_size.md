# E6 — Chunk size sensitivity

| Field | Value |
|-------|--------|
| **ID** | E6 |
| **Status** | **Complete** (validation + test for winner) |
| **Primary variable** | `chunk_size` (words) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

**Fixed overlap:** 20% (current baseline). **Baseline chunk size:** 100 words (111,170 chunks on full `fiqa`).

**Collections (one per size):** `fiqa` for 100-word (Week 4 baseline) or `fiqa_cs100` if using `--prefixed-collections-only`; `fiqa_cs200`; `fiqa_cs300`. Runner: `src/scripts/run_week5_e6.py` — **does not delete** old collections without explicit confirmation.

| Chunk size | Collection | Chunks | NDCG@10 (val) | NDCG@10 (test) | Recall@10 (val) | p50 retrieval (val) |
|-----------:|------------|-------:|--------------:|---------------:|----------------:|--------------------:|
| 100 | `fiqa` | 111,170 | 0.385 | — | 0.462 | 188.5 ms |
| 200 | `fiqa_cs200` | 72,851 | **0.393** | **0.376** | 0.468 | 126.8 ms |
| 300 | `fiqa_cs300` | 63,690 | 0.391 | — | 0.461 | 114.6 ms |

**Test:** one run for validation winner (**200** words) only (`week5_e6_cs200_test.json`). Compare test **0.376** vs E1 dense @100 words test **0.370** (same metric family; E6 re-indexed arms).

**Decision (provisional):** **Adopt chunk_size=200** at 20% overlap — best validation NDCG@10/MRR@10 and faster retrieval; test NDCG@10 improves vs Week 4 100-word baseline. **Production re-ingest** onto `fiqa` (or rename/swap collections) is a separate step — **do not delete** `fiqa` / other collections without explicit confirmation.

## Insights (interpretation)

- **Effect size:** Val NDCG@10 **0.385→0.393** (100→200w); test **0.370→0.376** (+**0.006** vs E1); 300w val **0.391** — no gain over 200w.
- **Mechanism:** Longer chunks give **more local context per vector** and fewer points to search (72k vs 111k) — modest recall/NDCG lift plus **lower** p50 (~127 ms vs ~189 ms at 100w). Failures at 100w are partly **boundary/fragmentation**, not only bad embeddings.
- **Interaction:** Required before E7/E8; E8 compares encoders **on** the 200w winner.
- **Caveats:** 100w arm reuses Week 4 `fiqa`; 200w is a **re-ingest** — tiny protocol drift vs E1 possible; 300w not run on test.

Artifacts: `data/benchmarks/week5_e6_*`; report: [`week5_e6_chunk_size.md`](../week5_e6_chunk_size.md).
