# E8 — Embedding model comparison

| Field | Value |
|-------|--------|
| **ID** | E8 |
| **Status** | **Complete** |
| **Primary variable** | Embedding model (re-ingest required) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

**Chunking:** 200w / 20% (`fiqa_cs200` baseline index). **Alternative:** `BAAI/bge-base-en-v1.5` → collection `fiqa_cs200_bge_base_en_v1_5`. Runner: `src/scripts/run_week5_e8.py`.

| Model | Collection | Dim | NDCG@10 (val) | NDCG@10 (test) | Recall@10 (val) | p50 val (ms) | Decision |
|-------|------------|----:|--------------:|---------------:|----------------:|-------------:|----------|
| `all-MiniLM-L6-v2` | `fiqa_cs200` | 384 | 0.393 | — | 0.468 | 119 | baseline |
| `BAAI/bge-base-en-v1.5` | `fiqa_cs200_bge_base_en_v1_5` | 768 | **0.418** | **0.394** | 0.506 | 286 | **Adopt** |

**Test:** winner only — BGE test NDCG@10 **0.394** vs MiniLM E6/E7 test **0.376** on same chunking.

**Decision:** **Adopt `BAAI/bge-base-en-v1.5`** for frozen retrieval — material val/test gain; **~2.4×** higher p50 retrieval latency vs MiniLM on same index size. Point production at `fiqa_cs200_bge_base_en_v1_5` or re-ingest into `fiqa` when ready.

**Guardrail:** each model has its own Qdrant + BM25 ingest.

## Insights (interpretation)

- **Effect size:** Val NDCG@10 **0.393→0.418** (+**0.025**); test **0.376→0.394** (+**0.018** vs MiniLM @200w); Recall@10 val **0.468→0.506**.
- **Mechanism:** Same 72,851 chunks — BGE retunes **all query–passage scores** (retrieval-oriented training, larger model). Recall@10 rise implies gains are not **only** reordering among the same doc set; many FiQA errors are **semantic matching**, which chunk sweeps cannot fix once size is reasonable.
- **Interaction:** Runs **on top of** E6+E7 (200w / 20%); largest incremental lift in Week 5 vs E6/E7.
- **Caveats:** ~**2.4×** p50 retrieval latency; BGE @100w not ablated; production must align `EMBEDDING_MODEL` with collection vectors.
