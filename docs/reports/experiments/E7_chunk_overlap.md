# E7 — Chunk overlap sensitivity

| Field | Value |
|-------|--------|
| **ID** | E7 |
| **Status** | **Complete** |
| **Primary variable** | `chunk_overlap_fraction` |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

**Fixed chunk size:** **200** words (E6 winner). **Collections:** 20% → `fiqa_cs200` (reuse); 10% → `fiqa_cs200_ov10`; 30% → `fiqa_cs200_ov30`. Runner: `src/scripts/run_week5_e7.py`. `fiqa_cs300` removed after E6.

| Overlap | Collection | Chunks | NDCG@10 (val) | NDCG@10 (test) | Recall@10 (val) | p50 (val) |
|--------:|------------|-------:|--------------:|---------------:|----------------:|----------:|
| 10% | `fiqa_cs200_ov10` | 71,954 | 0.384 | — | 0.462 | 128.1 ms |
| 20% | `fiqa_cs200` | 72,851 | **0.393** | **0.376** | 0.468 | 127.4 ms |
| 30% | `fiqa_cs200_ov30` | 74,077 | 0.390 | — | 0.462 | 128.8 ms |

**Test:** winner only (20% overlap) — matches E6 test @200w/20% (`week5_e7_ov20_test.json`).

**Decision:** **Keep 20% overlap** at 200-word chunks — best validation NDCG@10/MRR@10; 10% and 30% do not beat it. Frozen chunking with E6: **200w / 20% / `fiqa_cs200`** (production re-ingest still pending).

## Insights (interpretation)

- **Effect size:** Val NDCG@10 10% **0.384**, 20% **0.393**, 30% **0.390**; test unchanged vs E6 (**0.376** on shared `fiqa_cs200` arm).
- **Mechanism:** Overlap is a **second-order** knob — it reduces risk that answers sit on chunk cuts; at **200w**, 20% was already the E6 default, so E7 mostly **confirms** rather than moves test metrics. 10% loses boundary redundancy; 30% adds index bulk without val win.
- **Interaction:** Stacks on E6; enables a clean E8 comparison at fixed segmentation.
- **Caveats:** Only 20% run on test; ov10/ov30 collections were experimental and later removed from disk after user-approved cleanup.
