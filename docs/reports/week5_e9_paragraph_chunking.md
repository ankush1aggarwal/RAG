# Week 5 — E9 paragraph-aware chunking

**Embedding:** `BAAI/bge-base-en-v1.5`
**Baseline:** E8 fixed 200w / 20% overlap
**Treatment:** paragraph merge up to 200w; sub-split with 20% overlap
**Qrels split:** `validation`
**Corpus:** `BeIR/fiqa` **full corpus** (~57k documents)
**Retriever:** dense (no rerank)

## Indexes

| Arm | Collection | Points |
|-----|------------|-------:|
| fixed 200w / 20% (E8) | `fiqa_cs200_bge_base_en_v1_5` | 72,851 |
| paragraph-aware / max 200w | `fiqa_para_bge_base_en_v1_5` | 72,851 |

**Paragraph ingest probe (chunks):** 72,851

## Retrieval metrics (document-level)

| Arm | Metric | @1 | @5 | @10 | CR@50 |
|---|--------|------|------|------|------|
| fixed 200w / 20% (E8) | NDCG | 0.3980 | 0.3845 | 0.4176 | — |
| fixed 200w / 20% (E8) | MRR | 0.3980 | 0.4742 | 0.4868 | — |
| fixed 200w / 20% (E8) | Recall | 0.2193 | 0.4085 | 0.5064 | — |
| fixed 200w / 20% (E8) | Cand. recall | — | — | — | 0.6580 |
| paragraph-aware / max 200w | NDCG | 0.3980 | 0.3845 | 0.4176 | — |
| paragraph-aware / max 200w | MRR | 0.3980 | 0.4742 | 0.4868 | — |
| paragraph-aware / max 200w | Recall | 0.2193 | 0.4085 | 0.5064 | — |
| paragraph-aware / max 200w | Cand. recall | — | — | — | 0.6580 |

## First-stage retrieval latency

| Arm | mean (ms) | p50 (ms) | p95 (ms) |
|-----|----------:|---------:|---------:|
| fixed 200w / 20% (E8) | 280.8 | 277.3 | 306.0 |
| paragraph-aware / max 200w | 290.3 | 282.9 | 330.7 |

**Validation winner (NDCG@10):** fixed 200w / 20% (E8) (`fiqa_cs200_bge_base_en_v1_5`) — tied with paragraph on all reported metrics.

**Test (winner only, n=648):** NDCG@10 **0.394** (`week5_e9_fixed_200w_bge_test.json`) — matches E8 test.

**Decision:** **Reject** paragraph-aware index for frozen retrieval; **keep** `fiqa_cs200_bge_base_en_v1_5`. Treatment index **deleted** from disk (`cleanup_local_indexes.py`).

## Insights

- **ΔNDCG@10:** **0** on validation and test vs E8 — arms are metric-identical in this run.
- **Mechanism:** Probe and ingest both produced **72,851** chunks (same as E8); FiQA body text rarely retains `\n\n` paragraphs after BEIR normalization, so paragraph merge + 200w sub-split collapses to the **same segmentation** as fixed 200w/20%.
- **Takeaway:** Paragraph-aware chunking only helps when the corpus has reliable paragraph boundaries; for FiQA, E8 fixed windows remain the frozen chunking policy. No production re-ingest.