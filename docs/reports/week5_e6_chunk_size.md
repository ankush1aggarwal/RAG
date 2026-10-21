# Week 5 — E6 chunk size sensitivity

**Qrels split:** `validation`
**Overlap (fixed):** 20%
**Embedding:** all-MiniLM-L6-v2
**Corpus:** `BeIR/fiqa` **full corpus** (~57k documents)
**Retriever:** dense (no rerank)

## Collections (one index per chunk size)

| Chunk size (words) | Qdrant collection | Points (chunks) |
|-------------------:|-------------------|----------------:|
| 100 | `fiqa` | 111,170 |
| 200 | `fiqa_cs200` | 72,851 |
| 300 | `fiqa_cs300` | 63,690 |

## Retrieval metrics (document-level)

| Chunk size | Metric | @1 | @5 | @10 | CR@50 |
|----------|--------|------|------|------|------|
| 100 | NDCG | 0.3620 | 0.3580 | 0.3853 | — |
| 100 | MRR | 0.3620 | 0.4328 | 0.4463 | — |
| 100 | Recall | 0.2046 | 0.3781 | 0.4620 | — |
| 100 | Cand. recall | — | — | — | n/a |
| 200 | NDCG | 0.3720 | 0.3645 | 0.3925 | — |
| 200 | MRR | 0.3720 | 0.4491 | 0.4602 | — |
| 200 | Recall | 0.2070 | 0.3825 | 0.4680 | — |
| 200 | Cand. recall | — | — | — | n/a |
| 300 | NDCG | 0.3700 | 0.3628 | 0.3907 | — |
| 300 | MRR | 0.3700 | 0.4475 | 0.4581 | — |
| 300 | Recall | 0.2110 | 0.3793 | 0.4613 | — |
| 300 | Cand. recall | — | — | — | n/a |

## First-stage retrieval latency

| Chunk size | mean (ms) | p50 (ms) | p95 (ms) |
|-----------:|----------:|---------:|---------:|
| 100 | 190.5 | 188.5 | 204.3 |
| 200 | 127.8 | 126.8 | 136.1 |
| 300 | 115.5 | 114.6 | 123.2 |

**Validation winner (NDCG@10):** `200` words (collection `fiqa_cs200`).

**Test (winner only, n=648):** chunk_size **200** — NDCG@10 **0.376**, MRR@10 **0.453**, Recall@10 **0.447** (`week5_e6_cs200_test.json`). Week 4 dense @100w test reference: NDCG@10 **0.370**.

## Insights

- **ΔNDCG@10 (test):** +**0.006** vs E1 dense @100w (0.376 vs 0.370) — modest but consistent with val sweep.
- **Why not bigger:** 300w did not beat 200w — segmentation is in a **shallow optimum**; further word-count tuning unlikely to match encoder upgrades (see E8).
- **Latency:** Fewer chunks → faster dense search despite longer texts per vector.

_Old collections are not deleted automatically — confirm before removing any index._