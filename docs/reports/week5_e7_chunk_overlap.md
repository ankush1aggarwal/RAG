# Week 5 — E7 chunk overlap sensitivity

**Chunk size (fixed):** 200 words (E6 winner)
**Qrels split:** `validation`
**Embedding:** all-MiniLM-L6-v2
**Corpus:** `BeIR/fiqa` **full corpus** (~57k documents)
**Retriever:** dense (no rerank)

## Collections

| Overlap | Qdrant collection | Points (chunks) |
|--------:|-------------------|----------------:|
| 10% | `fiqa_cs200_ov10` | 71,954 |
| 20% | `fiqa_cs200` | 72,851 |
| 30% | `fiqa_cs200_ov30` | 74,077 |

## Retrieval metrics (document-level)

| Overlap | Metric | @1 | @5 | @10 | CR@50 |
|-------|--------|------|------|------|------|
| 10% | NDCG | 0.3520 | 0.3526 | 0.3835 | — |
| 10% | MRR | 0.3520 | 0.4317 | 0.4451 | — |
| 10% | Recall | 0.1967 | 0.3695 | 0.4622 | — |
| 10% | Cand. recall | — | — | — | n/a |
| 20% | NDCG | 0.3720 | 0.3645 | 0.3925 | — |
| 20% | MRR | 0.3720 | 0.4491 | 0.4602 | — |
| 20% | Recall | 0.2070 | 0.3825 | 0.4680 | — |
| 20% | Cand. recall | — | — | — | n/a |
| 30% | NDCG | 0.3700 | 0.3621 | 0.3904 | — |
| 30% | MRR | 0.3700 | 0.4439 | 0.4559 | — |
| 30% | Recall | 0.2092 | 0.3758 | 0.4617 | — |
| 30% | Cand. recall | — | — | — | n/a |

## First-stage retrieval latency

| Overlap | mean (ms) | p50 (ms) | p95 (ms) |
|--------:|----------:|---------:|---------:|
| 10% | 129.2 | 128.1 | 139.3 |
| 20% | 128.5 | 127.4 | 137.6 |
| 30% | 130.8 | 128.8 | 143.1 |

**Validation winner (NDCG@10):** 20% overlap (`fiqa_cs200`).

**Test (20% winner, n=648):** NDCG@10 **0.376**, MRR@10 **0.453** — same index as E6 test for 200w/20%.

## Insights

- **ΔNDCG@10 (test):** ~**0** vs E6 — overlap sweep **validates** the existing 20% default, not a new test lift.
- **Val pattern:** 10% overlap underperforms (~−0.009 NDCG@10 vs 20%) — boundary coverage matters at 200w; 30% adds chunks without beating 20%.
- **Takeaway:** Tune chunk **size** before overlap; overlap is fine-tuning after size is chosen.

**Decision:** retain **20%** overlap; optional cleanup of `fiqa_cs200_ov10` / `fiqa_cs200_ov30` only with explicit confirmation.