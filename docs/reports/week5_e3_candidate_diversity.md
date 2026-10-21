# Week 5 — E3 candidate diversity

**Collection:** `fiqa`
**Qrels split:** `test`
**Embedding:** all-MiniLM-L6-v2
**Corpus:** `BeIR/fiqa` **full corpus** (~57k documents)
**Retriever:** dense | **Reranker:** cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2

## Pool diagnostics (mean over queries, pre-rerank)

| Arm | mean pool size | mean unique docs | mean duplicate-slot rate |
|-----|---------------:|-----------------:|-------------------------:|
| Control — top 50 dense, no cap | 50.0 | 43.4 | 0.132 |
| Overfetch 150, cap 2 per doc_id, fill to 50 | 50.0 | 44.7 | 0.107 |

## Retrieval metrics (document-level, after rerank)

| Arm | Metric | @1 | @5 | @10 |
|-----|--------|------|------|------|
| control | NDCG | 0.3534 | 0.3438 | 0.3684 |
| control | MRR | 0.3534 | 0.4290 | 0.4407 |
| control | Recall | 0.1827 | 0.3618 | 0.4382 |
| control | Cand. recall | 0.0000 | 0.0000 | 0.0000 |
| overfetch_cap2 | NDCG | 0.3534 | 0.3431 | 0.3686 |
| overfetch_cap2 | MRR | 0.3534 | 0.4288 | 0.4405 |
| overfetch_cap2 | Recall | 0.1827 | 0.3630 | 0.4408 |
| overfetch_cap2 | Cand. recall | 0.0000 | 0.0000 | 0.0000 |

## Latency

| Arm | Stage | mean (ms) | p50 (ms) | p95 (ms) |
|-----|-------|-----------|----------|----------|
| control | first-stage retrieval | 218.1 | 214.4 | 240.9 |
| control | rerank | 925.7 | 885.7 | 1296.5 |
| control | retrieve + rerank | 1143.7 | 1103.7 | 1524.6 |
| overfetch_cap2 | first-stage retrieval | 215.2 | 213.3 | 232.9 |
| overfetch_cap2 | rerank | 926.7 | 883.3 | 1318.1 |
| overfetch_cap2 | retrieve + rerank | 1141.9 | 1099.5 | 1533.1 |

**Decision (test, n=648):** **Control** — NDCG@10/MRR@10 effectively tied with `overfetch_cap2`; validation-only gain did not hold on test. Frozen pool policy: top-50 dense, no per-doc cap (for any local rerank eval). Week 5 continues with **E6–E8**; E4 Jev removed from program.

## Insights

- **ΔNDCG@10 (test):** ~**0** (0.3684 vs 0.3686) despite lower duplicate-slot rate with cap2 (~11% vs ~13%).
- **Mechanism:** E2a duplicate crowding is **real** but fixing it via overfetch+cap does not move document-level ranked metrics when CE rerank is applied — pool changes do not beat ordering noise on test.
- **Takeaway:** Do not adopt cap policies for production; keep **control** for eval reproducibility if rerank is revisited.