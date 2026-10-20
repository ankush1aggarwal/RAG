# Week 4 — Retrieval evaluation (qrels) vs Week 3 baselines

**Generated:** 2026-10-19 08:17 UTC  
**Index collection:** `fiqa_sample`  
**Dense encoder:** `sentence_transformer` / `all-MiniLM-L6-v2`  
**Corpus / index:** `BeIR/fiqa` first `1000` documents  
**Qrels split:** `validation`  
**Eval dataset:** `data/benchmarks/fiqa_qrels_validation.json`  
**Retrieval depth (top-k):** `10`  
**Queries evaluated:** `15`  

## Week 4 — Labeled metrics (mean over queries)

Document-level ranking from chunk hits (first-seen `doc_id`). Graded NDCG uses BEIR relevance scores.

| Method | Metric | @1 | @5 | @10 |
|--------|--------|------|------|------|
| dense (Qdrant) | NDCG | 0.8000 | 0.8679 | 0.8679 |
| dense (Qdrant) | MRR | 0.8000 | 0.8467 | 0.8467 |
| dense (Qdrant) | MAP | 0.8000 | 0.8467 | 0.8467 |
| dense (Qdrant) | Precision | 0.8000 | 0.1867 | 0.0933 |
| dense (Qdrant) | Recall | 0.8000 | 0.9333 | 0.9333 |
| BM25 | NDCG | 0.4667 | 0.4954 | 0.5377 |
| BM25 | MRR | 0.4667 | 0.4833 | 0.5003 |
| BM25 | MAP | 0.4667 | 0.4833 | 0.5003 |
| BM25 | Precision | 0.4667 | 0.1067 | 0.0667 |
| BM25 | Recall | 0.4667 | 0.5333 | 0.6667 |
| hybrid (RRF) | NDCG | 0.4667 | 0.6282 | 0.6482 |
| hybrid (RRF) | MRR | 0.4667 | 0.5733 | 0.5807 |
| hybrid (RRF) | MAP | 0.4667 | 0.5733 | 0.5807 |
| hybrid (RRF) | Precision | 0.4667 | 0.1600 | 0.0867 |
| hybrid (RRF) | Recall | 0.4667 | 0.8000 | 0.8667 |

## Week 3 baseline reference (not directly comparable)

Week 3 [`week3_retrieval_comparison.md`](week3_retrieval_comparison.md) used a **fixed 10-query** set, **top-k=5**, and **unlabeled** signals (latency p50/p95, top-k Jaccard overlap). Those numbers measure **speed** and **ranking diversity**, not NDCG/MRR.

### Week 3 latency (ms, p50) — 1k-doc `fiqa_sample`, 10 queries

| Method | p50 |
|--------|-----|
| dense (Qdrant) | 18.5 |
| BM25 | 23.4 |
| hybrid (RRF) | 41.9 |

### Week 3 top-k overlap (mean Jaccard @5)

- Dense vs BM25: **0.19**
- Dense vs hybrid: **0.39**
- BM25 vs hybrid: **0.46**

## Interpretation

- Compare **NDCG@10 / MRR@10** across methods on the **same** qrels slice; higher is better.
- Week 3 **hybrid** had moderate Jaccard vs dense (0.39) but was **~2× slower**; Week 4 shows whether hybrid improves **labeled** retrieval on your index.
- Partial corpus indexes filter qrels to queries with ≥1 relevant doc in-index; full-corpus ingest + `prepare_fiqa_eval.py` yields the complete validation/test sets.
- Reserve **test** qrels for final reporting only.
