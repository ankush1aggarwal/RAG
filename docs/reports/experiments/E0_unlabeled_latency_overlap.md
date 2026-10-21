# E0 — Unlabeled latency and ranking overlap

| Field | Value |
|-------|--------|
| **ID** | E0 |
| **Status** | Complete |
| **Week** | 3 |
| **Primary variable** | Retrieval method (dense vs BM25 vs equal-weight hybrid RRF) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §4 |

---

## 1. Question and hypothesis

- **Question:** How do dense, BM25, and hybrid RRF compare on **latency** and **top-k overlap** on a small FiQA slice before labeled eval?
- **Hypothesis:** Hybrid will be slower than dense; overlap between methods will be moderate (not identical rankings).
- **Success criteria:** Document p50/p95 latency and Jaccard overlap; **not** used for production selection (no qrels).

## 2. Control and treatments

| Arm | Description |
|-----|-------------|
| Dense | Qdrant vector search, top-k=5 |
| BM25 | Lexical index, top-k=5 |
| Hybrid | RRF fusion; 15 candidates per list before fusion |

**Fixed:** `fiqa_sample` collection, 999 docs / 1900 chunks (1k doc ingest cap), `all-MiniLM-L6-v2`, 10-query benchmark set.

## 3. Evaluation protocol

| Item | Value |
|------|--------|
| Corpus | `BeIR/fiqa`, limit 1000 docs |
| Collection | `fiqa_sample` |
| Queries | 10 (`data/benchmarks/fiqa_sample_query_set.json`) |
| Final top-k | 5 |
| Timing | 3 warm-up passes; 7 timed repeats per query; embedder loaded **outside** timed region |

## 4. Results

### Latency (ms, pooled over queries × repeats)

| Method | p50 | p95 | mean |
|--------|----:|----:|-----:|
| dense | 18.5 | 21.3 | 18.7 |
| BM25 | 23.4 | 27.7 | 23.4 |
| hybrid (RRF) | 41.9 | 46.4 | 42.1 |

### Top-k overlap (mean Jaccard @5)

| Pair | Jaccard |
|------|--------:|
| dense vs BM25 | 0.19 |
| dense vs hybrid | 0.39 |
| BM25 vs hybrid | 0.46 |

**NDCG/MRR:** not computed (unlabeled query set).

## 5. Insights (interpretation)

- **Effect size:** N/A for NDCG; hybrid p50 ~**2.2×** dense on this slice; dense vs BM25 top-5 Jaccard ~**0.19** (rankings largely disjoint).
- **Mechanism:** Establishes **cost and diversity priors** only — hybrid fuses two slow stages; overlap is moderate, so methods are not redundant.
- **Caveats:** 1k-doc index, 10 queries, top-k=5 — **not** comparable to full-corpus E1 numbers; do not cite E0 for quality decisions.

## 6. Decision

| Outcome | **Defer** production choice to E1 (labeled qrels) |
| **Rationale** | E0 establishes speed and diversity priors only; hybrid ~2.2× dense p50 on this slice. |
| **Production impact** | None directly. |
| **Limitations** | 1k-doc index and 10 queries — not comparable to full-corpus Week 4 metrics. |

## 7. Artifacts and reproducibility

| Artifact | Path |
|----------|------|
| Report | [`week3_retrieval_comparison.md`](../week3_retrieval_comparison.md) |
