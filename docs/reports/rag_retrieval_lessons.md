# Lessons from the FiQA retrieval experiment program

**Audience:** Engineers learning how to improve RAG retrieval systematically.  
**Evidence:** E0–E9 on BEIR FiQA (document-level qrels); full numbers in [`rag_experiment_results.md`](rag_experiment_results.md).

**Frozen retrieval design (eval):** dense · no rerank · `top_k=10` · **200w / 20%** · **`BAAI/bge-base-en-v1.5`** · index **`fiqa_cs200_bge_base_en_v1_5`** (test NDCG@10 **0.394**).  
**Served API:** same frozen index and encoder as eval (`fiqa_cs200_bge_base_en_v1_5`, BGE-base, `top_k=10`, dense, no rerank).

---

## 1. Measure before you optimize

- **Unlabeled benchmarks (E0)** tell you latency and ranking overlap, not quality. Hybrid was ~2× slower than dense on a small slice; overlap with dense was moderate — enough to justify **labeled** eval (E1), not enough to pick a winner.
- **Lesson:** Always pair speed probes with **qrels + NDCG@10** (and document-level dedup if your index is chunked).

## 2. Pick the retrieval *family* first

- **E1:** Dense beat BM25 and equal-weight hybrid on NDCG@10 (**0.370** test) with the **best** latency among strong options.
- **E1b:** Validation-tuned weighted fusion (0.95 dense / 0.05 BM25) almost matched dense on test but cost ~**4×** p50 — reject for production.
- **Lesson:** For semantic Q&A corpora, **bi-encoder dense** is usually the right default; hybrid is an experiment, not a free lunch.

## 3. Reranking is not automatic value

- **E2:** Cross-encoder @50 moved **Recall@10** slightly but **flat** NDCG@10 (**0.368** vs **0.370**) at ~**5.6×** retrieve+rerank latency.
- **E2a:** Helps and hurts were **balanced** (~181 vs ~178 queries) — aggregate metrics hid churn. Many hurts involved **duplicate chunk slots** from the same document.
- **E3:** Pool policies that reduce duplicates did **not** improve test NDCG when rerank was on; with rerank **off**, pool policy does not matter for production.
- **Lesson:** Add a reranker only when it **wins on your primary metric on test**, not because “two-stage feels more RAG.” Error analysis explains *why* flat metrics happen.

## 4. Chunking: coarse tune, then stop

- **E6:** 100w → **200w** gave a **small** test gain (**0.370 → 0.376**) and **fewer** vectors (faster search). 300w did not beat 200w on validation.
- **E7:** Overlap 10% hurt; 30% did not beat **20%** — second-order knob once size is sane.
- **Lesson:** Sweep chunk **size** in a reasonable band (here ~100–300 words), fix overlap at your prior default, and **do not** grid-search forever — gains are often shallow.

## 5. Embeddings usually beat more chunk grid search

- **E8:** Same 200w/20% segmentation; swapping MiniLM → **BGE-base** gave the **largest** test step (**0.376 → 0.394**, +**0.018**) at ~**2.4×** retrieval p50.
- Recall@10 rose on validation — not only reordering among the same docs.
- **Lesson:** After chunking is in a good regime, **upgrade the encoder** (or train/adapt one) before chasing marginal chunk hyperparameters.

## 6. “Smarter chunking” needs signal in the text

- **E9:** Paragraph-aware merge (max 200w) produced the **same 72,851 chunks** as fixed windows on FiQA — BEIR text rarely keeps `\n\n` boundaries after normalization — **identical** validation metrics.
- **Lesson:** Structure-aware chunking only helps when the corpus **preserves** structure (HTML, PDF layout, markdown). **Probe chunk counts** before a full re-ingest.

## 7. How we chose what to ship

| Criterion | Rule used |
|-----------|-----------|
| Primary metric | NDCG@10 on **held-out test** |
| Tuning | Validation only; one test run per frozen candidate |
| Latency | Reject ties that cost multi× p50 (fusion, rerank, bigger encoder trade-offs explicit) |
| Complexity | No rerank depth sweep (E5 skipped) while rerank off |

**Cumulative test path (dense, no rerank):** **0.370** (E1, 100w MiniLM) → **0.376** (E6, 200w) → **0.394** (E8, BGE on 200w).

## 8. What this project did *not* optimize (yet)

- **Generation and citations (E10)** — retrieval was frozen first.
- **End-to-end latency** with LLM, concurrency, observability (Weeks 6–9).
- **Hybrid / RRF sweeps** — deferred after E1 showed dense wins.

## 9. Practical checklist for your next RAG eval sprint

1. Freeze corpus and qrels splits.  
2. Compare retrieval **methods** with one index setup (E1).  
3. Reject rerank/fusion unless test NDCG justifies cost (E2, E1b).  
4. Tune chunk **size**, confirm **overlap** (E6, E7).  
5. Try **one** stronger embedding with a **dedicated** index (E8).  
6. Only then try structure-aware chunking if the source has real boundaries (E9).  
7. Document **insights** per experiment, not only tables — see [`eval_protocol_reference.md`](eval_protocol_reference.md).

---

**Reporting note:** Week 5 markdown sheets briefly showed **Cand. recall = 0** because tables looked up `@1/@5/@10` while the metric lives at **`candidate_recall@50`** in JSON (`CR@50` column after the formatter fix). Use saved benchmarks for pool recall, not the old table cells.

**Indexes on disk (intended):** `fiqa` (production API) + `fiqa_cs200_bge_base_en_v1_5` (frozen retrieval eval). Cleanup helper: `python src/scripts/cleanup_local_indexes.py`.

**Benchmark artifacts retained (representative):** `fiqa_qrels_{validation,test}.json`; `week4_exp*` + error analysis; `week5_e8_bge_*`; `week5_e6_cs200_test.json`; `week5_e9_fixed_200w_bge_*` + protocols; `week5_e3_control_test.json`; BGE collection qrels under `fiqa_qrels_*_bge_base_en_v1_5.json`. Loser arms, run logs, and collection-specific qrels for deleted indexes were removed from `data/benchmarks/`.
