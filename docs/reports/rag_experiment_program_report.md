# RAG experiment program — master report

**Last updated:** 2026-10-20  
**Corpus:** BEIR FiQA (`fiqa` full index: 57,638 docs, 111,170 chunks)  
**Primary encoder (baseline):** `sentence_transformer` / `all-MiniLM-L6-v2`  
**Roadmap:** [`docs/RAG_PROJECT_ROADMAP.md`](../RAG_PROJECT_ROADMAP.md)  
**Consolidated results (template-filled):** [`rag_experiment_results.md`](rag_experiment_results.md)  
**Per-experiment sheets:** [`experiments/README.md`](experiments/README.md) · Template: [`RAG_EXPERIMENT_REPORT_TEMPLATE.md`](../RAG_EXPERIMENT_REPORT_TEMPLATE.md)

---

## 1. Production vs experimental retrieval

| Mode | Purpose | What runs |
|------|---------|-----------|
| **Production (served API)** | Stable demo and deployment | **One frozen design** — best measured **quality + latency** trade-off on held-out test |
| **Experimental (CLI / eval scripts)** | Compare hypotheses | Optional methods: `dense`, `bm25`, `hybrid`, `weighted_hybrid`, `use_rerank`, alternate chunking/embeddings, candidate policies |

**Current frozen production profile (from completed experiments):**

| Setting | Value | Rationale (test qrels, Exp 4) |
|---------|--------|-------------------------------|
| Retrieval | **`dense`** | NDCG@10 ≈ **0.370**; best quality/latency vs BM25, hybrid, weighted hybrid, dense+CE |
| Rerank | **off** | Exp 5: flat dense NDCG@10, ~5–6× mean latency |
| `top_k` | **10** | Week 4 eval protocol |
| Index | **`fiqa`** full corpus | Week 4 ingest |

Weighted hybrid (dense 0.95 / BM25 0.05) matches dense on test but adds fusion cost — **not** selected for production until a later experiment shows a material gain. BM25 and hybrid remain available for **experiments only**.

Implementation note: production defaults will be enforced via settings + API schema in **Week 6** (after the Week 5 experiment sprint). Until then, the API still accepts all retrieval methods for development.

---

## 2. Experiment catalog

| ID | Experiment | When | Status | Primary report / artifact |
|----|------------|------|--------|---------------------------|
| **E0** | Dense vs BM25 vs hybrid — latency & overlap | Week 3 | ✅ Done | Sheet: [`experiments/E0_unlabeled_latency_overlap.md`](experiments/E0_unlabeled_latency_overlap.md) · [`week3_retrieval_comparison.md`](week3_retrieval_comparison.md) |
| **E1** | Dense vs BM25 vs hybrid — **qrels** (labeled) | Week 4 | ✅ Done | [`experiments/E1_labeled_retrieval_baseline.md`](experiments/E1_labeled_retrieval_baseline.md) · Exp 2/4 reports |
| **E1b** | Weighted hybrid + fusion tuning | Week 4 | ✅ Done | [`experiments/E1b_weighted_hybrid_fusion.md`](experiments/E1b_weighted_hybrid_fusion.md) |
| **E2** | Retrieval + **local CE rerank** | Week 4 | ✅ Done | [`experiments/E2_cross_encoder_rerank.md`](experiments/E2_cross_encoder_rerank.md) |
| **E2a** | Dense rerank **error analysis** | Week 4 | ✅ Done | [`experiments/E2a_rerank_error_analysis.md`](experiments/E2a_rerank_error_analysis.md) |
| **E3** | Candidate diversity / duplicate-chunk crowding before rerank | **Week 5** | ⬜ Planned | [`experiments/E3_candidate_diversity.md`](experiments/E3_candidate_diversity.md) |
| **E4** | **Jev** reranker — `rerank()` ranking only (hosted API) | Week 5 | ⬜ Planned | [`experiments/E4_jev_rerank.md`](experiments/E4_jev_rerank.md) |
| **E4b** | Jev **relevance filtering** (optional) | Week 5 | ⬜ Optional | [`experiments/E4b_jev_relevance_filtering.md`](experiments/E4b_jev_relevance_filtering.md) |
| **E5** | Rerank candidate depth (10 / 20 / 30 / 50) | Week 5 | ⬜ Planned | [`experiments/E5_rerank_candidate_depth.md`](experiments/E5_rerank_candidate_depth.md) — after E4; skip if no rerank wins |
| **E6** | Chunk size sweep | Week 5 | ⬜ Planned | [`experiments/E6_chunk_size.md`](experiments/E6_chunk_size.md) |
| **E7** | Chunk overlap sweep | Week 5 | ⬜ Planned | [`experiments/E7_chunk_overlap.md`](experiments/E7_chunk_overlap.md) |
| **E8** | Embedding model comparison | Week 5 | ⬜ Planned | [`experiments/E8_embedding_model.md`](experiments/E8_embedding_model.md) |
| **E9** | Hybrid depth / RRF *k* | Week 5 stretch | ⬜ Optional | [`experiments/E9_hybrid_rrf_sensitivity.md`](experiments/E9_hybrid_rrf_sensitivity.md) |
| **E10** | Generation + citation grounding eval | Week 6–7 | ⬜ Planned | [`experiments/E10_generation_citations.md`](experiments/E10_generation_citations.md) |
| **E11** | E2E performance / concurrency | Week 8–9 | ⬜ Planned | Deployment target |

**Week 4 numbered experiments (subset of E1/E2):** see [`week4_experiment_plan.md`](week4_experiment_plan.md) and [`week4_retrieval_program_report.md`](week4_retrieval_program_report.md).

---

## 3. Results summary (completed)

### E0 — Unlabeled benchmark (Week 3)

- **Signal:** latency p50/p95; top-k Jaccard overlap between methods.
- **Not comparable** to NDCG/MRR — different query set and metrics.
- **Takeaway:** hybrid ~2× slower than dense on 1k slice; moderate overlap between dense and hybrid rankings.

### E1 — Full-corpus labeled retrieval (Week 4)

| Split | Method | NDCG@10 (mean) | Notes |
|-------|--------|----------------|--------|
| Validation | dense | **0.379** | Exp 2 |
| Validation | BM25 | 0.210 | |
| Validation | hybrid (equal RRF) | 0.336 | Below dense |
| Test | dense | **0.370** | Exp 4 |
| Test | weighted hybrid | **0.369** | Weights 0.95 / 0.05 |
| Test | BM25 | 0.191 | |

**Decision:** Adopt **dense** as production retriever; keep hybrid/BM25 for experiments.

### E2 — Rerank ablation (Week 4, Exp 5)

| Method | Test NDCG@10 | Mean latency (ms) |
|--------|--------------|-------------------|
| dense (no rerank) | 0.370 | ~199 |
| dense + CE rerank | 0.368 | ~1116 |
| BM25 + CE rerank | 0.289 | ~1590 |

**Decision:** **Do not** enable rerank in production default; CE remains optional in eval/CLI.

### E2a — Error analysis (dense only)

- 648 test queries: **181** helped, **178** hurt, **289** unchanged (ΔNDCG@10).
- **Helped:** almost all `irrelevant_candidate_ordering` (relevant doc in top-50 pool, better @10 after CE).
- **Hurt:** ordering demotions + **duplicate chunk slots** (48 queries) — motivates **E3** in Week 5.

---

## 4. Week 5 experiment sprint (planned)

Run in roadmap order (see `RAG_PROJECT_ROADMAP.md`):

1. **E3** — per-doc chunk cap / doc-diverse candidate pool before rerank.  
2. **E4** — Jev **`rerank()`** (no score filtering) vs CE vs no rerank; record API model/version, instructions, concurrency, cost, data handling.  
3. **E4b** (optional) — Jev relevance filtering if E4 is promising.  
4. **E5** — candidate depth for the E4-winning reranker (skip if dense still wins).  
5. **E6 → E7** — chunk size, then overlap.  
6. **E8** — alternate embedding model (if budget).  
7. **E9** — optional hybrid/RRF sweeps.

**Exit criteria:** explicit adopt/reject per experiment; updated production profile if a treatment beats dense on **test** without unacceptable latency; update [`rag_experiment_results.md`](rag_experiment_results.md) and the matching sheet under [`experiments/`](experiments/README.md).

---

## 5. Artifacts index

| Path | Description |
|------|-------------|
| `data/benchmarks/fiqa_qrels_{validation,test}.json` | Eval query sets |
| `data/benchmarks/week4_exp*.json` | Per-method per-query metrics |
| `data/benchmarks/week4_fusion_weights.json` | Validation-tuned RRF weights |
| `data/benchmarks/week4_eval_protocol.json` | Frozen eval run config |
| `data/benchmarks/week4_exp5_dense_rerank_error_analysis.json` | E2a classifications |
| `docs/reports/week4_*.md` | Week 4 experiment reports |
| `docs/reports/week3_retrieval_comparison.md` | E0 |

---

## 6. How to reproduce

```powershell
# Week 4 experiment program (requires full `fiqa` index)
$env:FIQA_CORPUS_LIMIT="0"
python src/scripts/run_week4_experiments.py all --collection fiqa --prepare

# Dense rerank error analysis
python src/scripts/analyze_rerank_dense.py --collection fiqa
```

Unit tests: `pytest src/tests/unit` (metrics, fusion, eval helpers — not full index integration).
