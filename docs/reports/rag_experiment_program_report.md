# RAG experiment program — master report

**Last updated:** 2026-10-21  
**Corpus:** BEIR FiQA (57,638 docs)  
**Frozen index (retrieval):** `fiqa_cs200_bge_base_en_v1_5` — **72,851** chunks @ **200w / 20%**  
**Primary encoder (frozen):** `BAAI/bge-base-en-v1.5` (768-d); MiniLM baseline in experiment arms  
**Roadmap:** [`docs/RAG_PROJECT_ROADMAP.md`](../RAG_PROJECT_ROADMAP.md)  
**Consolidated results:** [`rag_experiment_results.md`](rag_experiment_results.md) · **Lessons:** [`rag_retrieval_lessons.md`](rag_retrieval_lessons.md)  
**Per-experiment sheets:** [`experiments/README.md`](experiments/README.md) · Template: [`RAG_EXPERIMENT_REPORT_TEMPLATE.md`](../RAG_EXPERIMENT_REPORT_TEMPLATE.md)

---

## 1. Production vs experimental retrieval

| Mode | Purpose | What runs |
|------|---------|-----------|
| **Production (served API)** | Stable demo and deployment | **One frozen design** — best measured **quality + latency** trade-off on held-out test |
| **Experimental (CLI / eval scripts)** | Compare hypotheses | Optional methods: `dense`, `bm25`, `hybrid`, `weighted_hybrid`, `use_rerank`, alternate chunking/embeddings, candidate policies |

**Current frozen production profile (from completed experiments):**

| Setting | Value | Rationale (test qrels) |
|---------|--------|------------------------|
| Retrieval | **`dense`** | E1: beats BM25/hybrid; E1b fusion not worth latency |
| Rerank | **off** | E2: flat NDCG@10 @50; E3 pool neutral on test |
| `top_k` | **10** | Week 4 eval protocol (**not** the current API default of 3) |
| Chunking | **200w / 20%** | E6, E7 |
| Embedding | **`BAAI/bge-base-en-v1.5`** | E8: test NDCG@10 **0.394** |
| Index | **`fiqa_cs200_bge_base_en_v1_5`** | Full corpus; legacy `fiqa` @100w MiniLM retained on disk |

Weighted hybrid (dense 0.95 / BM25 0.05) matches dense on test but adds fusion cost — **not** selected for production until a later experiment shows a material gain. BM25 and hybrid remain available for **experiments only**.

**Eval discipline:** validation for tuning; test for final metrics — [`eval_protocol_reference.md`](eval_protocol_reference.md). **Primary metric:** NDCG@10; **candidate Recall@50** on first-stage pools (in eval code from this revision).

Implementation note: **Week 6** must **enforce** this profile on `/query` (`method`, `use_rerank`, `top_k`) in settings/schema, not only in these docs. Until then the API keeps dev defaults (`top_k=3`, all methods).

---

## 2. Experiment catalog

| ID | Experiment | When | Status | Primary report / artifact |
|----|------------|------|--------|---------------------------|
| **E0** | Dense vs BM25 vs hybrid — latency & overlap | Week 3 | ✅ Done | Sheet: [`experiments/E0_unlabeled_latency_overlap.md`](experiments/E0_unlabeled_latency_overlap.md) · [`week3_retrieval_comparison.md`](week3_retrieval_comparison.md) |
| **E1** | Dense vs BM25 vs hybrid — **qrels** (labeled) | Week 4 | ✅ Done | [`experiments/E1_labeled_retrieval_baseline.md`](experiments/E1_labeled_retrieval_baseline.md) · Exp 2/4 reports |
| **E1b** | Weighted hybrid + fusion tuning | Week 4 | ✅ Done | [`experiments/E1b_weighted_hybrid_fusion.md`](experiments/E1b_weighted_hybrid_fusion.md) |
| **E2** | Retrieval + **local CE rerank** | Week 4 | ✅ Done | [`experiments/E2_cross_encoder_rerank.md`](experiments/E2_cross_encoder_rerank.md) |
| **E2a** | Dense rerank **error analysis** | Week 4 | ✅ Done | [`experiments/E2a_rerank_error_analysis.md`](experiments/E2a_rerank_error_analysis.md) |
| **E3** | Candidate diversity / duplicate-chunk crowding before rerank | **Week 5** | ✅ Done | [`experiments/E3_candidate_diversity.md`](experiments/E3_candidate_diversity.md) |
| **E4** | ~~Jev reranker (hosted API)~~ | — | **Removed** | Cost — sheet archived |
| **E4b** | ~~Jev relevance filtering~~ | — | **Removed** | Depended on E4 |
| **E5** | Rerank candidate depth (10 / 20 / 30 / 50) | Week 5 | **Skipped** | E2 rejected CE; no new reranker |
| **E6** | Chunk size sweep | Week 5 | ✅ Done | [`experiments/E6_chunk_size.md`](experiments/E6_chunk_size.md) |
| **E7** | Chunk overlap sweep | Week 5 | ✅ Done | [`experiments/E7_chunk_overlap.md`](experiments/E7_chunk_overlap.md) |
| **E8** | Embedding model comparison | Week 5 | ✅ Done | [`experiments/E8_embedding_model.md`](experiments/E8_embedding_model.md) |
| **E9** | Paragraph-aware chunking (BGE) | Week 5 | ✅ Done | Tied E8 — reject paragraph |
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

### Week 5 — E3, E6, E7, E8 (insights)

| ID | Test ΔNDCG@10 vs prior frozen step | Insight |
|----|-----------------------------------:|---------|
| E3 | ~0 | Duplicate-slot crowding is measurable; cap/overfetch does not move test NDCG with CE. |
| E6 | +0.006 vs E1 | 200w shallow optimum; fewer chunks, faster search. |
| E7 | ~0 vs E6 | Overlap fine-tunes; 20% confirmed at 200w. |
| E8 | +0.018 vs E6 MiniLM | Encoder dominates chunk hyperparams; ~2.4× retrieval p50. |

---

## 4. Week 5 experiment sprint — complete

1. **E3** — control pool frozen (test tied vs cap2).  
2. **E4 / E4b** — removed (cost).  
3. **E5** — skipped.  
4. **E6 → E7** — **200w / 20%** chunking.  
5. **E8** — **BGE-base** adopted on test.  
6. **E9** — paragraph-aware chunking vs E8 fixed 200w (BGE).

**Incremental insights (test NDCG@10 path, dense, no rerank):** E1 **0.370** → E6 **0.376** (+0.006, segmentation) → E8 **0.394** (+0.018 vs 200w MiniLM, encoder). Chunk overlap (E7) confirms defaults without a separate test lift. Full ladder: [`eval_protocol_reference.md`](eval_protocol_reference.md).

**Exit criteria:** met — see [`rag_experiment_results.md`](rag_experiment_results.md) §5–7 and per-sheet **Insights** in [`experiments/`](experiments/README.md).

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
