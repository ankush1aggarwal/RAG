# E3 — Candidate diversity / duplicate-chunk crowding

| Field | Value |
|-------|--------|
| **ID** | E3 |
| **Status** | **Complete** |
| **Primary variable** | Per-`doc_id` chunk cap before rerank |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

## 1. Question and hypothesis

- **Question:** How should the fixed rerank budget (50 slots) be allocated across documents when multiple chunks from one `doc_id` appear in the dense pool?
- **Hypothesis:** Capping chunks per document will reduce redundant slots and can improve document-level NDCG@10 / MRR@10 without hurting candidate Recall@50 for relevant documents.
- **Success criteria:** Select policy on **validation** using quality + latency; confirm on test once.

## 2. Arms (from roadmap)

| Arm | Description |
|-----|-------------|
| 1 — Control | Top 50 dense chunks, no cap |
| 2 — Diagnostic | Deduplicate top 50 to one chunk per `doc_id` (no backfill) |
| 3 — Treatment | Overfetch (e.g. 100–150), cap **1** chunk per `doc_id`, fill to 50 |
| 4 — Treatment | Same overfetch, cap **2** per `doc_id` |

**Fixed:** full `fiqa` index, dense retriever, CE model, document-level qrels metrics.

## 3. Results

| Arm | Unique docs in pool | Duplicate-slot rate | Cand. Recall@50 | NDCG@10 | MRR@10 | Recall@10 | Rerank p50/p95 |
|-----|-------------------:|--------------------:|----------------:|--------:|-------:|----------:|----------------|
| Control | 43.8 | 0.123 | 0.627 | 0.378 | 0.457 | 0.442 | 865 / 1295 |
| Dedup top 50 | 43.8 | 0.000 | 0.627 | 0.375 | 0.443 | 0.454 | 738 / 1099 |
| Overfetch cap 1 | 50.0 | 0.000 | 0.646 | 0.373 | 0.441 | 0.452 | 816 / 1253 |
| Overfetch cap 2 | 45.0 | 0.099 | 0.633 | **0.382** | **0.459** | 0.449 | 801 / 1185 |

**Validation** (n=500): artifacts `data/benchmarks/week5_e3_*_validation.json`.

### Test — control vs validation winner (n=648)

| Arm | Unique docs in pool | Duplicate-slot rate | Cand. Recall@50 | NDCG@10 | MRR@10 | Recall@10 | Rerank p50/p95 |
|-----|-------------------:|--------------------:|----------------:|--------:|-------:|----------:|----------------|
| Control | 43.4 | 0.132 | 0.607 | 0.368 | 0.441 | 0.438 | 886 / 1296 |
| Overfetch cap 2 | 44.7 | 0.107 | 0.610 | 0.369 | 0.441 | 0.441 | 883 / 1318 |

Artifacts: `data/benchmarks/week5_e3_control_test.json`, `week5_e3_overfetch_cap2_test.json`; report: [`week5_e3_candidate_diversity.md`](../week5_e3_candidate_diversity.md).

## 4. Insights (interpretation)

- **Effect size:** Test NDCG@10 control **0.368** vs cap2 **0.369** (~**+0.001**, not material); validation cap2 +0.004 did **not** hold on test.
- **Mechanism:** Duplicate-slot rate ~**12–13%** in control — real crowding, but **replacing** slots with deeper overfetch did not improve document-level ranked metrics; dedup-without-backfill shrinks the pool and hurts val NDCG.
- **Interaction:** Explains E2a hurt mode; **no production impact** while rerank is off; pool policy frozen to **control** for any future CE eval.
- **Caveats:** Experiment assumes rerank on; metrics are post-rerank in arms that use CE.

## 5. Decision

**Adopt: control (top 50 dense chunks, no per-doc cap)** for any future **local** rerank eval (E5 skipped).

**Rationale:** Validation favored `overfetch_cap2` (+0.004 NDCG@10, +0.002 MRR@10), but **test is effectively tied** (NDCG@10 0.3686 vs 0.3684; MRR@10 0.4405 vs 0.4407). Duplicate-slot rate drops with cap2 (~11% vs 13%) without material ranked-metric gain. Dedup/cap1 arms failed validation quality checks — not carried to test.

**Production profile:** unchanged — **dense, rerank off, top_k=10** (diversification applies only when rerank is enabled in eval/CLI).

## 6. Motivation

See **E2a** (48 hurt queries with duplicate slots). Full protocol: [`RAG_PROJECT_ROADMAP.md`](../../RAG_PROJECT_ROADMAP.md).
