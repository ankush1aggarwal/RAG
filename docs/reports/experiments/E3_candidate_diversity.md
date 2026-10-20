# E3 — Candidate diversity / duplicate-chunk crowding

| Field | Value |
|-------|--------|
| **ID** | E3 |
| **Status** | **Planned** (Week 5) |
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
| Control | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Dedup top 50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Overfetch cap 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Overfetch cap 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 4. Decision

TBD — update after run; freeze chosen policy before **E4** (Jev / CE rerank comparisons).

## 5. Motivation

See **E2a** (48 hurt queries with duplicate slots). Full protocol: [`RAG_PROJECT_ROADMAP.md`](../../RAG_PROJECT_ROADMAP.md).
