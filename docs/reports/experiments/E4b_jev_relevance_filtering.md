# E4b — Jev relevance filtering (optional)

| Field | Value |
|-------|--------|
| **ID** | E4b |
| **Status** | **Optional** (Week 5, after **E4** if Jev rerank shows promise) |
| **Primary variable** | Score-based **filtering** of candidates (not just reordering) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

## 1. Question and hypothesis

- **Question:** Can Jev **relevance filtering** drop distracting passages while **preserving** evidence needed for document-level qrels metrics?
- **Hypothesis:** Filtering may improve precision@context for generation; it may **hurt** Recall@10 if relevant chunks are thresholded out.
- **Prerequisite:** **E4** completed with a Jev `rerank()` configuration worth extending.

**Not in scope for E4:** E4 uses `rerank()` only so ordering is isolated from pool-size changes.

## 2. Control and treatments

| Arm | Description |
|-----|-------------|
| Control | E4-winning Jev **`rerank()`** (no filter) |
| Treatment(s) | Jev filtering with documented threshold(s) / API parameters — TBD per library docs |

**Fixed:** E3 policy, same candidate fetch depth as E4, same model/version/instructions as E4.

## 3. Hosted API protocol

Same audit fields as [E4](E4_jev_rerank.md) §4, plus:

| Item | Value |
|------|--------|
| Filter threshold(s) | TBD |
| Min/max passages after filter | TBD — record actual pool size per query |
| Fallback if pool empty | TBD (must be explicit in code) |

## 4. Results

| Arm | NDCG@10 | Recall@10 | Mean pool size | Filtered-out relevant rate | p50/p95 | Cost |
|-----|--------:|----------:|---------------:|---------------------------:|---------|------|
| `rerank()` only | TBD | TBD | 50 → 10 | — | TBD | TBD |
| + filtering | TBD | TBD | TBD | TBD | TBD | TBD |

## 5. Decision

TBD — adopt only if retrieval metrics **and** downstream generation/citation goals (see **E10**) justify smaller, filtered context.
