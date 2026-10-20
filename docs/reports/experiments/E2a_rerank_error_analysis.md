# E2a — Dense rerank error analysis (Exp 4 vs Exp 5)

| Field | Value |
|-------|--------|
| **ID** | E2a |
| **Status** | Complete |
| **Week** | 4 (post Exp 5) |
| **Primary variable** | Per-query ΔNDCG@10 (dense, rerank on vs off) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §8 |

---

## 1. Question and hypothesis

- **Question:** When CE rerank helps or hurts dense retrieval, what mechanism explains the change?
- **Hypothesis:** Gains are mostly reordering within a good candidate pool; hurts include duplicate chunk slots crowding out distinct relevant documents.

## 2. Control and treatments

Compare per-query metrics: **Exp 4** (no rerank) vs **Exp 5** (dense + CE), same 648 test queries.

## 3. Results

| Category | Query count |
|----------|------------:|
| Helped (ΔNDCG@10 > 0) | 181 |
| Hurt (ΔNDCG@10 < 0) | 178 |
| Unchanged | 289 |

**Helped (typical):** relevant document already in top-50 pool; CE improves @10 ordering (`irrelevant_candidate_ordering`).

**Hurt (typical):** demotions at @10; **48 queries** with duplicate chunk slots from same `doc_id` consuming rerank budget.

**Missing candidates:** no cases classified as relevant doc absent from top-50 pool (coverage already strong).

## 4. Decision

| Outcome | **Inform E3** — test per-document candidate caps before another rerank production decision |
| **Rationale** | Aggregate flat NDCG masks symmetric help/hurt; duplicate crowding is a fixable candidate-policy issue. |
| **Production impact** | None until E3 → **E4** (Jev) / **E5** (depth) complete. |

## 5. Artifacts

`data/benchmarks/week4_exp5_dense_rerank_error_analysis.json`; narrative in [`week4_exp5_test_rerank.md`](../week4_exp5_test_rerank.md).

```powershell
python src/scripts/analyze_rerank_dense.py --collection fiqa
```
