# E5 — Reranker candidate depth

| Field | Value |
|-------|--------|
| **ID** | E5 |
| **Status** | **Planned** (Week 5, after **E4**; skip if no rerank beats dense) |
| **Primary variable** | `rerank_candidate_limit` |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

## 1. Question and hypothesis

- **Question:** What candidate budget (10 / 20 / 30 / 50) balances NDCG@10 and rerank latency for the **E4-winning** reranker (Jev or CE)?
- **Hypothesis:** Smaller pools reduce latency and API cost; depth >30 may not help if Recall@50 already saturates under the E3 policy.

## 2. Results

| Budget | NDCG@10 | MRR@10 | Recall@10 | p50 rerank ms | p95 rerank ms | API cost (if Jev) | Decision |
|-------:|--------:|-------:|----------:|--------------:|--------------:|------------------:|----------|
| 10 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD | TBD | TBD | TBD | E2/E4 reference @50 |

**Fixed:** E3 candidate policy; reranker = outcome of **E4** (not both CE and Jev in one table).

## 3. Decision

TBD
