# E5 — Reranker candidate depth

| Field | Value |
|-------|--------|
| **ID** | E5 |
| **Status** | **Skipped** (Week 5) — E2 rejected CE @50; E3 pool policy neutral on test |
| **Primary variable** | `rerank_candidate_limit` |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

## 1. Question and hypothesis

- **Question:** What candidate budget (10 / 20 / 30 / 50) balances NDCG@10 and rerank latency for the winning **local** reranker?
- **Hypothesis:** Smaller pools reduce latency and API cost; depth >30 may not help if Recall@50 already saturates under the E3 policy.

## 2. Results

| Budget | NDCG@10 | MRR@10 | Recall@10 | p50 rerank ms | p95 rerank ms | Decision |
|-------:|--------:|-------:|----------:|--------------:|--------------:|----------|
| 10 | — | — | — | — | — | Not run |
| 20 | — | — | — | — | — | Not run |
| 30 | — | — | — | — | — | Not run |
| 50 | 0.368‡ | 0.441‡ | 0.438‡ | — | — | E2 reference @50 |

‡Week 4 E2 dense+CE under pre-E3 pool; E3 test tied control vs cap2 — no rerank adoption.

**Fixed:** E3 control pool if re-opened; local CE only.

## 3. Insights (interpretation)

- **Effect size:** Not measured — only E2 reference @50 (**0.368** dense+CE).
- **Mechanism:** Depth sweeps matter when rerank **adds** NDCG; with E2 flat @50 and E3 neutral, varying 10–30 cannot change the production decision.
- **Caveats:** Re-open if a new reranker beats dense on test under E3 control pool.

## 4. Decision

**Skip** — production stays dense-only; depth sweep only relevant if a future reranker beats dense on test.
