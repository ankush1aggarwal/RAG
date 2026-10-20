# E2 — Local cross-encoder rerank ablation

| Field | Value |
|-------|--------|
| **ID** | E2 |
| **Status** | Complete |
| **Week** | 4 (Exp 5) |
| **Primary variable** | `use_rerank` with fixed candidate pool |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §4–5 |

---

## 1. Question and hypothesis

- **Question:** Does `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking on the top-50 chunk pool improve document-level NDCG@10 enough to justify latency?
- **Hypothesis:** CE may reorder candidates when relevant docs are in the pool but poorly ranked at @10.
- **Success criteria:** Material NDCG@10 gain on **test** with acceptable p95 latency increase.

## 2. Control and treatments

| Arm | Description |
|-----|-------------|
| Control | E1 test baselines (no rerank) |
| Treatment | Rerank top **50** chunks per method before truncating to `top_k=10` |

**Fixed:** Same test queries (648); reranker `cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2`; stub LLM with citations for end-to-end timing.

## 3. Evaluation protocol

| Item | Value |
|------|--------|
| `rerank_candidate_limit` | 50 |
| `top_k` | 10 |
| Split | test |

## 4. Results

### Test (Exp 5 vs Exp 4 for dense)

| Method | NDCG@10 | MRR@10 | Recall@10 | mean ms | p50 ms | p95 ms |
|--------|--------:|-------:|----------:|--------:|-------:|-------:|
| dense (no rerank) | **0.3698** | 0.4496 | 0.4304 | 198.7 | 197.9 | 204.5 |
| dense + CE | 0.3684 | 0.4407 | 0.4382 | 1116.2 | 1083.0 | 1495.5 |
| BM25 + CE | 0.2886 | 0.3679 | 0.3247 | 1590.3 | 1592.5 | 2087.4 |
| weighted hybrid + CE | 0.3674 | 0.4379 | 0.4376 | 1702.0 | 1668.8 | 2192.1 |

**Relative cost (dense):** ~5.6× mean retrieval+rerank latency; flat NDCG@10.

## 5. Decision

| Outcome | **Reject** default production rerank |
| **Rationale** | No meaningful NDCG@10 gain on dense; large latency increase. BM25+CE improves BM25 but remains below dense. |
| **Production impact** | `use_rerank=false` in frozen profile; CE remains optional in scripts/API during dev. |
| **Limitations** | Single CE model and 50-candidate budget; E3 pool policy and **E4** (Jev) may change conclusion. |

## 6. Artifacts

[`week4_exp5_test_rerank.md`](../week4_exp5_test_rerank.md), `data/benchmarks/week4_exp5*.json`.

```powershell
python src/scripts/run_week4_experiments.py exp5 --collection fiqa
```
