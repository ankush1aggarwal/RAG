# E1b — Weighted hybrid (RRF fusion tuning)

| Field | Value |
|-------|--------|
| **ID** | E1b |
| **Status** | Complete |
| **Week** | 4 (Exp 3–4) |
| **Primary variable** | Dense vs BM25 fusion weights in weighted RRF |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §4 |

---

## 1. Question and hypothesis

- **Question:** Can validation-tuned fusion weights beat equal-weight hybrid and match or beat dense alone?
- **Hypothesis:** Up-weighting dense will recover most of dense quality while retaining some lexical signal.
- **Success criteria:** Maximize validation NDCG@10; confirm on **test** without re-tuning weights.

## 2. Control and treatments

| Arm | Description |
|-----|-------------|
| Control | Equal-weight hybrid (Exp 3 baseline row) |
| Treatment | Grid-searched weights on validation |

**Selected weights (frozen for test):** dense **0.950**, BM25 **0.050** → `data/benchmarks/week4_fusion_weights.json`.

## 3. Evaluation protocol

Same index and splits as E1; `top_k=10`; no rerank.

## 4. Results

### Validation (Exp 3)

| Method | NDCG@10 | MRR@10 | Recall@10 | p50 ms | p95 ms |
|--------|--------:|-------:|----------:|-------:|-------:|
| hybrid (equal) | 0.3358 | 0.3833 | 0.4163 | 771.9 | 1184.2 |
| weighted hybrid | **0.3849** | 0.4494 | 0.4482 | 874.1 | 1373.0 |
| dense (E1 reference) | 0.3792 | 0.4446 | 0.4449 | 194.8 | 226.0 |

### Test (Exp 4)

| Method | NDCG@10 | MRR@10 | Recall@10 | p50 ms | p95 ms |
|--------|--------:|-------:|----------:|-------:|-------:|
| dense | **0.3698** | 0.4496 | 0.4304 | 197.9 | 204.5 |
| weighted hybrid | 0.3691 | 0.4474 | 0.4308 | 809.2 | 1172.0 |

## 5. Insights (interpretation)

- **Effect size:** Validation NDCG@10 **0.385** weighted vs **0.379** dense (+0.006); test **0.3691** vs **0.3698** dense (~**−0.001**, noise).
- **Mechanism:** Tuning dense-heavy weights fixes most of the equal-hybrid validation gap by trusting vectors; on test, lexical tail adds **no durable NDCG** but still pays **~4×** dense p50 for second-stage BM25 + fusion.
- **Caveats:** Single validation grid; weights frozen for test — correct discipline, but marginal test tie was predictable from E1.

## 6. Decision

| Outcome | **Reject** weighted hybrid for production default |
| **Rationale** | Test NDCG@10 within ~0.0007 of dense but ~4× p50 latency and extra fusion complexity. |
| **Production impact** | Keep `weighted_hybrid` as **eval/CLI-only**; weights archived for experiments. |
| **Limitations** | Single grid on validation; no test-time weight search. |

## 7. Artifacts

[`week4_exp3_weighted_hybrid_validation.md`](../week4_exp3_weighted_hybrid_validation.md), [`week4_exp4_test_baseline.md`](../week4_exp4_test_baseline.md), `data/benchmarks/week4_fusion_weights.json`.
