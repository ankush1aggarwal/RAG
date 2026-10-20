# Week 4 — Retrieval program report

**Status:** Complete (2026-10-19)  
**Collection:** `fiqa` (full corpus)  
**Encoder:** `sentence_transformer` / `all-MiniLM-L6-v2`  
**Corpus:** `BeIR/fiqa` — **57,638** documents, **111,170** chunks indexed  

## Experiment flow

| Step | ID | Split | Methods | Rerank | Purpose |
|------|-----|-------|---------|--------|---------|
| 1 | Exp 1 | validation (1k slice) | dense, BM25, hybrid | no | Week 3 narrative / sanity |
| 2 | Exp 2 | validation | dense, BM25, hybrid | no | Full-corpus labeled baseline |
| 3 | Exp 3 | validation | hybrid, weighted hybrid | no | Tune RRF weights |
| 4 | Exp 4 | test | dense, BM25, weighted hybrid | no | Frozen-weight test baseline |
| 5 | Exp 5 | test | dense, BM25, weighted hybrid | **local CE** | Rerank vs Exp 4 |

Plan and commands: [`week4_experiment_plan.md`](week4_experiment_plan.md).

## Results summary (mean NDCG@10)

| Context | Best method | NDCG@10 (approx.) | Notes |
|---------|-------------|-------------------|--------|
| Validation (Exp 2) | **dense** | 0.38 | Hybrid below dense; BM25 ~0.21 |
| Validation (Exp 3) | **weighted hybrid** | Tuned vs equal hybrid | Frozen weights: dense **0.95**, BM25 **0.05** |
| Test (Exp 4) | **dense** ≈ weighted hybrid | **0.37** / **0.37** | BM25 **0.19** |
| Test (Exp 5, dense only) | dense + CE | **0.37** vs 0.37 (Exp 4) | ~5–6× mean latency; BM25 +CE **0.29** vs 0.19 |

Per-query examples, latency tables, and protocol JSON are in each experiment report below.

## Learnings and trade-offs

- **Dense vs lexical on FiQA:** MiniLM dense retrieval dominates BM25 on qrels; weighted hybrid matches dense when fusion weights favor dense (Exp 3 grid).
- **Equal-weight hybrid (Exp 2 validation):** Underperformed dense on NDCG@10 — fusion cost without validation-tuned weights.
- **Weight tuning (Exp 3):** Validation `ndcg@10` ≈ **0.385** at selected weights; test uses frozen weights only (Exp 4/5).
- **Rerank (Exp 5):** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescoring from top-50 candidates; strong BM25 lift, **flat dense NDCG@10**, higher p95 latency.
- **Error analysis (dense):** 181 queries helped vs 178 hurt on ΔNDCG@10; wins mostly **ordering** within the candidate pool; losses often **ordering** or **duplicate chunk slots** — see [`week4_exp5_test_rerank.md`](week4_exp5_test_rerank.md#dense-rerank-error-analysis-exp-4-vs-exp-5).
- **Reproducibility:** `CitationPreservingEvalLLM` + `week4_eval_protocol.json`; citations in eval responses from retrieved chunks.

## Artifacts

- Protocol: `data/benchmarks/week4_eval_protocol.json`
- Weights: `data/benchmarks/week4_fusion_weights.json`
- Per-experiment JSON: `data/benchmarks/week4_exp*.json`
- Rerank error analysis: `data/benchmarks/week4_exp5_dense_rerank_error_analysis.json`
- Reports: `docs/reports/week4_exp*.md`

## Linked reports

- [week4_exp2_full_corpus_validation.md](week4_exp2_full_corpus_validation.md)
- [week4_exp3_weighted_hybrid_validation.md](week4_exp3_weighted_hybrid_validation.md)
- [week4_exp4_test_baseline.md](week4_exp4_test_baseline.md)
- [week4_exp5_test_rerank.md](week4_exp5_test_rerank.md) (includes dense rerank error analysis)
- [week4_retrieval_eval.md](week4_retrieval_eval.md) (Experiment 1)
- [week4_experiment_plan.md](week4_experiment_plan.md)
