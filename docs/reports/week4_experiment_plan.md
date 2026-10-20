# Week 4 — Experiment plan (retrieval + optional rerank)

| # | Goal | Index | Qrels split | Methods | Rerank | Output |
|---|------|-------|-------------|---------|--------|--------|
| **1** | Sanity vs Week 3 narrative | 1k `fiqa_sample` | validation (filtered) | dense, BM25, hybrid | no | `week4_retrieval_eval.md` |
| **2** | Full-corpus baseline | Full `fiqa` | **validation** | dense, BM25, hybrid (equal RRF) | no | `week4_exp2_full_corpus_validation.md` |
| **3** | Learn fusion weights | Full `fiqa` | **validation** | hybrid vs weighted hybrid | no | `week4_exp3_weighted_hybrid_validation.md`, `week4_fusion_weights.json` |
| **4** | Held-out baseline | Full `fiqa` | **test** | dense, BM25, weighted hybrid (frozen exp 3) | no | `week4_exp4_test_baseline.md` |
| **5** | Rerank ablation | Full `fiqa` | **test** | same as exp 4 | **local cross-encoder** | `week4_exp5_test_rerank.md` |

**Master report:** `week4_retrieval_program_report.md` (flow, learnings, trade-offs; links all artifacts).

## Reproducibility

- **Quality:** BEIR qrels + frozen `EvalProtocol` JSON (`data/benchmarks/week4_eval_protocol.json`) saved with each run.
- **Latency:** per-query retrieval time (ms) aggregated as mean / p50 / p95 in each report.
- **RAG path:** `CitationPreservingEvalLLM` stub — deterministic answers; **citations** always built from retrieved chunks via `citations_from_hits`.

## Commands

```powershell
# One-time full index
$env:FIQA_CORPUS_LIMIT="0"
$env:COLLECTION_NAME="fiqa"
python src/scripts/ingest.py --full-corpus --collection fiqa --recreate

# Experiments 2–5 (add --prepare to refresh qrels JSON)
python src/scripts/run_week4_experiments.py exp2 --collection fiqa --prepare
python src/scripts/run_week4_experiments.py exp3 --collection fiqa
python src/scripts/run_week4_experiments.py exp4 --collection fiqa --prepare
python src/scripts/run_week4_experiments.py exp5 --collection fiqa

# Or sequential + master report
python src/scripts/run_week4_experiments.py all --collection fiqa --prepare
```

**Discipline:** Tune only on validation (exp 3). Report test (exp 4/5) with frozen weights.

**Tuning objective:** maximize mean `ndcg@10` on validation (override with `--tune-metric mrr@10`, etc.).
