# Evaluation protocol reference

Shared rules for Week 4+ labeled retrieval experiments and Week 5+ reranker trials.

## Qrels splits

| Split | Use |
|-------|-----|
| **Validation** | Hyperparameter selection, fusion weights, E3–E8 arm choice, Jev listwise vs pointwise on validation |
| **Test** | **Final** reporting only — one run per frozen configuration |

Do not iterate on test metrics to pick settings.

## Quality metrics

| Metric | Role |
|--------|------|
| **NDCG@10** | **Primary** — must justify adopting a treatment vs the previous baseline |
| MRR@10, Recall@10 | Supporting ranked-list metrics at reported `top_k` |
| **Candidate Recall@50** | Document-level recall on the **first-stage** pool (depth 50 or rerank candidate limit) **before** rerank reordering; measures whether relevant documents enter the candidate pool |

Ranked-list metrics use document-level qrels after chunk→doc deduplication (first-seen rank).

## Adoption rule

Adopt a change for the production profile only if **test NDCG@10** improves by a **material** margin (pre-specify on validation what “material” means) **and** the trade-off is acceptable for:

- **p95** first-stage and/or rerank latency  
- Hosted API **cost** (e.g. Jev E4)  
- **Operational complexity** (extra provider, data handling, concurrency limits)

Otherwise keep the prior baseline.

## Latency labels (eval scripts)

| Label | Includes | Excludes |
|-------|----------|----------|
| **First-stage retrieval** | Dense/BM25/hybrid search for the candidate pool | Rerank; embedding **model load** |
| **Rerank** | Cross-encoder or hosted reranker scoring | First-stage retrieval; generation |
| **Retrieve + rerank** | Sum of the two | Stub/real LLM generation |
| **Generation** | LLM call for eval stub or production model | Retrieval |

Week 3 microbenchmarks document their own warm-up/repeat protocol in [`week3_retrieval_comparison.md`](week3_retrieval_comparison.md).

## Reproducibility fields

Each run should persist:

- `eval_dataset_path` — **repository-relative** (e.g. `data/benchmarks/fiqa_qrels_validation.json`)
- `protocol` JSON — `top_k`, `k_values`, rerank flags, `candidate_recall_k`, fusion weights
- `git_commit` — short SHA when available
- `evaluated_at` — UTC timestamp

Regenerate markdown reports via `src/scripts/run_week4_experiments.py` (or Week 5 runners) after protocol changes.

## Production profile vs API defaults

The **documented production profile** (`dense`, no rerank, `top_k=10`) reflects the Week 4 eval protocol. The HTTP API may still default `top_k=3` until **Week 6** enforces the frozen profile in code.
