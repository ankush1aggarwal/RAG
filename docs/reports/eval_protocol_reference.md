# Evaluation protocol reference

Shared rules for Week 4+ labeled retrieval experiments and Week 5+ reranker trials.

## Qrels splits

| Split | Use |
|-------|-----|
| **Validation** | Hyperparameter selection, fusion weights, E3–E8 arm choice |
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
- Hosted API **cost** (if any paid eval API is used)  
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

## Incremental experiment insights (how to read results)

Each controlled experiment changes **one main variable** while holding corpus, splits, and metric code fixed. After every run, record **effect size** (ΔNDCG@10 on validation vs the step’s baseline), **what mechanism moved** (coverage vs ordering vs latency), and **caveats** — not only adopt/reject. Detail lives on per-experiment sheets ([`experiments/README.md`](experiments/README.md)); this section is the cumulative ladder through Week 5.

| Step | Δ test NDCG@10 (vs prior frozen step) | What moved | Interpretation |
|------|--------------------------------------:|------------|----------------|
| **E0** | — (unlabeled) | Latency, Jaccard overlap | Hybrid ~2× dense p50 on 1k slice; rankings differ — **cannot** justify NDCG choices. |
| **E1** | Baseline **0.370** (dense @100w MiniLM) | Method | Dense beats BM25/hybrid on quality **and** latency at full corpus; semantic retrieval fits FiQA Q&A. |
| **E1b** | ~0 vs dense | Fusion weights | Validation gain from 0.95/0.05 dense/BM25 **does not** survive as a win on test once latency (~4× p50) is counted. |
| **E2** | **−0.001** (dense+CE) | Rerank @50 | CE reorders within a strong pool but **flat** aggregate NDCG; ~5.6× retrieve+rerank cost — reject for prod. |
| **E2a** | (diagnostic) | Per-query Δ | Help/hurt ~symmetric; hurts include **duplicate chunk slots** (48 queries) — motivates E3, not a prod fix while rerank is off. |
| **E3** | ~0 (control vs cap2) | Pool policy pre-rerank | Caps reduce duplicate-slot rate but **test tied**; validation-only edge — keep **control** for any future rerank eval. |
| **E5** | — (skipped) | Depth | No new reranker; depth sweep pointless while E2 rejects CE @50. |
| **E6** | **+0.006** (200w vs 100w) | Segmentation | Modest gain: fewer, longer chunks improve context per vector; 300w not better than 200w; **faster** p50 (fewer vectors). Indexing hyperparam in a reasonable band. |
| **E7** | ~0 (20% already E6 index) | Overlap | Second-order knob: 10% **hurts**, 30% flat — 20% already near optimum at 200w. |
| **E8** | **+0.018** (BGE vs MiniLM @200w/20%) | Encoder | Largest lift: better query–passage geometry; Recall@10 up (val 0.468→0.506) — not only reordering. Cost ~**2.4×** p50 vs MiniLM at same chunk count. |
| **E9** | **0** vs E8 (val/test) | Segmentation | Same 72,851 chunks — paragraph mode equivalent to fixed 200w on FiQA; **reject** paragraph index. |

**Cumulative test path (dense, no rerank):** E1 **0.370** (100w MiniLM) → E6 **0.376** (200w) → E8 **0.394** (BGE on frozen chunking). E7 does not change the test number (same `fiqa_cs200` arm).

**Layering mental model:** chunking (E6/E7) sets **where boundaries fall** and vectors-per-doc; embeddings (E8) retune **every similarity score** on fixed segments — expect larger metric moves from encoder upgrades once chunk size is in a sensible range.

**Writing discipline:** add or update an **Insights** subsection on each experiment sheet when results land; mirror one-line mechanism + ΔNDCG in [`rag_experiment_results.md`](rag_experiment_results.md) §5.
