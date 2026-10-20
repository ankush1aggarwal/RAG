# Retrieval Experiment Results — Production-Grade RAG

**Repository:** https://github.com/ankush1aggarwal/RAG  
**Status:** Living report — completed experiments filled; Week 5 arms remain `TBD` until run  
**Implementation decision:** **Provisional** — dense retrieval, no rerank, `top_k=10` (Week 6 API lock)  
**Selected production configuration commit:** TBD after Week 5 freeze (baseline measured at `6210bdd`)

**Per-experiment detail:** [`experiments/README.md`](experiments/README.md)  
**Template (do not edit for results):** [`RAG_EXPERIMENT_REPORT_TEMPLATE.md`](../RAG_EXPERIMENT_REPORT_TEMPLATE.md)

> Measured values only. Per-query JSON under `data/benchmarks/`. Do not use Week 3 unlabeled latency as NDCG evidence.

---

## 1. Executive summary

- **Question investigated:** Which retrieval (+ optional rerank) configuration gives the best quality/latency trade-off on BEIR FiQA with document-level qrels?
- **Selected production pipeline (provisional):** **Dense** vector retrieval (`all-MiniLM-L6-v2`), **no** cross-encoder rerank, **`top_k=10`**, full **`fiqa`** index (100-word chunks, 20% overlap).
- **Why selected:** Test **NDCG@10 ≈ 0.370** with **p50 ≈ 198 ms** retrieval — best among dense, BM25, weighted hybrid, and dense+CE (E1, E1b, E2).
- **Important rejected alternatives:** BM25 and equal-weight hybrid (lower NDCG); weighted hybrid (matches dense NDCG but ~4× slower); dense+CE rerank (~5.6× latency, flat NDCG@10).
- **Main limitations:** Candidate Recall@50 not logged in E1/E2; rerank revisit via **E3** → **E4** (Jev ranking) → **E5** (depth); chunking/embedding sweeps (E6–E8) not yet run; generation rubric (**E10**) pending.

---

## 2. Evaluation protocol

| Item | Value |
|------|--------|
| Corpus/dataset ID and version | `BeIR/fiqa` (HuggingFace), full ingest |
| Indexed corpus size / number of chunks | 57,638 documents → **111,170** chunks |
| Query split and query count | **Validation 500**; **test 648** (`data/benchmarks/fiqa_qrels_{validation,test}.json`) |
| Qrels count and indexed relevant-document coverage | Full-corpus index; partial 1k sample used only for smoke eval (15 queries) |
| Chunk size / overlap | **100** words / **20%** overlap (`settings.chunk_size`, `chunk_overlap_fraction`) |
| Embedding model and vector dimension | `sentence_transformer` / **`all-MiniLM-L6-v2`**, **384**-d |
| Dense/BM25/hybrid candidate policy | Per-method top-k retrieval; hybrid RRF with equal or tuned weights (E1b) |
| Reranker model and candidate budget | `cross-encoder/ms-marco-MiniLM-L-6-v2`; **50** chunks when enabled (E2) |
| Final output cutoff | **`top_k=10`** document-level list after chunk→doc dedup (first-seen rank) |
| Hardware/runtime/dependency versions | Local Windows dev run (Oct 2025); see `pyproject.toml` / lockfile at commit |
| Timing protocol (warm-up, repeats) | Week 4 eval: per-query retrieval ms in experiment reports; E0: 3 warm-up + 7 repeats |

Chunk hits are ranked at chunk level; metrics use **document-level** qrels with graded NDCG. Queries without in-index relevant docs are handled by the eval pipeline (see `retrieval_eval.py`).

---

## 3. Metrics and selection rule

**Retrieval quality:** NDCG@10, MRR@10, Recall@5, Recall@10; **Candidate Recall@50** starting in E3/E5.

**Latency:** p50/p95 retrieval (and rerank when on); Jev **E4** also records API concurrency and hosted round-trip; end-to-end API latency in **E11**.

**Generation/citations:** Week 4 used **`CitationPreservingEvalLLM`** (deterministic stub) for citation structure only — not human rubric scores (**E10**).

**Hosted rerank (E4):** document model/version, instructions, concurrency, cost, and data-handling before running validation.

**Selection rule (applied):** Prefer higher **test** NDCG@10 among first-stage configs; break ties with lower p95 latency and lower ops complexity. Rerank requires **material** NDCG gain, not Recall-only gains at much higher latency. Tuning on **validation** only (E1b weights); test touched once per experiment stage.

---

## 4. Baseline comparison (E0–E2)

**Primary labeled comparison — FiQA test, full index, no rerank (E1 / E1b):**

| Configuration | NDCG@10 | MRR@10 | Recall@10 | Candidate Recall@50 | p50 latency | p95 latency | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense | **0.3698** | 0.4496 | 0.4304 | not logged | 197.9 ms | 204.5 ms | [E1](experiments/E1_labeled_retrieval_baseline.md) Exp 4 |
| BM25 | 0.1913 | 0.2386 | 0.2434 | not logged | 607.7 ms | 936.0 ms | Exp 4 |
| Hybrid RRF (equal) | 0.3358† | 0.3833† | 0.4163† | not logged | 796.3 ms† | 1200.0 ms† | †**validation** only, Exp 2 |
| Weighted hybrid | 0.3691 | 0.4474 | 0.4308 | not logged | 809.2 ms | 1172.0 ms | [E1b](experiments/E1b_weighted_hybrid_fusion.md) Exp 4 |
| Dense + CE rerank | 0.3684 | 0.4407 | 0.4382 | not logged | 1083.0 ms | 1495.5 ms | [E2](experiments/E2_cross_encoder_rerank.md) Exp 5 |

**E0 (unlabeled):** latency and Jaccard overlap on 1k-doc sample — [`week3_retrieval_comparison.md`](week3_retrieval_comparison.md), [E0 sheet](experiments/E0_unlabeled_latency_overlap.md). **Not** mixed into the table above.

---

## 5. Controlled experiment results

Detail sheets: [`experiments/README.md`](experiments/README.md).

### E3 — Candidate diversity / duplicate-chunk crowding

**Status:** Planned (Week 5). See [E3_candidate_diversity.md](experiments/E3_candidate_diversity.md).

**Hypothesis:** Per-document chunk caps before rerank can improve document-level NDCG@10 when duplicate slots crowd the pool (motivated by [E2a](experiments/E2a_rerank_error_analysis.md)).

| Arm | Unique docs in input | Duplicate-slot rate | Candidate Recall@50 | NDCG@10 | MRR@10 | Recall@10 | Rerank p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---|
| Current top 50, no cap | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Deduplicate top 50, no backfill | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Overfetch + cap 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Overfetch + cap 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Per-query findings:** TBD  
**Decision:** TBD

### E4 — Jev reranker (ranking only)

**Status:** Planned (Week 5, after E3). Sheet: [E4_jev_rerank.md](experiments/E4_jev_rerank.md).

**Scope:** Jev **`rerank()`** via `jev-reranker` — **no score-based filtering** (isolates ordering). Compare **no rerank**, **local CE** (E2 control under E3 policy), and **Jev** @50 candidates. Hold **listwise vs pointwise** to one mode per decision pass (document in sheet).

| Arm | NDCG@10 | MRR@10 | Recall@10 | Cand. Recall@50 | p50/p95 | API cost | Notes |
|-----|--------:|-------:|----------:|----------------:|---------|----------|-------|
| No rerank | TBD | TBD | TBD | TBD | TBD | $0 | E3 policy |
| CE @50 | TBD | TBD | TBD | TBD | TBD | $0 | E2 under E3 |
| Jev `rerank()` @50 | TBD | TBD | TBD | TBD | TBD | TBD | model/version TBD |

**API audit:** model/version, instructions, concurrency, data-handling — see sheet §4.

**Decision:** TBD

### E4b — Jev relevance filtering (optional)

See [E4b_jev_relevance_filtering.md](experiments/E4b_jev_relevance_filtering.md). Only if E4 is promising; tests filtering vs `rerank()`-only.

### E5 — Reranker candidate depth

See [E5_rerank_candidate_depth.md](experiments/E5_rerank_candidate_depth.md). E3 policy + **E4-winning** reranker; skip if dense still wins.

| Candidate budget | NDCG@10 | MRR@10 | Recall@10 | p50 rerank ms | p95 rerank ms | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 10 | TBD | TBD | TBD | TBD | TBD | TBD |
| 20 | TBD | TBD | TBD | TBD | TBD | TBD |
| 30 | TBD | TBD | TBD | TBD | TBD | TBD |
| 50 | 0.3684‡ | 0.4407‡ | 0.4382‡ | TBD | TBD | CE @50 (E2); re-run under E3 |

‡Historical E2 dense+CE; E5 re-runs under E3 (+ Jev if E4 wins).

**Decision and rationale:** TBD

### E6 — Chunk size sensitivity

See [E6_chunk_size.md](experiments/E6_chunk_size.md).

| Chunk size | Overlap held fixed | Documents/chunks | NDCG@10 | Recall@10 | Index/build time | p50/p95 query latency |
|---:|---|---|---:|---:|---|---|
| 100 words | 20% | 111,170 chunks | 0.3698 (baseline) | 0.4304 | Week 4 ingest | 197.9 / 204.5 ms |
| 200 words | 20% | TBD | TBD | TBD | TBD | TBD |
| 300 words | 20% | TBD | TBD | TBD | TBD | TBD |

**Decision and rationale:** TBD

### E7 — Chunk overlap sensitivity

See [E7_chunk_overlap.md](experiments/E7_chunk_overlap.md). TBD after E6.

### E8 — Embedding model comparison

See [E8_embedding_model.md](experiments/E8_embedding_model.md). TBD

### E9 — Hybrid/RRF sensitivity (optional/stretch)

See [E9_hybrid_rrf_sensitivity.md](experiments/E9_hybrid_rrf_sensitivity.md). **Not started** — may skip if E1/E1b conclusions stand.

### E10 — Generation quality and citation grounding

See [E10_generation_citations.md](experiments/E10_generation_citations.md). **Planned** Week 6–7 after retrieval freeze.

| Configuration | Answer relevance | Evidence support | Citation correctness | Unsupported claims | Latency | Cost/usage |
|---|---:|---:|---:|---:|---|---|
| Eval stub (Week 4) | N/A | N/A | structural | N/A | in retrieval eval | $0 |
| Production LLM + frozen retrieval | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6. Final design selection (provisional)

| Component | Selected configuration | Supporting experiment(s) | Rationale |
|---|---|---|---|
| Corpus/split for evaluation | Full `fiqa`; test for final reporting | E1 | Complete coverage vs 1k sample |
| Chunk size / overlap | 100 w / 20% | E6 / E7 pending | Week 4 baseline until sweeps run |
| Embedding model | `all-MiniLM-L6-v2` | E8 pending | Best measured test NDCG so far |
| Candidate retrieval / RRF | **Dense only** (no fusion in prod) | E1, E1b | Beats hybrid; weighted hybrid not worth latency |
| Candidate diversification policy | None (rerank off) | E3 pending | Revisit if rerank re-enabled |
| Reranker and candidate budget | **Off** | E2, **E4**, E5 pending | CE flat @50; Jev TBD |
| Final context/result count | 10 | E1 | Week 4 protocol |
| LLM / prompt configuration | Ollama/Cohere per settings | E10 pending | Stub used for retrieval eval only |

---

## 7. Final held-out comparison

**Current baseline vs provisional selected design** (both dense, no rerank — no change yet from E1):

| Metric | Baseline (dense test) | Selected design | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| NDCG@10 | 0.3698 | 0.3698 | 0 | — |
| MRR@10 | 0.4496 | 0.4496 | 0 | — |
| Recall@10 | 0.4304 | 0.4304 | 0 | — |
| Retrieval p50 | 197.9 ms | 197.9 ms | 0 | — |
| Retrieval p95 | 204.5 ms | 204.5 ms | 0 | — |
| End-to-end p50/p95 | TBD | TBD | TBD | TBD |
| Citation correctness | TBD (E10) | TBD | TBD | TBD |
| Estimated cost per query | TBD | TBD | TBD | TBD |

Update this section after Week 5 freeze when a treatment beats the dense baseline on test.

---

## 8. Error analysis

**E2a — dense rerank (Exp 4 vs 5):** 648 test queries — **181** helped, **178** hurt, **289** unchanged (ΔNDCG@10). Helps: reordering within a strong top-50 pool. Hurts: demotions + **duplicate chunk slots** (48 queries). No `missing_candidates` class in top-50.

Full JSON: `data/benchmarks/week4_exp5_dense_rerank_error_analysis.json`. Sheet: [E2a](experiments/E2a_rerank_error_analysis.md).

---

## 9. Decision log and limitations

| Decision | Evidence | Chosen option | Alternatives not chosen | Remaining limitation |
|---|---|---|---|---|
| First-stage retriever | E1 test NDCG@10 | **Dense** | BM25, equal hybrid, weighted hybrid | Hybrid not on full test for equal weights |
| Default rerank | E2 latency + NDCG | **Off** | dense+CE @50; **E4 Jev** TBD | E3 → E4 → E5 |
| Fusion in production | E1b test | **None** | weighted 0.95/0.05 | Small NDCG tie on test |
| Chunking / embeddings | E1 only | **100w / MiniLM** | larger chunks, other models | E6–E8 not run |

---

## 10. Reproducibility

- **Commit SHA:** `6210bdd` (update on re-run)
- **Commands:**

```powershell
$env:FIQA_CORPUS_LIMIT="0"
python src/scripts/run_week4_experiments.py all --collection fiqa --prepare
python src/scripts/analyze_rerank_dense.py --collection fiqa
```

- **Settings:** `src/app/config/settings.py`, `data/benchmarks/week4_eval_protocol.json`
- **Dataset/split:** `data/benchmarks/fiqa_qrels_validation.json`, `fiqa_qrels_test.json`
- **Output paths:** `data/benchmarks/week4_exp*.json`, `docs/reports/week4_exp*.md`
- **Tests:** `pytest src/tests/unit` (metrics and eval helpers)

---

## 11. Public-facing summary for README and PROJECT_SPEC

**Architecture (one line):** FiQA RAG with **dense** chunk retrieval over Qdrant + BM25 index for experiments, optional CE rerank in eval only; production API will expose **one** frozen profile after Week 6.

**Baseline vs selected (test, NDCG@10):**

| Method | NDCG@10 |
|--------|--------:|
| Dense (selected) | **0.370** |
| BM25 | 0.19 |
| Dense + CE | 0.37 |

**Most important experiments:** E1 → dense wins; E2 CE rerank rejected; E2a → **E3** pool policy → **E4** Jev ranking (hosted API, no filtering) → **E5** depth if rerank wins.

**Full report:** this file. **Sheets:** [`experiments/README.md`](experiments/README.md). **Program catalog:** [`rag_experiment_program_report.md`](rag_experiment_program_report.md).
