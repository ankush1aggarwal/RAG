# Retrieval Experiment Results — Production-Grade RAG

**Repository:** https://github.com/ankush1aggarwal/RAG  
**Status:** **Retrieval experiment program complete** (E0–E9; E4/E5 out of scope/skipped)  
**Teaching summary:** [`rag_retrieval_lessons.md`](rag_retrieval_lessons.md)  
**Implementation decision:** **Frozen (retrieval)** — dense, no rerank, `top_k=10`, **200w / 20% / BGE-base** (Week 6 API lock)  
**Selected production configuration commit:** TBD at Week 6 merge (eval index: `fiqa_cs200_bge_base_en_v1_5`)

**Per-experiment detail:** [`experiments/README.md`](experiments/README.md)  
**Template (do not edit for results):** [`RAG_EXPERIMENT_REPORT_TEMPLATE.md`](../RAG_EXPERIMENT_REPORT_TEMPLATE.md)

> Measured values only. Per-query JSON under `data/benchmarks/`. Do not use Week 3 unlabeled latency as NDCG evidence.

---

## 1. Executive summary

- **Question investigated:** Which retrieval (+ optional rerank) configuration gives the best quality/latency trade-off on BEIR FiQA with document-level qrels?
- **Selected production pipeline (retrieval):** **Dense** `BAAI/bge-base-en-v1.5`, **no** rerank, **`top_k=10`**, **200-word / 20% overlap**, collection **`fiqa_cs200_bge_base_en_v1_5`** (72,851 chunks).
- **Why selected:** Test **NDCG@10 0.394** (E8) vs **0.370** Week 4 dense baseline (E1 @100w MiniLM); path **0.370 → 0.376 (E6) → 0.394 (E8)**. Method choice: dense beats BM25/hybrid (E1); fusion not worth latency (E1b); CE flat @50 (E2).
- **Important rejected alternatives:** BM25, hybrid, weighted hybrid, dense+CE, MiniLM @200w, chunk overlap ≠20%, E3 cap policies (test tied).
- **Incremental insight:** **Indexing hyperparams (E6/E7)** gave small, stable gains; **encoder (E8)** gave the largest test ΔNDCG — see [`eval_protocol_reference.md`](eval_protocol_reference.md) § Incremental experiment insights and **§5.1** below.
- **Main limitations:** Served API still uses legacy **`fiqa`** (100w MiniLM) until Week 6; API defaults `top_k=3` until code lock; E9 paragraph index removed from disk after tie with E8.

---

## 2. Evaluation protocol

| Item | Value |
|------|--------|
| Corpus/dataset ID and version | `BeIR/fiqa` (HuggingFace), full ingest |
| Indexed corpus size / number of chunks | 57,638 documents → **72,851** chunks (frozen E8 index); legacy **`fiqa`**: 111,170 @ 100w |
| Query split and query count | **Validation 500**; **test 648** (`data/benchmarks/fiqa_qrels_{validation,test}.json`) |
| Qrels count and indexed relevant-document coverage | Full-corpus index; partial 1k sample used only for smoke eval (15 queries) |
| Chunk size / overlap | **200** words / **20%** overlap (E6, E7) |
| Embedding model and vector dimension | **`BAAI/bge-base-en-v1.5`**, **768**-d (E8); MiniLM baseline in experiment arms |
| Dense/BM25/hybrid candidate policy | Per-method top-k retrieval; hybrid RRF with equal or tuned weights (E1b) |
| Reranker model and candidate budget | `cross-encoder/ms-marco-MiniLM-L-6-v2`; **50** chunks when enabled (E2) |
| Final output cutoff | **`top_k=10`** document-level list after chunk→doc dedup (first-seen rank) |
| Hardware/runtime/dependency versions | Local Windows dev run (Oct 2025); see `pyproject.toml` / lockfile at commit |
| Timing protocol | See [`eval_protocol_reference.md`](eval_protocol_reference.md): first-stage vs rerank vs retrieve+rerank; **no model load** in timed eval; E0 uses warm-up + 7 repeats |
| Artifact paths | Repo-relative (`data/benchmarks/...`); `git_commit` in protocol JSON on new runs |

Chunk hits are ranked at chunk level; metrics use **document-level** qrels with graded NDCG. **Tune on validation; test for final reporting only.**

---

## 3. Metrics and selection rule

**Retrieval quality:** **NDCG@10 (primary)**, MRR@10, Recall@5, Recall@10, **candidate Recall@50** (first-stage pool, document-level).

**Latency:** Report **first-stage**, **rerank**, and **retrieve+rerank** separately (see protocol reference).

**Generation/citations:** Week 4 stub LLM — structural citations only (**E10** for real rubric).

**Selection rule:** A treatment is adopted only if **test NDCG@10** improves **materially** vs the prior baseline **and** the added **p95 latency**, **hosting cost**, and **operational complexity** are acceptable. Hyperparameters and arm choice on **validation** only; **one** test evaluation per frozen candidate config. Rerank is not adopted for recall-only gains at much higher latency.

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

**Insights (E0–E2):** E0 defers quality to labeled eval; E1 establishes dense **0.370** test baseline; E1b fusion ties on test at ~4× latency; E2 CE **flat** NDCG with ~5.6× cost — rerank off. Sheets: [E1](experiments/E1_labeled_retrieval_baseline.md), [E1b](experiments/E1b_weighted_hybrid_fusion.md), [E2](experiments/E2_cross_encoder_rerank.md).

---

## 5. Controlled experiment results

Detail sheets: [`experiments/README.md`](experiments/README.md). **Interpretation ladder:** [`eval_protocol_reference.md`](eval_protocol_reference.md) § Incremental experiment insights.

### §5.1 Cross-experiment insights (cumulative)

| ID | Δ test NDCG@10 vs prior step | Mechanism (one line) |
|----|-----------------------------:|----------------------|
| E1 | **0.370** baseline | Dense wins on semantic FiQA Q&A vs lexical/fusion. |
| E1b | ~0 vs E1 | Fusion tie on test, ~4× latency — reject. |
| E2 | −0.001 | CE reorder only; flat aggregate, high cost. |
| E2a | (diag.) | Symmetric help/hurt; duplicate slots → E3. |
| E3 | ~0 | Caps change pool stats, not test NDCG. |
| E6 | +0.006 | 200w segmentation + fewer vectors. |
| E7 | ~0 | Overlap confirms 20% at 200w. |
| E8 | +0.018 | BGE retunes all scores; Recall@10 up. |
| E9 | 0 | Same 72,851 chunks as E8 on FiQA — paragraph mode no-op. |

### E3 — Candidate diversity / duplicate-chunk crowding

**Status:** Complete (Week 5). See [E3_candidate_diversity.md](experiments/E3_candidate_diversity.md).

**Hypothesis:** Per-document chunk caps before rerank can improve document-level NDCG@10 when duplicate slots crowd the pool (motivated by [E2a](experiments/E2a_rerank_error_analysis.md)).

| Arm | Split | Unique docs in pool | Dup.-slot rate | Cand. Recall@50 | NDCG@10 | MRR@10 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Control | val | 43.8 | 0.123 | 0.627 | 0.378 | 0.457 | 0.442 |
| Dedup top 50 | val | 43.8 | 0.000 | 0.627 | 0.375 | 0.443 | 0.454 |
| Overfetch cap 1 | val | 50.0 | 0.000 | 0.646 | 0.373 | 0.441 | 0.452 |
| Overfetch cap 2 | val | 45.0 | 0.099 | 0.633 | 0.382 | 0.459 | 0.449 |
| Control | test | 43.4 | 0.132 | 0.607 | 0.368 | 0.441 | 0.438 |
| Overfetch cap 2 | test | 44.7 | 0.107 | 0.610 | 0.369 | 0.441 | 0.441 |

**Per-query findings:** ~12% mean duplicate-slot rate in control pools; cap2 lowers duplicates slightly; ranked metrics flat on test.  
**Insights:** Real crowding, weak link to document-level NDCG when rerank is the only consumer of pool policy — validation cap2 edge did not replicate on test.  
**Decision:** **Control (no cap)** if rerank is used in eval; no production change (rerank remains off).

### E4 — Jev reranker (ranking only)

**Status:** **Removed** from program (hosted API cost). Sheet archived: [E4_jev_rerank.md](experiments/E4_jev_rerank.md). Rerank evidence remains **E2** (local CE).

**Decision:** Not run.

### E4b — Jev relevance filtering (optional)

**Status:** **Removed** (depended on E4). Sheet: [E4b_jev_relevance_filtering.md](experiments/E4b_jev_relevance_filtering.md).

### E5 — Reranker candidate depth

**Status:** **Skipped** — E2 rejected CE @50; E3 did not change adoption; no new reranker. See [E5_rerank_candidate_depth.md](experiments/E5_rerank_candidate_depth.md).

| Candidate budget | NDCG@10 | MRR@10 | Recall@10 | Decision |
|---:|---:|---:|---:|---|
| 10–30 | — | — | — | Not run |
| 50 | 0.3684‡ | 0.4407‡ | 0.4382‡ | E2 reference @50 |

‡Week 4 E2 dense+CE (pre-E3 pool).

**Insights:** Skipped because E2 already shows depth @50 does not beat no-rerank; E3 shows pool policy does not unlock rerank value on test.  
**Decision:** **Skip** depth sweep while production rerank stays off.

### E6 — Chunk size sensitivity

**Status:** Complete. See [E6_chunk_size.md](experiments/E6_chunk_size.md).

| Chunk size | Collection | Chunks | NDCG@10 (val) | NDCG@10 (test) | Recall@10 (val) | p50 val (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 100 | `fiqa` | 111,170 | 0.385 | (E1 test 0.370) | 0.462 | 188.5 |
| 200 | `fiqa_cs200` | 72,851 | **0.393** | **0.376** | 0.468 | 126.8 |
| 300 | `fiqa_cs300` | 63,690 | 0.391 | — | 0.461 | 114.6 |

**Insights:** Shallow optimum near **200w**; gain is **segmentation + index size**, not a substitute for a better encoder (E8 larger Δ).  
**Decision (provisional):** **200 words** @ 20% overlap — validation winner; test NDCG@10 **0.376** vs Week 4 **0.370** @ 100w. Production index still **`fiqa`** @ 100w until re-ingest/swap; extra collections retained.

### E7 — Chunk overlap sensitivity

**Status:** Complete. Fixed **200w** (E6). See [E7_chunk_overlap.md](experiments/E7_chunk_overlap.md).

| Overlap | Collection | NDCG@10 (val) | NDCG@10 (test) | Decision |
|--------:|------------|--------------:|---------------:|----------|
| 10% | `fiqa_cs200_ov10` | 0.384 | — | Reject |
| **20%** | `fiqa_cs200` | **0.393** | **0.376** | **Keep** |
| 30% | `fiqa_cs200_ov30` | 0.390 | — | Reject |

`fiqa_cs300` removed before E7. Overlap **20%** confirmed with E6 chunk size **200**.

**Insights:** Second-order knob — confirms E6 default; test unchanged on shared `fiqa_cs200` arm.

### E8 — Embedding model comparison

**Status:** Complete. 200w / 20% overlap. See [E8_embedding_model.md](experiments/E8_embedding_model.md).

| Model | Collection | NDCG@10 (val) | NDCG@10 (test) | p50 val | Decision |
|-------|------------|--------------:|---------------:|--------:|----------|
| MiniLM | `fiqa_cs200` | 0.393 | 0.376‡ | 119 ms | Reject |
| BGE-base | `fiqa_cs200_bge_base_en_v1_5` | **0.418** | **0.394** | 286 ms | **Adopt** |

‡MiniLM test from E6/E7; E8 validation MiniLM matches `fiqa_cs200`.

**Insights:** Largest Week 5 test lift; improves **matching + recall@10**, not only order; ~2.4× retrieval p50 vs MiniLM at same chunking.

### E9 — Paragraph-aware chunking (vs E8 BGE)

**Status:** Complete. See [E9_paragraph_chunking.md](experiments/E9_paragraph_chunking.md), [`week5_e9_paragraph_chunking.md`](week5_e9_paragraph_chunking.md).

| Arm | Collection | NDCG@10 (val) | NDCG@10 (test) | Decision |
|-----|------------|--------------:|---------------:|----------|
| Fixed 200w (E8) | `fiqa_cs200_bge_base_en_v1_5` | **0.418** | **0.394** | **Keep** |
| Paragraph | `fiqa_para_bge_base_en_v1_5` | **0.418** | — | **Reject** |

**Insights:** Identical chunk count (72,851) and validation metrics — FiQA lacks usable paragraph boundaries; no gain over E8. Hybrid/RRF sweeps deferred (E1).

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
| Chunk size / overlap | **200 w / 20%** | E6, E7, E9 | E9 tied E8 — keep fixed windows |
| Index (eval / post–Week 6) | **`fiqa_cs200_bge_base_en_v1_5`** | E8 | 72,851 chunks |
| Index (served until Week 6) | **`fiqa`** (100w MiniLM) | Week 4 | Swap via settings when API locked |
| Embedding model | **`BAAI/bge-base-en-v1.5`** | E8 | test NDCG@10 **0.394** |
| Candidate retrieval / RRF | **Dense only** (no fusion in prod) | E1, E1b | Beats hybrid; weighted hybrid not worth latency |
| Candidate diversification policy | **Top-50 dense, no cap** (when rerank on in eval) | E3 | Test tied vs cap2 |
| Reranker and candidate budget | **Off** | E2, E5 skipped | CE flat @50; E4 Jev removed |
| Final context/result count | 10 | E1 | Week 4 protocol |
| LLM / prompt configuration | Ollama/Cohere per settings | E10 pending | Stub used for retrieval eval only |

---

## 7. Final held-out comparison

**Week 4 dense baseline (E1)** vs **frozen Week 5 design (E6+E7+E8)** — both dense, no rerank, document-level test qrels:

| Metric | Baseline (E1: 100w MiniLM) | Selected (200w BGE) | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| NDCG@10 | 0.3698 | **0.394** | **+0.024** | +6.5% |
| MRR@10 | 0.4496 | **0.477** | +0.027 | — |
| Recall@10 | 0.4304 | **0.447**† | +0.017 | — |
| Retrieval p50 | 197.9 ms | **~286 ms**‡ | +88 ms | +44% |
| Retrieval p95 | 204.5 ms | **~319 ms**‡ | — | — |
| End-to-end p50/p95 | TBD | TBD | TBD | TBD |
| Citation correctness | TBD (E10) | TBD | TBD | TBD |
| Estimated cost per query | TBD | TBD | TBD | TBD |

†BGE test from `week5_e8_bge_base_en_v1_5_test.json`. ‡E8 validation timing on BGE index; re-measure on test hardware when API is wired.

**Decomposition:** ~**+0.006** NDCG@10 attributable to **E6** chunk size (MiniLM @200w); ~**+0.018** additional from **E8** encoder on that chunking.

---

## 8. Error analysis

**E2a — dense rerank (Exp 4 vs 5):** 648 test queries — **181** helped, **178** hurt, **289** unchanged (ΔNDCG@10). Helps: reordering within a strong top-50 pool. Hurts: demotions + **duplicate chunk slots** (48 queries). No `missing_candidates` class in top-50.

Full JSON: `data/benchmarks/week4_exp5_dense_rerank_error_analysis.json`. Sheet: [E2a](experiments/E2a_rerank_error_analysis.md).

---

## 9. Decision log and limitations

| Decision | Evidence | Chosen option | Alternatives not chosen | Remaining limitation |
|---|---|---|---|---|
| First-stage retriever | E1 test NDCG@10 | **Dense** | BM25, equal hybrid, weighted hybrid | Hybrid not on full test for equal weights |
| Default rerank | E2 latency + NDCG | **Off** | dense+CE @50 (E2) | E3 done; E4 removed; E5 skipped |
| Fusion in production | E1b test | **None** | weighted 0.95/0.05 | Small NDCG tie on test |
| Chunking / embeddings | E6–E8 test | **200w / 20% / BGE** | 100w, 300w, MiniLM, overlap 10%/30% | API may still serve legacy `fiqa` until Week 6 |

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
| Dense + BGE (selected, 200w) | **0.394** |
| Dense MiniLM (Week 4, 100w) | 0.370 |
| BM25 | 0.19 |
| Dense + CE | 0.37 |

**Most important experiments:** E1 dense; E2 CE rejected; E3 control; **E6→E7** chunking; **E8** BGE (+0.018 test vs 200w MiniLM); E4/E5 out of scope/skipped.

**Full report:** this file. **Lessons (teaching):** [`rag_retrieval_lessons.md`](rag_retrieval_lessons.md). **Sheets:** [`experiments/README.md`](experiments/README.md). **Program catalog:** [`rag_experiment_program_report.md`](rag_experiment_program_report.md).

---

## 12. Lessons learned — improving RAG retrieval (summary)

See **[`rag_retrieval_lessons.md`](rag_retrieval_lessons.md)** for the full narrative. In short:

1. **Label your eval** — speed/overlap alone (E0) cannot choose a retriever.  
2. **Dense first** on semantic QA; fusion must **earn** its latency on test (E1, E1b).  
3. **Rerankers** can churn rankings without NDCG gain — analyze per-query before adopting (E2, E2a, E3).  
4. **Chunk size** matters a little; **overlap** is fine-tuning (E6, E7).  
5. **Better embeddings** often beat more chunk grid search (E8 vs E6/E7).  
6. **Structure-aware chunking** needs real structure in the source (E9 on FiQA).  
7. **Discipline:** one variable, validation tune, single test report, document rejections.

**On-disk indexes (intended):** `fiqa` + `fiqa_cs200_bge_base_en_v1_5` only (`cleanup_local_indexes.py`).
