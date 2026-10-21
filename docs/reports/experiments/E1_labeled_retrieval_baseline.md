# E1 — Dense vs BM25 vs hybrid (FiQA qrels)

| Field | Value |
|-------|--------|
| **ID** | E1 |
| **Status** | Complete |
| **Week** | 4 |
| **Primary variable** | Retrieval method |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §4–5 |

---

## 1. Question and hypothesis

- **Question:** On full-corpus FiQA with BEIR qrels, which first-stage retriever maximizes document-level NDCG@10?
- **Hypothesis:** Dense embeddings outperform BM25; equal-weight hybrid may help recall but at higher latency.
- **Success criteria:** Compare validation (tuning) and **held-out test** (report once); prefer higher NDCG@10 at acceptable p95 latency.

## 2. Control and treatments

| Arm | Description |
|-----|-------------|
| **Control / primary** | Dense (Qdrant), top-k=10 |
| BM25 | In-memory BM25 over chunks |
| hybrid (RRF) | Equal-weight dense + BM25 fusion |

**Fixed:** 57,638 docs → 111,170 chunks; chunk 100 words / 20% overlap; `all-MiniLM-L6-v2` (384-d); no rerank; document-level metrics from chunk ranks.

**Smoke run (not used for decisions):** 15 validation queries on 1k-doc `fiqa_sample` — [`week4_retrieval_eval.md`](../week4_retrieval_eval.md).

## 3. Evaluation protocol

| Item | Value |
|------|--------|
| Collection | `fiqa` (full corpus) |
| Validation | 500 queries — `data/benchmarks/fiqa_qrels_validation.json` |
| Test | 648 queries — `data/benchmarks/fiqa_qrels_test.json` (Exp 4 dense/BM25; hybrid equal-weight on validation only) |
| `top_k` | 10 |

## 4. Results

### Validation (Exp 2)

| Method | NDCG@10 | MRR@10 | Recall@10 | p50 ms | p95 ms |
|--------|--------:|-------:|----------:|-------:|-------:|
| dense | **0.3792** | 0.4446 | 0.4449 | 194.8 | 226.0 |
| BM25 | 0.2103 | 0.2615 | 0.2630 | 600.0 | 1011.5 |
| hybrid (RRF) | 0.3358 | 0.3833 | 0.4163 | 796.3 | 1200.0 |

### Test (Exp 4 — dense & BM25; no rerank)

| Method | NDCG@10 | MRR@10 | Recall@10 | p50 ms | p95 ms |
|--------|--------:|-------:|----------:|-------:|-------:|
| dense | **0.3698** | 0.4496 | 0.4304 | 197.9 | 204.5 |
| BM25 | 0.1913 | 0.2386 | 0.2434 | 607.7 | 936.0 |

**Candidate Recall@50:** not logged in Exp 2/4 (added in later eval protocol from E3 onward).

### Per-query analysis

See per-query tables and examples in [`week4_exp2_full_corpus_validation.md`](../week4_exp2_full_corpus_validation.md) and [`week4_exp4_test_baseline.md`](../week4_exp4_test_baseline.md). JSON: `data/benchmarks/week4_exp*.json`.

## 5. Insights (interpretation)

- **Effect size:** Test NDCG@10 dense **0.370** vs BM25 **0.191** (~**+0.18**); hybrid validation **0.336** — below dense despite higher Recall@10 on val.
- **Mechanism:** FiQA labels are document-level Q&A; **semantic dense** retrieval matches paraphrase and topic better than lexical BM25; equal RRF dilutes dense signal and adds fusion latency without beating dense on NDCG.
- **Interaction:** Becomes the **Week 4–5 retrieval-method baseline** (100w / 20% / MiniLM) for E2–E8.
- **Caveats:** Equal-weight hybrid not on full test; candidate Recall@50 not logged in Exp 2/4.

## 6. Decision

| Outcome | **Adopt dense** as first-stage production retriever |
| **Rationale** | Best NDCG@10 on validation and test; lowest latency. Hybrid below dense on validation; BM25 much weaker. |
| **Production impact** | Frozen profile: `dense`, `top_k=10`, rerank off (see E2). |
| **Limitations** | Equal-weight hybrid not re-run on full test set; weighted hybrid covered in E1b. |

## 7. Artifacts and reproducibility

```powershell
$env:FIQA_CORPUS_LIMIT="0"
python src/scripts/run_week4_experiments.py exp2 --collection fiqa
python src/scripts/run_week4_experiments.py exp4 --collection fiqa
```

| Artifact | Path |
|----------|------|
| Protocol snapshot | `data/benchmarks/week4_eval_protocol.json` |
| Commit (report era) | `6210bdd` (update when re-run) |
