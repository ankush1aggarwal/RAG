# E4 — Jev reranker (ranking only, hosted API)

> **Removed from program (2026-10-20):** Hosted Jev API is not free; experiment not run. Rerank conclusions remain **E2** (local CE only). Week 5 continues with **E6–E8** per [`README.md`](README.md).

| Field | Value |
|-------|--------|
| **ID** | E4 |
| **Status** | **Removed** — out of scope (hosted cost) |
| **Primary variable** | Reranker backend: local CE (E2 control) vs **Jev** (`jev-reranker`, ranking-only) |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

## 1. Question and hypothesis

- **Question:** After fixing the candidate pool (E3), does **Jev** improve document-level ranking vs **no rerank** and vs **local cross-encoder** (E2)?
- **Hypothesis:** A stronger hosted reranker may recover ordering gains that CE missed on FiQA, without changing first-stage dense retrieval.
- **Scope (this experiment):** **Ranking only** — use Jev **`rerank()`** mode with **no score-based relevance filtering**, so the only effect is **re-ordering** the same candidate passages (isolates ordering vs filtering).

## 2. Jev capabilities (context; not all tested here)

The [`jev-reranker`](https://pypi.org/project/jev-reranker/) library supports:

| Capability | E4 uses? | Notes |
|------------|----------|--------|
| **Reranking** (`rerank()`) | **Yes** | Listwise or pointwise scoring — pick one mode and hold fixed for the run |
| **Relevance filtering** (score thresholds) | **No** | Optional follow-up: [**E4b**](E4b_jev_relevance_filtering.md) |
| **Listwise scoring** | TBD | Document chosen mode in protocol table |
| **Pointwise scoring** | TBD | Alternative arm or defer to a second validation pass |

**Do not** enable filtering in E4; comparing filtered vs unfiltered pools confounds ordering with pool size.

## 3. Control and treatments

| Arm | Description |
|-----|-------------|
| **Control A** | Dense, **no rerank** (E1 test baseline under E3 policy) |
| **Control B** | Dense + **local CE** @50 candidates (E2 protocol, E3 policy) |
| **Treatment** | Dense + **Jev** `rerank()` @50 candidates, E3 policy, **no filtering** |

**Fixed:** full `fiqa` index, `top_k=10`, document-level qrels, E3-winning candidate diversification policy, candidate budget **50** (depth sweep deferred to **E5**).

## 4. Hosted API protocol (record before running)

| Item | Value |
|------|--------|
| Provider / library | Jev — `jev-reranker` |
| API model name & version | TBD (exact string from provider dashboard) |
| Scoring mode | `listwise` **or** `pointwise` (one per run; state in results) |
| Instructions / system prompt (if any) | TBD — paste verbatim; treat as part of reproducibility |
| Max concurrency | TBD (e.g. 1 vs N parallel requests) |
| Retries / timeout | TBD |
| Queries sent | FiQA **validation** for model/mode selection; **test** once for reporting |
| Estimated cost | TBD — API calls × price; tokens/chunks if billed per unit |
| Data handling | TBD — provider retention, logging, subprocessors, whether query+passage text leaves your environment; link to Jev/privacy terms |

**Latency:** report p50/p95 **rerank stage** separately from dense retrieval; include API round-trip and client queueing.

## 5. Results

### Validation (select listwise vs pointwise if both tried)

| Arm | NDCG@10 | MRR@10 | Recall@10 | Cand. Recall@50 | Rerank p50/p95 | API cost (val) | Notes |
|-----|--------:|-------:|----------:|----------------:|----------------|---------------|-------|
| No rerank | TBD | TBD | TBD | TBD | — | $0 | E3 policy |
| CE @50 | TBD | TBD | TBD | TBD | TBD | $0 local | |
| Jev `rerank()` @50 | TBD | TBD | TBD | TBD | TBD | TBD | no filter |

### Test (frozen arm from validation)

| Arm | NDCG@10 | MRR@10 | Recall@10 | Rerank p50/p95 | API cost (test) |
|-----|--------:|-------:|----------:|----------------|----------------|
| Winner vs no rerank | TBD | TBD | TBD | TBD | TBD |

### Per-query analysis

TBD — reuse E2a-style ΔNDCG vs no rerank for Jev; note API failures/timeouts by query ID.

## 6. Decision

| Outcome | TBD |
| **Adopt rerank in prod?** | Only if test NDCG gain is material at acceptable p95 **and** cost/latency/data policy acceptable |
| **If Jev wins over CE** | Run **E5** depth sweep on **Jev**; if neither beats no rerank, **skip E5** rerank depth |
| **If CE wins under E3** | Optional: skip Jev test repeat or document Jev as rejected on validation |

## 7. Artifacts and reproducibility

| Artifact | Path |
|----------|------|
| Per-query JSON | TBD — `data/benchmarks/week5_e4_jev_*.json` |
| API audit log (redacted) | TBD — model version, timestamps, concurrency, call counts |
| Commit SHA | TBD |

```powershell
# TBD — eval script / CLI flags once Jev provider wired in src/app/retrieval/
```
