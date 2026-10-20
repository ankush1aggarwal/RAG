# E{N} — {Title}

| Field | Value |
|-------|--------|
| **ID** | E{N} |
| **Status** | Planned / In progress / Complete |
| **Week** | TBD |
| **Primary variable** | TBD |
| **Consolidated report** | [`rag_experiment_results.md`](../rag_experiment_results.md) §5 |

---

## 1. Question and hypothesis

- **Question:** TBD
- **Hypothesis:** TBD
- **Success criteria:** TBD on **validation** (including material **NDCG@10** gain vs baseline); pre-register p95/cost limits. **Test** run once after selection.

## 2. Control and treatments

| Arm | Description |
|-----|-------------|
| **Control** | TBD |
| **Treatment(s)** | TBD |

**Fixed settings (do not change across arms):** corpus snapshot, qrels split, chunking, embedding model, metric implementation, `top_k`, hardware, timing protocol.

## 3. Evaluation protocol

| Item | Value |
|------|--------|
| Corpus / collection | TBD |
| Chunks indexed | TBD |
| Qrels split / query count | TBD |
| Chunk size / overlap | TBD |
| Embedding / reranker | TBD |
| Candidate policy | TBD |
| Timing (warm-up, repeats) | TBD |

Document-level qrels: chunk hits deduplicated to first-seen `doc_id` per ranked list (same as Week 4 eval).

## 4. Results

### Aggregate metrics

| Arm | NDCG@10 | MRR@10 | Recall@10 | Candidate Recall@50 | p50 ms | p95 ms | Notes |
|-----|--------:|-------:|----------:|--------------------:|-------:|-------:|-------|
| Control | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Per-query analysis

TBD — link JSON/CSV, summarize failure modes and example query IDs.

## 5. Decision

| Outcome | Adopt / Reject / Defer |
| **Rationale** | TBD |
| **Production impact** | TBD |
| **Limitations** | TBD |

## 6. Artifacts and reproducibility

| Artifact | Path |
|----------|------|
| Per-query metrics JSON | TBD |
| Generated markdown report | TBD |
| Commit SHA | TBD |

```powershell
# Commands
TBD
```
