# Week 4 — Experiment 3 (weighted hybrid tuning, validation)

**Generated:** 2026-10-19 11:00 UTC  
**Index collection:** `fiqa`  
**Dense encoder:** `sentence_transformer` / `all-MiniLM-L6-v2`  
**Corpus / index:** `BeIR/fiqa` **full corpus** (~57k documents)  
**Qrels split:** `validation`  
**Eval dataset:** `data/benchmarks/fiqa_qrels_validation.json`  
**Retrieval depth (top-k):** `10`  
**Queries evaluated:** `500`  

### Reproducibility protocol

```json
{
  "top_k": 10,
  "k_values": [
    1,
    5,
    10
  ],
  "generation_top_k": 10,
  "use_rerank": false,
  "reranker": null,
  "rerank_candidate_limit": 50,
  "fusion_weights": null,
  "llm_label": "eval_stub",
  "include_rag_responses": true
}
```

**Experiment 3:** Grid-search RRF weights on **validation** to maximize `ndcg@10` (score **0.3849**).

- **Selected weights (normalized):** dense=0.950, BM25=0.050

- Compare equal-weight **hybrid** vs **weighted hybrid** on validation.

## Quality metrics (mean over queries)

| Method | Metric | @1 | @5 | @10 |
|--------|--------|------|------|------|
| hybrid (RRF) | NDCG | 0.2900 | 0.3118 | 0.3358 |
| hybrid (RRF) | MRR | 0.2900 | 0.3729 | 0.3833 |
| hybrid (RRF) | MAP | 0.1702 | 0.2524 | 0.2677 |
| hybrid (RRF) | Precision | 0.2900 | 0.1380 | 0.0874 |
| hybrid (RRF) | Recall | 0.1702 | 0.3469 | 0.4163 |
| weighted hybrid (RRF) | NDCG | 0.3640 | 0.3658 | 0.3849 |
| weighted hybrid (RRF) | MRR | 0.3640 | 0.4390 | 0.4494 |
| weighted hybrid (RRF) | MAP | 0.2081 | 0.3042 | 0.3174 |
| weighted hybrid (RRF) | Precision | 0.3640 | 0.1612 | 0.0970 |
| weighted hybrid (RRF) | Recall | 0.2081 | 0.3861 | 0.4482 |

## Retrieval latency (milliseconds)

| Method | mean (ms) | p50 (ms) | p95 (ms) |
|--------|-----------|----------|----------|
| hybrid (RRF) | 793.9 | 771.9 | 1184.2 |
| weighted hybrid (RRF) | 905.4 | 874.1 | 1373.0 |

## Per-query examples

### `1` — Claiming business expenses for a business with no income

**Qrels relevant docs:** 14255

**hybrid (RRF)** — retrieval 843.5 ms, NDCG@10 0.431
- Ranked docs (top 5): 541809, 246461, 338700, 14255, 130631
- Citations in response: [541809:0], [246461:0], [338700:0], [14255:0], [130631:0], [562777:0], [392484:0], [100764:2], [196374:4], [47260:0]
- Top snippet: "No, your business cannot deduct your non-business expenses. You can only deduct from your business income those reasona...

**weighted hybrid (RRF)** — retrieval 775.7 ms, NDCG@10 0.631
- Ranked docs (top 5): 541809, 14255, 47260, 363495, 536849
- Citations in response: [541809:0], [14255:0], [47260:0], [363495:0], [536849:0], [581265:3], [130631:0], [338700:0], [212661:1], [246461:0]
- Top snippet: "No, your business cannot deduct your non-business expenses. You can only deduct from your business income those reasona...

### `1003` — The difference between Islamic Banks and Western Banks

**Qrels relevant docs:** 133388, 368233, 566830

**hybrid (RRF)** — retrieval 745.9 ms, NDCG@10 0.000
- Ranked docs (top 5): 396285, 180241, 206140, 503362, 374078
- Citations in response: [396285:0], [396285:1], [180241:2], [206140:0], [503362:0], [374078:0], [523415:2], [261846:0], [380071:0], [123535:3]
- Top snippet: The most relevant aspect of Islamic banking that distinguishes it from non-Islamic banking is the prohibition of chargin...

**weighted hybrid (RRF)** — retrieval 698.9 ms, NDCG@10 0.156
- Ranked docs (top 5): 396285, 180241, 374078, 261846, 264669
- Citations in response: [396285:0], [396285:1], [180241:2], [374078:0], [261846:0], [264669:2], [180241:1], [93518:4], [368233:0], [543580:0]
- Top snippet: The most relevant aspect of Islamic banking that distinguishes it from non-Islamic banking is the prohibition of chargin...

### `10048` — Higher mortgage to increase savings to invest?

**Qrels relevant docs:** 441353, 554814

**hybrid (RRF)** — retrieval 684.8 ms, NDCG@10 0.000
- Ranked docs (top 5): 493982, 59147, 423403, 493336, 387722
- Citations in response: [493982:1], [59147:6], [423403:0], [493336:0], [387722:0], [326542:0], [340791:0], [473427:1], [120283:1], [92403:6]
- Top snippet: goal than investing the money at a higher interest versus your mortgage interest rate. After your are completely debt fr...

**weighted hybrid (RRF)** — retrieval 636.7 ms, NDCG@10 0.000
- Ranked docs (top 5): 493982, 326542, 473427, 92403, 215149
- Citations in response: [493982:1], [326542:0], [473427:1], [92403:6], [215149:1], [59147:6], [440719:0], [338606:4], [234286:6], [81906:0]
- Top snippet: goal than investing the money at a higher interest versus your mortgage interest rate. After your are completely debt fr...
