# Week 4 — Experiment 2 (full corpus, validation)

**Generated:** 2026-10-19 10:32 UTC  
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

**Experiment 2:** Full-corpus index; BEIR **validation** qrels. Compare **dense**, **BM25**, and equal-weight **hybrid (RRF)**. Each query runs retrieval + stub LLM with **citations** from top chunks.

## Quality metrics (mean over queries)

| Method | Metric | @1 | @5 | @10 |
|--------|--------|------|------|------|
| dense (Qdrant) | NDCG | 0.3620 | 0.3580 | 0.3792 |
| dense (Qdrant) | MRR | 0.3620 | 0.4328 | 0.4446 |
| dense (Qdrant) | MAP | 0.2046 | 0.2968 | 0.3105 |
| dense (Qdrant) | Precision | 0.3620 | 0.1584 | 0.0964 |
| dense (Qdrant) | Recall | 0.2046 | 0.3781 | 0.4449 |
| BM25 | NDCG | 0.2060 | 0.1887 | 0.2103 |
| BM25 | MRR | 0.2060 | 0.2502 | 0.2615 |
| BM25 | MAP | 0.1140 | 0.1515 | 0.1625 |
| BM25 | Precision | 0.2060 | 0.0776 | 0.0510 |
| BM25 | Recall | 0.1140 | 0.1975 | 0.2630 |
| hybrid (RRF) | NDCG | 0.2900 | 0.3118 | 0.3358 |
| hybrid (RRF) | MRR | 0.2900 | 0.3729 | 0.3833 |
| hybrid (RRF) | MAP | 0.1702 | 0.2524 | 0.2677 |
| hybrid (RRF) | Precision | 0.2900 | 0.1380 | 0.0874 |
| hybrid (RRF) | Recall | 0.1702 | 0.3469 | 0.4163 |

## Retrieval latency (milliseconds)

| Method | mean (ms) | p50 (ms) | p95 (ms) |
|--------|-----------|----------|----------|
| dense (Qdrant) | 199.7 | 194.8 | 226.0 |
| BM25 | 622.6 | 600.0 | 1011.5 |
| hybrid (RRF) | 814.3 | 796.3 | 1200.0 |

## Per-query examples

### `1` — Claiming business expenses for a business with no income

**Qrels relevant docs:** 14255

**dense (Qdrant)** — retrieval 389.9 ms, NDCG@10 0.631
- Ranked docs (top 5): 541809, 14255, 47260, 363495, 536849
- Citations in response: [541809:0], [14255:0], [47260:0], [363495:0], [536849:0], [581265:3], [212661:1], [146657:1], [130631:0], [338700:0]
- Top snippet: "No, your business cannot deduct your non-business expenses. You can only deduct from your business income those reasona...

**BM25** — retrieval 560.0 ms, NDCG@10 0.000
- Ranked docs (top 5): 541809, 100764, 196374, 208989, 246461
- Citations in response: [541809:0], [100764:2], [196374:4], [208989:1], [246461:0], [523564:1], [196374:5], [289620:1], [562777:0], [307779:2]
- Top snippet: "No, your business cannot deduct your non-business expenses. You can only deduct from your business income those reasona...

**hybrid (RRF)** — retrieval 777.1 ms, NDCG@10 0.431
- Ranked docs (top 5): 541809, 246461, 338700, 14255, 130631
- Citations in response: [541809:0], [246461:0], [338700:0], [14255:0], [130631:0], [562777:0], [392484:0], [100764:2], [196374:4], [47260:0]
- Top snippet: "No, your business cannot deduct your non-business expenses. You can only deduct from your business income those reasona...

### `1003` — The difference between Islamic Banks and Western Banks

**Qrels relevant docs:** 133388, 368233, 566830

**dense (Qdrant)** — retrieval 239.3 ms, NDCG@10 0.156
- Ranked docs (top 5): 396285, 374078, 261846, 180241, 264669
- Citations in response: [396285:0], [396285:1], [374078:0], [261846:0], [180241:2], [264669:2], [180241:1], [93518:4], [368233:0], [543580:0]
- Top snippet: The most relevant aspect of Islamic banking that distinguishes it from non-Islamic banking is the prohibition of chargin...

**BM25** — retrieval 491.3 ms, NDCG@10 0.000
- Ranked docs (top 5): 396285, 523415, 180241, 380071, 123535
- Citations in response: [396285:1], [396285:0], [523415:2], [180241:2], [380071:0], [123535:3], [550939:4], [523415:1], [400532:0], [257691:0]
- Top snippet: institutions deploy and trade financial derivatives to provide the means to risk management. This is done on a very tech...

**hybrid (RRF)** — retrieval 730.3 ms, NDCG@10 0.000
- Ranked docs (top 5): 396285, 180241, 206140, 503362, 374078
- Citations in response: [396285:0], [396285:1], [180241:2], [206140:0], [503362:0], [374078:0], [523415:2], [261846:0], [380071:0], [123535:3]
- Top snippet: The most relevant aspect of Islamic banking that distinguishes it from non-Islamic banking is the prohibition of chargin...

### `10048` — Higher mortgage to increase savings to invest?

**Qrels relevant docs:** 441353, 554814

**dense (Qdrant)** — retrieval 204.1 ms, NDCG@10 0.000
- Ranked docs (top 5): 493982, 326542, 473427, 92403, 215149
- Citations in response: [493982:1], [326542:0], [473427:1], [92403:6], [215149:1], [440719:0], [338606:4], [234286:6], [59147:6], [81906:0]
- Top snippet: goal than investing the money at a higher interest versus your mortgage interest rate. After your are completely debt fr...

**BM25** — retrieval 422.5 ms, NDCG@10 0.000
- Ranked docs (top 5): 387722, 340791, 423403, 120283, 352178
- Citations in response: [387722:0], [340791:0], [423403:0], [120283:1], [352178:1], [257894:3], [490888:3], [105707:2], [154714:3], [493982:1]
- Top snippet: "The mortgage has a higher interest rate, how can it make sense to pay off the HELOC first?? As for the mutual fund, it ...

**hybrid (RRF)** — retrieval 631.7 ms, NDCG@10 0.000
- Ranked docs (top 5): 493982, 59147, 423403, 493336, 387722
- Citations in response: [493982:1], [59147:6], [423403:0], [493336:0], [387722:0], [326542:0], [340791:0], [473427:1], [120283:1], [92403:6]
- Top snippet: goal than investing the money at a higher interest versus your mortgage interest rate. After your are completely debt fr...
