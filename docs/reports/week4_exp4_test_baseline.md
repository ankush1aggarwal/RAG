# Week 4 — Experiment 4 (test baseline, no rerank)

**Generated:** 2026-10-19 11:34 UTC  
**Index collection:** `fiqa`  
**Dense encoder:** `sentence_transformer` / `all-MiniLM-L6-v2`  
**Corpus / index:** `BeIR/fiqa` **full corpus** (~57k documents)  
**Qrels split:** `test`  
**Eval dataset:** `data/benchmarks/fiqa_qrels_test.json`  
**Retrieval depth (top-k):** `10`  
**Queries evaluated:** `648`  

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

**Experiment 4:** Held-out **test** qrels; **no reranker**. Methods: **dense**, **BM25**, **weighted hybrid** (frozen weights from Exp 3).

- **Fusion weights:** dense=0.950, BM25=0.050

- Test set touched once for final baseline reporting.

## Quality metrics (mean over queries)

| Method | Metric | @1 | @5 | @10 |
|--------|--------|------|------|------|
| dense (Qdrant) | NDCG | 0.3611 | 0.3472 | 0.3698 |
| dense (Qdrant) | MRR | 0.3611 | 0.4398 | 0.4496 |
| dense (Qdrant) | MAP | 0.1928 | 0.2818 | 0.2976 |
| dense (Qdrant) | Precision | 0.3611 | 0.1611 | 0.0997 |
| dense (Qdrant) | Recall | 0.1928 | 0.3594 | 0.4304 |
| BM25 | NDCG | 0.1728 | 0.1727 | 0.1913 |
| BM25 | MRR | 0.1728 | 0.2294 | 0.2386 |
| BM25 | MAP | 0.0868 | 0.1311 | 0.1413 |
| BM25 | Precision | 0.1728 | 0.0833 | 0.0539 |
| BM25 | Recall | 0.0868 | 0.1885 | 0.2434 |
| weighted hybrid (RRF) | NDCG | 0.3534 | 0.3478 | 0.3691 |
| weighted hybrid (RRF) | MRR | 0.3534 | 0.4375 | 0.4474 |
| weighted hybrid (RRF) | MAP | 0.1862 | 0.2804 | 0.2955 |
| weighted hybrid (RRF) | Precision | 0.3534 | 0.1636 | 0.1008 |
| weighted hybrid (RRF) | Recall | 0.1862 | 0.3642 | 0.4308 |

## Retrieval latency (milliseconds)

| Method | mean (ms) | p50 (ms) | p95 (ms) |
|--------|-----------|----------|----------|
| dense (Qdrant) | 198.7 | 197.9 | 204.5 |
| BM25 | 604.8 | 607.7 | 936.0 |
| weighted hybrid (RRF) | 815.0 | 809.2 | 1172.0 |

## Per-query examples

### `10034` — Tax implications of holding EWU (or other such UK ETFs) as a US citizen?

**Qrels relevant docs:** 181942, 480749

**dense (Qdrant)** — retrieval 375.1 ms, NDCG@10 0.000
- Ranked docs (top 5): 188816, 510599, 392313, 561750, 592926
- Citations in response: [188816:8], [510599:0], [392313:0], [561750:2], [592926:0], [562007:0], [104128:0], [459953:4], [72135:0], [447197:0]
- Top snippet: the issue is very complex. Taxes are complex everywhere, but especially so in the US. I suggest you not do anything befo...

**BM25** — retrieval 758.9 ms, NDCG@10 0.000
- Ranked docs (top 5): 44955, 197478, 180146, 597519, 145999
- Citations in response: [44955:0], [197478:0], [180146:0], [597519:0], [145999:0], [561695:0], [198532:0], [381884:0], [67625:1], [103176:0]
- Top snippet: Transferring money you own from one place to another pretty much never has tax implications. It might have other implica...

**weighted hybrid (RRF)** — retrieval 1017.1 ms, NDCG@10 0.193
- Ranked docs (top 5): 188816, 510599, 392313, 561750, 592926
- Citations in response: [188816:8], [510599:0], [392313:0], [561750:2], [592926:0], [562007:0], [104128:0], [181942:0], [459953:4], [72135:0]
- Top snippet: the issue is very complex. Taxes are complex everywhere, but especially so in the US. I suggest you not do anything befo...

### `10039` — Do individual investors use Google to obtain stock quotes?

**Qrels relevant docs:** 67785

**dense (Qdrant)** — retrieval 268.5 ms, NDCG@10 0.500
- Ranked docs (top 5): 575018, 90519, 67785, 383704, 314235
- Citations in response: [575018:0], [90519:4], [67785:0], [383704:0], [314235:0], [390529:1], [138790:0], [6826:0], [53263:0], [3669:0]
- Top snippet: Yup. What I wanted to know was where they are pulling it up from. Have casually used Google finance for personal investm...

**BM25** — retrieval 502.1 ms, NDCG@10 0.000
- Ranked docs (top 5): 171964, 177036, 5572, 587959, 131421
- Citations in response: [171964:1], [177036:1], [5572:4], [587959:0], [131421:1], [349866:2], [567531:0], [216757:4], [172783:0], [376396:0]
- Top snippet: sites (Google, Yahoo, MSN) and other sites that provide stock quotes. If you're looking to do basic comparisons versus i...

**weighted hybrid (RRF)** — retrieval 753.0 ms, NDCG@10 0.500
- Ranked docs (top 5): 575018, 90519, 67785, 383704, 314235
- Citations in response: [575018:0], [90519:4], [67785:0], [383704:0], [314235:0], [390529:1], [53263:0], [171964:1], [138790:0], [6826:0]
- Top snippet: Yup. What I wanted to know was where they are pulling it up from. Have casually used Google finance for personal investm...

### `10109` — Why does Charles Schwab have a Mandatory Settlement Period after selling stocks?

**Qrels relevant docs:** 156029, 28314, 406974, 499849, 506374, 566591, 93231

**dense (Qdrant)** — retrieval 214.9 ms, NDCG@10 0.504
- Ranked docs (top 5): 93231, 332243, 28314, 563826, 591757
- Citations in response: [93231:0], [332243:0], [28314:0], [563826:0], [28314:1], [591757:2], [370635:0], [566591:0], [402726:0], [115817:4]
- Top snippet: quid's answer explains the settlement period well. However, it should be noted that you can avoid the settlement period ...

**BM25** — retrieval 611.7 ms, NDCG@10 0.198
- Ranked docs (top 5): 266725, 555047, 113644, 570248, 93231
- Citations in response: [266725:1], [555047:0], [113644:0], [570248:0], [93231:0], [266767:0], [28314:1], [417133:1], [563826:0], [108671:0]
- Top snippet: that are in your Charles Schwab account using some other broker. However, you can (possibly now, definitely eventually, ...

**weighted hybrid (RRF)** — retrieval 805.7 ms, NDCG@10 0.499
- Ranked docs (top 5): 93231, 563826, 28314, 332243, 402726
- Citations in response: [93231:0], [563826:0], [28314:1], [332243:0], [28314:0], [402726:0], [591757:2], [370635:0], [566591:0], [36193:2]
- Top snippet: quid's answer explains the settlement period well. However, it should be noted that you can avoid the settlement period ...
