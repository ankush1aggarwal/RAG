# Week 4 — Experiment 5 (test + local rerank)

**Generated:** 2026-10-19 12:50 UTC  
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
  "use_rerank": true,
  "reranker": "cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2",
  "rerank_candidate_limit": 50,
  "fusion_weights": null,
  "llm_label": "eval_stub",
  "include_rag_responses": true
}
```

**Experiment 5:** Same as Experiment 4 plus **local cross-encoder rerank** (`cross_encoder:cross-encoder/ms-marco-MiniLM-L-6-v2`) before generation.

- **Fusion weights (frozen):** dense=0.950, BM25=0.050

- Compare rerank overhead via latency table vs Experiment 4.

## Quality metrics (mean over queries)

| Method | Metric | @1 | @5 | @10 |
|--------|--------|------|------|------|
| dense (Qdrant) | NDCG | 0.3534 | 0.3438 | 0.3684 |
| dense (Qdrant) | MRR | 0.3534 | 0.4290 | 0.4407 |
| dense (Qdrant) | MAP | 0.1827 | 0.2776 | 0.2944 |
| dense (Qdrant) | Precision | 0.3534 | 0.1633 | 0.1006 |
| dense (Qdrant) | Recall | 0.1827 | 0.3618 | 0.4382 |
| BM25 | NDCG | 0.3009 | 0.2773 | 0.2886 |
| BM25 | MRR | 0.3009 | 0.3613 | 0.3679 |
| BM25 | MAP | 0.1548 | 0.2227 | 0.2313 |
| BM25 | Precision | 0.3009 | 0.1244 | 0.0725 |
| BM25 | Recall | 0.1548 | 0.2883 | 0.3247 |
| weighted hybrid (RRF) | NDCG | 0.3503 | 0.3435 | 0.3674 |
| weighted hybrid (RRF) | MRR | 0.3503 | 0.4265 | 0.4379 |
| weighted hybrid (RRF) | MAP | 0.1821 | 0.2772 | 0.2936 |
| weighted hybrid (RRF) | Precision | 0.3503 | 0.1636 | 0.1005 |
| weighted hybrid (RRF) | Recall | 0.1821 | 0.3637 | 0.4376 |

## Retrieval latency (milliseconds)

| Method | mean (ms) | p50 (ms) | p95 (ms) |
|--------|-----------|----------|----------|
| dense (Qdrant) | 1116.2 | 1083.0 | 1495.5 |
| BM25 | 1590.3 | 1592.5 | 2087.4 |
| weighted hybrid (RRF) | 1702.0 | 1668.8 | 2192.1 |

## Per-query examples

### `10034` — Tax implications of holding EWU (or other such UK ETFs) as a US citizen?

**Qrels relevant docs:** 181942, 480749

**dense (Qdrant)** — retrieval 1546.2 ms, NDCG@10 0.307
- Ranked docs (top 5): 180146, 459953, 181942, 561750, 545267
- Citations in response: [180146:0], [459953:4], [181942:0], [561750:2], [545267:0], [392313:0], [95390:2], [94477:1], [506448:2], [447197:1]
- Top snippet: As long as the relative in the UK is not a US citizen and isn't considered domiciled in the US, there will be no estate ...

**BM25** — retrieval 2089.0 ms, NDCG@10 0.264
- Ranked docs (top 5): 44955, 180146, 197478, 181942, 561750
- Citations in response: [44955:0], [180146:0], [197478:0], [181942:0], [561750:2], [561695:0], [506448:1], [120133:2], [437584:0], [95390:2]
- Top snippet: Transferring money you own from one place to another pretty much never has tax implications. It might have other implica...

**weighted hybrid (RRF)** — retrieval 2149.3 ms, NDCG@10 0.307
- Ranked docs (top 5): 180146, 459953, 181942, 561750, 545267
- Citations in response: [180146:0], [459953:4], [181942:0], [561750:2], [545267:0], [392313:0], [95390:2], [94477:1], [506448:2], [447197:1]
- Top snippet: As long as the relative in the UK is not a US citizen and isn't considered domiciled in the US, there will be no estate ...

### `10039` — Do individual investors use Google to obtain stock quotes?

**Qrels relevant docs:** 67785

**dense (Qdrant)** — retrieval 985.3 ms, NDCG@10 0.631
- Ranked docs (top 5): 171964, 67785, 383704, 53993, 177114
- Citations in response: [171964:1], [67785:0], [383704:0], [53993:2], [177114:0], [575018:0], [260726:0], [131421:1], [52940:0], [270027:0]
- Top snippet: sites (Google, Yahoo, MSN) and other sites that provide stock quotes. If you're looking to do basic comparisons versus i...

**BM25** — retrieval 1379.3 ms, NDCG@10 0.000
- Ranked docs (top 5): 171964, 177036, 574777, 376396, 131421
- Citations in response: [171964:1], [177036:1], [574777:0], [376396:0], [131421:1], [212572:0], [404222:0], [270027:0], [567531:1], [349866:2]
- Top snippet: sites (Google, Yahoo, MSN) and other sites that provide stock quotes. If you're looking to do basic comparisons versus i...

**weighted hybrid (RRF)** — retrieval 1509.3 ms, NDCG@10 0.631
- Ranked docs (top 5): 171964, 67785, 383704, 53993, 177114
- Citations in response: [171964:1], [67785:0], [383704:0], [53993:2], [177114:0], [575018:0], [260726:0], [131421:1], [52940:0], [404222:0]
- Top snippet: sites (Google, Yahoo, MSN) and other sites that provide stock quotes. If you're looking to do basic comparisons versus i...

### `10109` — Why does Charles Schwab have a Mandatory Settlement Period after selling stocks?

**Qrels relevant docs:** 156029, 28314, 406974, 499849, 506374, 566591, 93231

**dense (Qdrant)** — retrieval 965.4 ms, NDCG@10 0.535
- Ranked docs (top 5): 28314, 93231, 332243, 402726, 370635
- Citations in response: [28314:1], [93231:0], [332243:0], [28314:0], [402726:0], [370635:0], [226984:0], [36193:2], [566591:0], [563826:0]
- Top snippet: might still be two days until the money is received; because the other side also has three days. Guaranteed one day sett...

**BM25** — retrieval 1663.5 ms, NDCG@10 0.448
- Ranked docs (top 5): 28314, 93231, 332243, 266767, 402726
- Citations in response: [28314:1], [93231:0], [332243:0], [28314:0], [266767:0], [402726:0], [370635:0], [537212:1], [226984:0], [257832:0]
- Top snippet: might still be two days until the money is received; because the other side also has three days. Guaranteed one day sett...

**weighted hybrid (RRF)** — retrieval 1605.6 ms, NDCG@10 0.535
- Ranked docs (top 5): 28314, 93231, 332243, 402726, 370635
- Citations in response: [28314:1], [93231:0], [332243:0], [28314:0], [402726:0], [370635:0], [226984:0], [36193:2], [566591:0], [563826:0]
- Top snippet: might still be two days until the money is received; because the other side also has three days. Guaranteed one day sett...


## Dense rerank error analysis (Exp 4 vs Exp 5)

Comparison uses **dense retrieval only**: Experiment 4 (no rerank) vs Experiment 5 (CE rerank, top-50 candidates → top-10).

### Summary

- Queries: **648** | helped (ΔNDCG@10>0): **181** | hurt: **178** | unchanged: **289**

**Helped — primary cause counts**

| Cause | Count |
|-------|-------|
| `irrelevant_candidate_ordering` | 180 |
| `query_wording_chunk_mismatch` | 1 |

**Hurt — primary cause counts**

| Cause | Count |
|-------|-------|
| `irrelevant_candidate_ordering` | 128 |
| `duplicate_chunks` | 48 |
| `query_wording_chunk_mismatch` | 2 |

### Cause definitions

| Cause | Meaning |
|-------|---------|
| `missing_candidates` | Relevant doc not in dense top-N pool |
| `irrelevant_candidate_ordering` | Relevant doc in pool; @10 rank changes after CE |
| `duplicate_chunks` | Same `doc_id` in multiple of first 10 chunk slots |
| `query_wording_chunk_mismatch` | CE shifts top hit lexical overlap with query |

### Top 8 helped queries

#### `5155` (ΔNDCG@10=+1.0000, **irrelevant_candidate_ordering**)

**Q:** For insurance, why should you refuse $4,000/year for only 10 years and prefer $500/year indefinitely...

**Relevant docs:** 462892

- Dense @10: 590994, 83572, 286930, 129027, 457254...
- Rerank @10: 462892, 63892, 257715, 129027, 83572...
- **Diagnosis:** Relevant doc entered top-9 after rerank (was outside dense top-10 doc ranking).

#### `7096` (ΔNDCG@10=+1.0000, **irrelevant_candidate_ordering**)

**Q:** What's the formula for profits and losses when I delta hedge?

**Relevant docs:** 482238

- Dense @10: 315106, 346641, 272929, 477597, 266898...
- Rerank @10: 482238, 168006, 538580, 315106, 197863...
- **Diagnosis:** Relevant doc entered top-10 after rerank (was outside dense top-10 doc ranking).

#### `9643` (ΔNDCG@10=+1.0000, **irrelevant_candidate_ordering**)

**Q:** Is there any public data available to determine an ETF's holdings?

**Relevant docs:** 112208

- Dense @10: 295993, 71230, 386299, 448908, 276593...
- Rerank @10: 112208, 71230, 295993, 192299, 224765...
- **Diagnosis:** Relevant doc entered top-6 after rerank (was outside dense top-8 doc ranking).

#### `10558` (ΔNDCG@10=+0.8316, **irrelevant_candidate_ordering**)

**Q:** Investment strategy for 401k when rolling over soon

**Relevant docs:** 222485, 483268

- Dense @10: 493578, 105011, 526271, 379639, 188289...
- Rerank @10: 222485, 105011, 4181, 439402, 344526...
- **Diagnosis:** Relevant doc entered top-10 after rerank (was outside dense top-10 doc ranking).

#### `549` (ΔNDCG@10=+0.6845, **irrelevant_candidate_ordering**)

**Q:** Where to request ACH Direct DEBIT of funds from MY OWN personal bank account?

**Relevant docs:** 214024

- Dense @10: 376898, 240328, 221061, 210855, 131703...
- Rerank @10: 214024, 449279, 116684, 210855, 595633...
- **Diagnosis:** Relevant doc best rank moved from 8 (dense top-10) to 1 (reranked top-10) within the candidate pool.

#### `7754` (ΔNDCG@10=+0.6845, **irrelevant_candidate_ordering**)

**Q:** Inverse Relationship between Volatility and Beta

**Relevant docs:** 197527

- Dense @10: 79415, 260094, 471611, 272190, 78675...
- Rerank @10: 197527, 79415, 151902, 79998, 450949...
- **Diagnosis:** Relevant doc best rank moved from 8 (dense top-8) to 1 (reranked top-9) within the candidate pool.

#### `9556` (ΔNDCG@10=+0.6667, **irrelevant_candidate_ordering**)

**Q:** How does pre-market trading work?

**Relevant docs:** 37040

- Dense @10: 218670, 56742, 371720, 241059, 314478...
- Rerank @10: 37040, 218670, 241059, 362212, 265365...
- **Diagnosis:** Relevant doc best rank moved from 7 (dense top-10) to 1 (reranked top-10) within the candidate pool.

#### `4464` (ΔNDCG@10=+0.6309, **irrelevant_candidate_ordering**)

**Q:** Are Australian mutual fund fees large compared to US?

**Relevant docs:** 102904

- Dense @10: 14185, 402314, 191006, 49168, 555237...
- Rerank @10: 49168, 102904, 159471, 400845, 108081...
- **Diagnosis:** Relevant doc entered top-9 after rerank (was outside dense top-10 doc ranking).

### Top 8 hurt queries

#### `106` (ΔNDCG@10=-1.0000, **irrelevant_candidate_ordering**)

**Q:** What approaches are there for pricing a small business?

**Relevant docs:** 76695

- Dense @10: 76695, 525905, 585050, 61947, 336601...
- Rerank @10: 178874, 336601, 37134, 88201, 365410...
- **Diagnosis:** Relevant doc dropped out of top-9 after rerank (was rank 1 for dense).

#### `18` (ΔNDCG@10=-1.0000, **duplicate_chunks**)

**Q:** 1 EIN doing business under multiple business names

**Relevant docs:** 88124

- Dense @10: 88124, 377152, 348480, 366761, 506108...
- Rerank @10: 377152, 348480, 397510, 431685, 203820...
- **Diagnosis:** Dense top-10 chunk slots include 4 duplicate doc_id slot(s) (same document repeated), shrinking candidate diversity before rerank.

#### `5585` (ΔNDCG@10=-1.0000, **irrelevant_candidate_ordering**)

**Q:** Is there any site you can find out about the 'bonus features' of credit cards?

**Relevant docs:** 300117

- Dense @10: 300117, 13656, 335859, 183774, 501976...
- Rerank @10: 501976, 183774, 335859, 14731, 337817...
- **Diagnosis:** Relevant doc dropped out of top-10 after rerank (was rank 1 for dense).

#### `6625` (ΔNDCG@10=-1.0000, **duplicate_chunks**)

**Q:** What does a high theta mean for an option position?

**Relevant docs:** 88892

- Dense @10: 88892, 168006, 480337, 209492, 244448...
- Rerank @10: 168006, 480337, 572579, 371621, 16292...
- **Diagnosis:** Dense top-10 chunk slots include 2 duplicate doc_id slot(s) (same document repeated), shrinking candidate diversity before rerank.

#### `945` (ΔNDCG@10=-1.0000, **duplicate_chunks**)

**Q:** Paid by an American company but working from France: where should I pay taxes?

**Relevant docs:** 352052

- Dense @10: 352052, 332626, 237262, 185384, 187706...
- Rerank @10: 332626, 343477, 470879, 18003, 26182...
- **Diagnosis:** Dense top-10 chunk slots include 2 duplicate doc_id slot(s) (same document repeated), shrinking candidate diversity before rerank.

#### `9644` (ΔNDCG@10=-1.0000, **irrelevant_candidate_ordering**)

**Q:** If early exercise is a bad idea, why American option is more expensive than European [duplicate]

**Relevant docs:** 194605

- Dense @10: 194605, 358520, 353467, 484362, 72789...
- Rerank @10: 72789, 153728, 358520, 193303, 364575...
- **Diagnosis:** Relevant doc dropped out of top-9 after rerank (was rank 1 for dense).

#### `5343` (ΔNDCG@10=-0.6990, **irrelevant_candidate_ordering**)

**Q:** “International credit report” for French nationals?

**Relevant docs:** 2860

- Dense @10: 2860, 259228, 558388, 588153, 346166...
- Rerank @10: 118383, 184365, 486729, 588153, 204075...
- **Diagnosis:** Relevant doc best rank moved from 1 (dense top-9) to 9 (reranked top-10) within the candidate pool.

#### `9929` (ΔNDCG@10=-0.6731, **irrelevant_candidate_ordering**)

**Q:** Investing in commodities, pros and cons?

**Relevant docs:** 266323, 568006

- Dense @10: 266323, 160030, 85349, 545859, 568006...
- Rerank @10: 105790, 94408, 591757, 379140, 160030...
- **Diagnosis:** Relevant doc best rank moved from 1 (dense top-9) to 10 (reranked top-10) within the candidate pool.
