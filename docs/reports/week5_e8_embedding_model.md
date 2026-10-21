# Week 5 — E8 embedding model comparison

**Chunk size / overlap:** 200 words / 20% (E6+E7)
**Qrels split:** `validation`
**Corpus:** `BeIR/fiqa` **full corpus** (~57k documents)
**Retriever:** dense (no rerank)

## Indexes

| Model | Collection | Dim | Points |
|-------|------------|----:|-------:|
| `all-MiniLM-L6-v2` | `fiqa_cs200` | 384 | 72,851 |
| `bge-base-en-v1.5` | `fiqa_cs200_bge_base_en_v1_5` | 768 | 72,851 |

## Retrieval metrics (document-level)

| Model | Metric | @1 | @5 | @10 |
|-------|--------|------|------|------|
| all-MiniLM-L6-v2 | NDCG | 0.3720 | 0.3645 | 0.3925 |
| all-MiniLM-L6-v2 | MRR | 0.3720 | 0.4491 | 0.4602 |
| all-MiniLM-L6-v2 | Recall | 0.2070 | 0.3825 | 0.4680 |
| all-MiniLM-L6-v2 | Cand. recall | 0.0000 | 0.0000 | 0.0000 |
| bge-base-en-v1.5 | NDCG | 0.3980 | 0.3845 | 0.4176 |
| bge-base-en-v1.5 | MRR | 0.3980 | 0.4742 | 0.4868 |
| bge-base-en-v1.5 | Recall | 0.2193 | 0.4085 | 0.5064 |
| bge-base-en-v1.5 | Cand. recall | 0.0000 | 0.0000 | 0.0000 |

## First-stage retrieval latency

| Model | mean (ms) | p50 (ms) | p95 (ms) |
|-------|----------:|---------:|---------:|
| all-MiniLM-L6-v2 | 119.6 | 118.6 | 125.3 |
| bge-base-en-v1.5 | 289.4 | 285.9 | 319.2 |

**Validation winner (NDCG@10):** `bge-base-en-v1.5` (`fiqa_cs200_bge_base_en_v1_5`).

**Test (winner, n=648):** NDCG@10 **0.394**, MRR@10 **0.477** (`week5_e8_bge_base_en_v1_5_test.json`). MiniLM on same chunking (E6/E7 test): **0.376**.

**Decision:** adopt **BGE-base**; expect ~**290 ms** p50 dense retrieval vs ~**119 ms** MiniLM.

## Insights

- **ΔNDCG@10 (test):** +**0.018** vs MiniLM on identical 200w/20% chunking (0.394 vs 0.376) — **larger than E6’s +0.006** from 100→200w.
- **Mechanism:** Encoder change moves **matching and ranking** for every chunk; Recall@10 val +0.038 suggests more queries get a relevant doc in the top 10, not just better order.
- **Trade-off:** Accept ~**2.4×** p50 retrieval latency for the quality gain; chunking (E6/E7) remains prerequisite, not a substitute.