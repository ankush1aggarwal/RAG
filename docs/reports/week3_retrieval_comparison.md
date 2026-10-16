# Week 3 — Retrieval comparison report (latency)

**Generated:** 2026-10-16 12:36 UTC  
**Corpus:** `BeIR/fiqa` (ingest limit `1000` documents)  
**Indexed:** `999` unique `doc_id`s, `1900` chunks  
**Collection:** `fiqa_sample`  
**Dense encoder:** `sentence_transformer` / `all-MiniLM-L6-v2`  
**Query set:** `data/benchmarks/fiqa_sample_query_set.json` (10 queries)  
**Top-k (final):** 5

## Timing protocol

- **Model loading in timed latency:** **no** (embedder must be constructed before `benchmark_query_set_timed`).

- **Model load time (reported separately):** 15724.6 ms

- **Warm-up:** `3` full passes per method on query `'broker fees commission'` (discarded).
- **Timed repeats:** `7` per query per method (70 samples per method total).
- **Latency stats:** pooled over all queries and repeats; **p50** = median, **p95** = 95th percentile.

## Candidate pool sizes

| Method | Candidates retrieved per query |
|--------|--------------------------------|
| Dense | 5 |
| BM25 | 5 |
| Hybrid (each list before RRF) | 15 |

## Latency (milliseconds)

| Method | p50 (median) | p95 | mean | min | max | n |
|--------|--------------|-----|------|-----|-----|---|
| dense (Qdrant) | 18.5 | 21.3 | 18.7 | 16.1 | 22.0 | 70 |
| BM25 | 23.4 | 27.7 | 23.4 | 18.5 | 28.6 | 70 |
| hybrid (RRF) | 41.9 | 46.4 | 42.1 | 36.5 | 48.5 | 70 |

## BM25 overlap eligibility pass (microbenchmark)

Isolated timing for lexical overlap **filter + sort** inside `Bm25ChunkIndex.search` (does **not** include `BM25Okapi.get_scores`). Legacy path rebuilds `set(corpus_tokens[i])` on each overlap check; current path uses frozensets built at index load and one overlap vector per query.

- **Query:** `broker fees commission`
- **Indexed chunks:** `1900`
- **Repeats (mean ms):** `50`

| Path | Mean (ms) |
|------|-----------|
| Legacy (re-tokenize per overlap check) | 5.309 |
| Precomputed token sets (current) | 0.382 |
| **Speedup** | **13.90×** |

## Top-k overlap (single quality pass, mean Jaccard)

- Dense vs BM25: **0.19**
- Dense vs hybrid: **0.39**
- BM25 vs hybrid: **0.46**

## Per-query latency (median ms across repeats)

| Query | dense p50 | BM25 p50 | hybrid p50 |
|-------|-----------|----------|------------|
| q01 | 18.1 | 23.4 | 40.4 |
| q02 | 19.1 | 23.5 | 40.6 |
| q03 | 19.6 | 25.7 | 44.1 |
| q04 | 17.9 | 21.8 | 39.8 |
| q05 | 18.4 | 23.3 | 44.4 |
| q06 | 18.5 | 24.7 | 44.0 |
| q07 | 17.4 | 24.2 | 41.2 |
| q08 | 18.2 | 24.6 | 41.4 |
| q09 | 18.7 | 20.3 | 38.8 |
| q10 | 19.2 | 21.6 | 41.8 |

## Interpretation

- Use **p50 / p95** for comparisons; ignore exploratory single-pass means when cold-start dominated.
- Overlap metrics come from one post-warm-up quality run (not averaged across repeats).
- BM25 overlap microbenchmark is a narrow slice of BM25 search; end-to-end BM25 p50/p95 still includes vector scoring and hit assembly.
- Add BEIR qrels in Week 4 for NDCG/MRR.
