"""Retrieval comparison benchmark (dense vs BM25 vs hybrid) without qrels."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.ingestion.schemas import SearchHit
from app.pipeline.rag import retrieve_hits
from app.retrieval.bm25_store import Bm25ChunkIndex, tokenize
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.hybrid import resolve_hybrid_candidate_limit
from app.retrieval.qdrant_store import QdrantChunkStore

RetrievalMethod = Literal["dense", "bm25", "hybrid"]
ChunkKey = tuple[str, int]
LatencySummary = dict[RetrievalMethod, dict[str, float]]

METHODS: tuple[RetrievalMethod, ...] = ("dense", "bm25", "hybrid")
METHOD_LABELS = {
    "dense": "dense (Qdrant)",
    "bm25": "BM25",
    "hybrid": "hybrid (RRF)",
}


@dataclass(frozen=True)
class QuerySpec:
    id: str
    question: str
    theme: str = ""


@dataclass(frozen=True)
class QuerySetSpec:
    version: int
    description: str
    corpus: str
    corpus_limit: int
    collection: str
    top_k: int
    queries: list[QuerySpec]


@dataclass(frozen=True)
class MethodRun:
    method: RetrievalMethod
    latency_ms: float
    hits: list[SearchHit]


@dataclass(frozen=True)
class QueryBenchmarkResult:
    query: QuerySpec
    runs: dict[RetrievalMethod, MethodRun]


@dataclass(frozen=True)
class BenchmarkTimingConfig:
    warmup_repeats: int = 3
    timed_repeats_per_query: int = 7
    warmup_query: str = "broker fees commission"
    overlap_microbench_repeats: int = 50


@dataclass(frozen=True)
class Bm25OverlapEligibilityMicrobench:
    """Time to filter/sort chunks by lexical overlap (excludes BM25Okapi scoring)."""

    query: str
    indexed_chunks: int
    repeats: int
    legacy_mean_ms: float
    precomputed_mean_ms: float

    @property
    def speedup(self) -> float:
        if self.precomputed_mean_ms <= 0:
            return 0.0
        return self.legacy_mean_ms / self.precomputed_mean_ms


@dataclass(frozen=True)
class IndexStats:
    corpus_document_limit: int
    indexed_unique_documents: int
    indexed_chunks: int
    top_k: int
    dense_candidate_limit: int
    bm25_candidate_limit: int
    hybrid_candidate_limit_per_list: int


@dataclass(frozen=True)
class TimedBenchmarkResult:
    spec: QuerySetSpec
    config: BenchmarkTimingConfig
    index_stats: IndexStats
    latency_samples: dict[RetrievalMethod, list[float]]
    per_query_latency: dict[str, dict[RetrievalMethod, list[float]]]
    quality_results: list[QueryBenchmarkResult]
    model_load_ms: float | None
    model_loading_in_timed_latency: bool = False
    bm25_overlap_microbench: Bm25OverlapEligibilityMicrobench | None = None


def _legacy_overlap_filter_and_sort(index: Bm25ChunkIndex, query: str) -> None:
    """Previous pattern: ``set(corpus_tokens[i])`` on every overlap check."""
    query_token_set = set(tokenize(query))
    corpus_tokens = index._corpus_tokens
    n = index.num_chunks

    def overlap_count(chunk_index: int) -> int:
        return len(query_token_set & set(corpus_tokens[chunk_index]))

    eligible = [i for i in range(n) if overlap_count(i) > 0]
    eligible.sort(key=lambda i: overlap_count(i), reverse=True)


def _precomputed_overlap_filter_and_sort(index: Bm25ChunkIndex, query: str) -> None:
    query_token_set = set(tokenize(query))
    overlaps = [
        len(query_token_set & index._corpus_token_sets[i])
        for i in range(index.num_chunks)
    ]
    eligible = [i for i, count in enumerate(overlaps) if count > 0]
    eligible.sort(key=lambda i: overlaps[i], reverse=True)


def benchmark_bm25_overlap_eligibility_pass(
    bm25_index: Bm25ChunkIndex,
    query: str,
    *,
    repeats: int = 50,
) -> Bm25OverlapEligibilityMicrobench:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    start = time.perf_counter()
    for _ in range(repeats):
        _legacy_overlap_filter_and_sort(bm25_index, query)
    legacy_ms = (time.perf_counter() - start) * 1000.0 / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        _precomputed_overlap_filter_and_sort(bm25_index, query)
    precomputed_ms = (time.perf_counter() - start) * 1000.0 / repeats

    return Bm25OverlapEligibilityMicrobench(
        query=query,
        indexed_chunks=bm25_index.num_chunks,
        repeats=repeats,
        legacy_mean_ms=legacy_ms,
        precomputed_mean_ms=precomputed_ms,
    )


def load_query_set(path: str | Path) -> QuerySetSpec:
    target = Path(path)
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    queries = [
        QuerySpec(
            id=item["id"],
            question=item["question"],
            theme=item.get("theme", ""),
        )
        for item in payload["queries"]
    ]
    return QuerySetSpec(
        version=int(payload["version"]),
        description=str(payload.get("description", "")),
        corpus=str(payload.get("corpus", "BeIR/fiqa")),
        corpus_limit=int(payload.get("corpus_limit", 1000)),
        collection=str(payload.get("collection", "fiqa_sample")),
        top_k=int(payload.get("top_k", 5)),
        queries=queries,
    )


def chunk_keys(hits: list[SearchHit], limit: int | None = None) -> set[ChunkKey]:
    sliced = hits[:limit] if limit is not None else hits
    return {(hit.doc_id, hit.chunk_id) for hit in sliced}


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (p / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_index_stats(
    spec: QuerySetSpec,
    bm25_index: Bm25ChunkIndex,
    top_k: int | None = None,
) -> IndexStats:
    k = top_k if top_k is not None else spec.top_k
    hybrid_pool = resolve_hybrid_candidate_limit(k)
    return IndexStats(
        corpus_document_limit=spec.corpus_limit,
        indexed_unique_documents=bm25_index.unique_doc_count(),
        indexed_chunks=bm25_index.num_chunks,
        top_k=k,
        dense_candidate_limit=k,
        bm25_candidate_limit=k,
        hybrid_candidate_limit_per_list=hybrid_pool,
    )


def jaccard_similarity(left: set[ChunkKey], right: set[ChunkKey]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _time_retrieve_ms(
    question: str,
    method: RetrievalMethod,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int,
) -> float:
    start = time.perf_counter()
    retrieve_hits(
        question,
        retrieval_method=method,
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=top_k,
    )
    return (time.perf_counter() - start) * 1000.0


def warmup_retrieval(
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int,
    config: BenchmarkTimingConfig,
) -> None:
    for _ in range(config.warmup_repeats):
        for method in METHODS:
            _time_retrieve_ms(
                config.warmup_query,
                method,
                store=store,
                embedder=embedder,
                bm25_index=bm25_index,
                top_k=top_k,
            )


def benchmark_query_set_timed(
    spec: QuerySetSpec,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int | None = None,
    config: BenchmarkTimingConfig | None = None,
    model_load_ms: float | None = None,
) -> TimedBenchmarkResult:
    if bm25_index is None:
        raise ValueError("BM25 index is required for timed benchmark")
    timing = config or BenchmarkTimingConfig()
    k = top_k if top_k is not None else spec.top_k
    index_stats = build_index_stats(spec, bm25_index, k)

    warmup_retrieval(
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=k,
        config=timing,
    )

    latency_samples: dict[RetrievalMethod, list[float]] = {method: [] for method in METHODS}
    per_query_latency: dict[str, dict[RetrievalMethod, list[float]]] = {}

    for query in spec.queries:
        per_query_latency[query.id] = {method: [] for method in METHODS}
        for _ in range(timing.timed_repeats_per_query):
            for method in METHODS:
                elapsed = _time_retrieve_ms(
                    query.question,
                    method,
                    store=store,
                    embedder=embedder,
                    bm25_index=bm25_index,
                    top_k=k,
                )
                latency_samples[method].append(elapsed)
                per_query_latency[query.id][method].append(elapsed)

    quality_results = benchmark_query_set(
        spec,
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=k,
    )

    overlap_microbench = benchmark_bm25_overlap_eligibility_pass(
        bm25_index,
        timing.warmup_query,
        repeats=timing.overlap_microbench_repeats,
    )

    return TimedBenchmarkResult(
        spec=spec,
        config=timing,
        index_stats=index_stats,
        latency_samples=latency_samples,
        per_query_latency=per_query_latency,
        quality_results=quality_results,
        model_load_ms=model_load_ms,
        bm25_overlap_microbench=overlap_microbench,
    )


def summarize_timed_latencies(
    timed: TimedBenchmarkResult,
) -> LatencySummary:
    summary: LatencySummary = {}
    for method in METHODS:
        samples = timed.latency_samples[method]
        summary[method] = {
            "p50_ms": statistics.median(samples),
            "p95_ms": percentile(samples, 95),
            "mean_ms": statistics.mean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "n": float(len(samples)),
        }
    return summary


def _run_method(
    question: str,
    method: RetrievalMethod,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int,
) -> MethodRun:
    start = time.perf_counter()
    hits = retrieve_hits(
        question,
        retrieval_method=method,
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        top_k=top_k,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return MethodRun(method=method, latency_ms=elapsed_ms, hits=hits)


def benchmark_query(
    query: QuerySpec,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int,
) -> QueryBenchmarkResult:
    runs: dict[RetrievalMethod, MethodRun] = {}
    for method in METHODS:
        runs[method] = _run_method(
            query.question,
            method,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=top_k,
        )
    return QueryBenchmarkResult(query=query, runs=runs)


def benchmark_query_set(
    spec: QuerySetSpec,
    *,
    store: QdrantChunkStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25ChunkIndex | None,
    top_k: int | None = None,
) -> list[QueryBenchmarkResult]:
    k = top_k if top_k is not None else spec.top_k
    return [
        benchmark_query(
            query,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=k,
        )
        for query in spec.queries
    ]


def summarize_latencies(results: list[QueryBenchmarkResult]) -> LatencySummary:
    summary: LatencySummary = {}
    for method in METHODS:
        samples = [row.runs[method].latency_ms for row in results]
        summary[method] = {
            "mean_ms": statistics.mean(samples),
            "p50_ms": statistics.median(samples),
            "max_ms": max(samples),
        }
    return summary


def summarize_overlaps(
    results: list[QueryBenchmarkResult],
    top_k: int,
) -> dict[str, float]:
    pairs = [
        ("dense_vs_bm25", "dense", "bm25"),
        ("dense_vs_hybrid", "dense", "hybrid"),
        ("bm25_vs_hybrid", "bm25", "hybrid"),
    ]
    out: dict[str, float] = {}
    for label, left, right in pairs:
        scores = [
            jaccard_similarity(
                chunk_keys(row.runs[left].hits, top_k),
                chunk_keys(row.runs[right].hits, top_k),
            )
            for row in results
        ]
        out[f"{label}_jaccard_mean"] = statistics.mean(scores)
    hybrid_unique_vs_dense = []
    hybrid_unique_vs_bm25 = []
    for row in results:
        dense_keys = chunk_keys(row.runs["dense"].hits, top_k)
        bm25_keys = chunk_keys(row.runs["bm25"].hits, top_k)
        hybrid_keys = chunk_keys(row.runs["hybrid"].hits, top_k)
        hybrid_unique_vs_dense.append(len(hybrid_keys - dense_keys))
        hybrid_unique_vs_bm25.append(len(hybrid_keys - bm25_keys))
    out["hybrid_chunks_not_in_dense_topk_mean"] = statistics.mean(hybrid_unique_vs_dense)
    out["hybrid_chunks_not_in_bm25_topk_mean"] = statistics.mean(hybrid_unique_vs_bm25)
    return out


def _format_hit_line(rank: int, hit: SearchHit) -> str:
    snippet = hit.chunk_text.replace("\n", " ")
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    return (
        f"{rank}. `[{hit.doc_id}:{hit.chunk_id}]` score={hit.score:.4f} — {snippet}"
    )


def render_exploratory_report(
    spec: QuerySetSpec,
    results: list[QueryBenchmarkResult],
    *,
    embedding_method: str,
    embedding_model: str,
    generated_at: datetime | None = None,
) -> str:
    """Single-pass per query; latencies may include cold-start (exploratory only)."""
    when = generated_at or datetime.now(UTC)
    top_k = spec.top_k
    latency = summarize_latencies(results)
    overlaps = summarize_overlaps(results, top_k)

    lines: list[str] = [
        "# Week 3 — Retrieval comparison (exploratory snapshot)",
        "",
        "> **Note:** One timed run per query per method; means can be skewed by cold start.  ",
        "> For **median / p95** after warm-up and repeats, see `week3_retrieval_comparison.md`.",
        "",
        f"**Generated:** {when.strftime('%Y-%m-%d %H:%M UTC')}  ",
        (
            f"**Corpus:** `{spec.corpus}` (first `{spec.corpus_limit}` documents, "
            "same as `ingest.py` default)  "
        ),
        f"**Collection:** `{spec.collection}`  ",
        f"**Dense encoder:** `{embedding_method}` / `{embedding_model}`  ",
        (
            "**Query set:** `data/benchmarks/fiqa_sample_query_set.json` "
            f"({len(spec.queries)} queries)  "
        ),
        f"**Top-k:** {top_k}",
        "",
        "## Methodology",
        "",
        (
            "- Compare **dense** (Qdrant + embeddings), **BM25** (lexical), and "
            "**hybrid** (RRF over both candidate lists)."
        ),
        (
            "- No qrels in this slice: metrics are **latency**, **top-k chunk overlap** "
            "(Jaccard on `(doc_id, chunk_id)`), and **qualitative** top-hit snippets."
        ),
        (
            "- Re-run after re-ingest: `python src/scripts/ingest.py` then execute "
            "`notebooks/retrieval_benchmark.ipynb`."
        ),
        "",
        "## Latency (per query, milliseconds)",
        "",
        "| Method | Mean | P50 (median) | Max |",
        "|--------|------|--------------|-----|",
    ]
    for method in METHODS:
        row = latency[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['mean_ms']:.1f} | "
            f"{row['p50_ms']:.1f} | {row['max_ms']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Top-k overlap (mean Jaccard across queries)",
            "",
            f"- Dense vs BM25: **{overlaps['dense_vs_bm25_jaccard_mean']:.2f}**",
            f"- Dense vs hybrid: **{overlaps['dense_vs_hybrid_jaccard_mean']:.2f}**",
            f"- BM25 vs hybrid: **{overlaps['bm25_vs_hybrid_jaccard_mean']:.2f}**",
            "",
            "## Hybrid diversity",
            "",
            (
                f"- Mean hybrid hits **not** in dense top-{top_k}: "
                f"**{overlaps['hybrid_chunks_not_in_dense_topk_mean']:.2f}**"
            ),
            (
                f"- Mean hybrid hits **not** in BM25 top-{top_k}: "
                f"**{overlaps['hybrid_chunks_not_in_bm25_topk_mean']:.2f}**"
            ),
            "",
            "## Fixed query set",
            "",
            "| ID | Theme | Question |",
            "|----|-------|----------|",
        ]
    )
    for query in spec.queries:
        q = query.question.replace("|", "\\|")
        lines.append(f"| {query.id} | {query.theme} | {q} |")

    lines.extend(["", "## Per-query results", ""])
    for row in results:
        lines.append(f"### {row.query.id} — {row.query.theme}")
        lines.append("")
        lines.append(f"**Question:** {row.query.question}")
        lines.append("")
        for method in METHODS:
            run = row.runs[method]
            lines.append(f"**{METHOD_LABELS[method]}** ({run.latency_ms:.1f} ms)")
            if not run.hits:
                lines.append("- _(no hits)_")
            else:
                for rank, hit in enumerate(run.hits, start=1):
                    lines.append(f"- {_format_hit_line(rank, hit)}")
            lines.append("")

    lines.extend(
        [
            "## Interpretation notes",
            "",
            (
                "- **Low dense–BM25 Jaccard** is expected: lexical and semantic rankings "
                "diverge on acronym-heavy or paraphrased queries."
            ),
            (
                "- **Hybrid** should surface chunks boosted by both lists; overlap with "
                "dense-only is often high when dense already ranks well."
            ),
            (
                "- This report is **not** NDCG/MRR; add BEIR qrels in Week 4 for "
                "labeled retrieval metrics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


render_comparison_report = render_exploratory_report


def _render_bm25_overlap_microbench_section(
    bench: Bm25OverlapEligibilityMicrobench,
) -> list[str]:
    return [
        "## BM25 overlap eligibility pass (microbenchmark)",
        "",
        (
            "Isolated timing for lexical overlap **filter + sort** inside "
            "`Bm25ChunkIndex.search` (does **not** include `BM25Okapi.get_scores`). "
            "Legacy path rebuilds `set(corpus_tokens[i])` on each overlap check; "
            "current path uses frozensets built at index load and one overlap vector per query."
        ),
        "",
        f"- **Query:** `{bench.query}`",
        f"- **Indexed chunks:** `{bench.indexed_chunks}`",
        f"- **Repeats (mean ms):** `{bench.repeats}`",
        "",
        "| Path | Mean (ms) |",
        "|------|-----------|",
        f"| Legacy (re-tokenize per overlap check) | {bench.legacy_mean_ms:.3f} |",
        f"| Precomputed token sets (current) | {bench.precomputed_mean_ms:.3f} |",
        f"| **Speedup** | **{bench.speedup:.2f}×** |",
        "",
    ]


def render_latency_report(
    timed: TimedBenchmarkResult,
    *,
    embedding_method: str,
    embedding_model: str,
    generated_at: datetime | None = None,
) -> str:
    when = generated_at or datetime.now(UTC)
    spec = timed.spec
    stats = timed.index_stats
    latency = summarize_timed_latencies(timed)
    overlaps = summarize_overlaps(timed.quality_results, stats.top_k)
    cfg = timed.config
    total_samples = int(latency["dense"]["n"])

    lines: list[str] = [
        "# Week 3 — Retrieval comparison report (latency)",
        "",
        f"**Generated:** {when.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Corpus:** `{spec.corpus}` (ingest limit `{stats.corpus_document_limit}` documents)  ",
        (
            f"**Indexed:** `{stats.indexed_unique_documents}` unique `doc_id`s, "
            f"`{stats.indexed_chunks}` chunks  "
        ),
        f"**Collection:** `{spec.collection}`  ",
        f"**Dense encoder:** `{embedding_method}` / `{embedding_model}`  ",
        (
            "**Query set:** `data/benchmarks/fiqa_sample_query_set.json` "
            f"({len(spec.queries)} queries)  "
        ),
        f"**Top-k (final):** {stats.top_k}",
        "",
        "## Timing protocol",
        "",
        (
            f"- **Model loading in timed latency:** "
            f"{'yes' if timed.model_loading_in_timed_latency else '**no**'} "
            "(embedder must be constructed before `benchmark_query_set_timed`)."
        ),
        "",
    ]
    if timed.model_load_ms is not None:
        lines.append(f"- **Model load time (reported separately):** {timed.model_load_ms:.1f} ms")
        lines.append("")
    lines.extend(
        [
            f"- **Warm-up:** `{cfg.warmup_repeats}` full passes per method on query "
            f"`{cfg.warmup_query!r}` (discarded).",
            f"- **Timed repeats:** `{cfg.timed_repeats_per_query}` per query per method "
            f"({total_samples} samples per method total).",
            (
                "- **Latency stats:** pooled over all queries and repeats; "
                "**p50** = median, **p95** = 95th percentile."
            ),
            "",
            "## Candidate pool sizes",
            "",
            "| Method | Candidates retrieved per query |",
            "|--------|--------------------------------|",
            f"| Dense | {stats.dense_candidate_limit} |",
            f"| BM25 | {stats.bm25_candidate_limit} |",
            f"| Hybrid (each list before RRF) | {stats.hybrid_candidate_limit_per_list} |",
            "",
            "## Latency (milliseconds)",
            "",
            "| Method | p50 (median) | p95 | mean | min | max | n |",
            "|--------|--------------|-----|------|-----|-----|---|",
        ]
    )
    for method in METHODS:
        row = latency[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | "
            f"{row['mean_ms']:.1f} | {row['min_ms']:.1f} | {row['max_ms']:.1f} | {int(row['n'])} |"
        )

    if timed.bm25_overlap_microbench is not None:
        lines.extend([""] + _render_bm25_overlap_microbench_section(timed.bm25_overlap_microbench))

    lines.extend(
        [
            "",
            "## Top-k overlap (single quality pass, mean Jaccard)",
            "",
            f"- Dense vs BM25: **{overlaps['dense_vs_bm25_jaccard_mean']:.2f}**",
            f"- Dense vs hybrid: **{overlaps['dense_vs_hybrid_jaccard_mean']:.2f}**",
            f"- BM25 vs hybrid: **{overlaps['bm25_vs_hybrid_jaccard_mean']:.2f}**",
            "",
            "## Per-query latency (median ms across repeats)",
            "",
            "| Query | dense p50 | BM25 p50 | hybrid p50 |",
            "|-------|-----------|----------|------------|",
        ]
    )
    for query in spec.queries:
        row = timed.per_query_latency[query.id]
        lines.append(
            f"| {query.id} | {statistics.median(row['dense']):.1f} | "
            f"{statistics.median(row['bm25']):.1f} | {statistics.median(row['hybrid']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Use **p50 / p95** for comparisons; ignore exploratory single-pass means "
                "when cold-start dominated."
            ),
            (
                "- Overlap metrics come from one post-warm-up quality run "
                "(not averaged across repeats)."
            ),
            (
                "- BM25 overlap microbenchmark is a narrow slice of BM25 search; "
                "end-to-end BM25 p50/p95 still includes vector scoring and hit assembly."
            ),
            "- Add BEIR qrels in Week 4 for NDCG/MRR.",
            "",
        ]
    )
    return "\n".join(lines)


def timed_results_to_records(timed: TimedBenchmarkResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query_id, by_method in timed.per_query_latency.items():
        query = next(q for q in timed.spec.queries if q.id == query_id)
        for method in METHODS:
            for repeat_index, latency_ms in enumerate(by_method[method], start=1):
                records.append(
                    {
                        "query_id": query_id,
                        "theme": query.theme,
                        "method": method,
                        "repeat": repeat_index,
                        "latency_ms": latency_ms,
                    }
                )
    return records


def results_to_records(results: list[QueryBenchmarkResult]) -> list[dict[str, Any]]:
    """Flatten benchmark output for pandas."""
    records: list[dict[str, Any]] = []
    for row in results:
        for method in METHODS:
            run = row.runs[method]
            keys = chunk_keys(run.hits, None)
            records.append(
                {
                    "query_id": row.query.id,
                    "theme": row.query.theme,
                    "method": method,
                    "latency_ms": run.latency_ms,
                    "num_hits": len(run.hits),
                    "top_doc_ids": ",".join(hit.doc_id for hit in run.hits),
                    "top_keys": ",".join(f"{d}:{c}" for d, c in sorted(keys)),
                }
            )
    return records
