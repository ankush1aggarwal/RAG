"""Retrieval comparison benchmark (dense vs BM25 vs hybrid) without qrels."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.ingestion.schemas import SearchHit
from app.pipeline.rag import retrieve_hits
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.qdrant_store import QdrantChunkStore

RetrievalMethod = Literal["dense", "bm25", "hybrid"]
ChunkKey = tuple[str, int]

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


def jaccard_similarity(left: set[ChunkKey], right: set[ChunkKey]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


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


def summarize_latencies(results: list[QueryBenchmarkResult]) -> dict[RetrievalMethod, dict[str, float]]:
    summary: dict[RetrievalMethod, dict[str, float]] = {}
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


def render_comparison_report(
    spec: QuerySetSpec,
    results: list[QueryBenchmarkResult],
    *,
    embedding_method: str,
    embedding_model: str,
    generated_at: datetime | None = None,
) -> str:
    when = generated_at or datetime.now(timezone.utc)
    top_k = spec.top_k
    latency = summarize_latencies(results)
    overlaps = summarize_overlaps(results, top_k)

    lines: list[str] = [
        "# Week 3 — Retrieval comparison report",
        "",
        f"**Generated:** {when.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Corpus:** `{spec.corpus}` (first `{spec.corpus_limit}` documents, same as `ingest.py` default)  ",
        f"**Collection:** `{spec.collection}`  ",
        f"**Dense encoder:** `{embedding_method}` / `{embedding_model}`  ",
        f"**Query set:** `data/benchmarks/fiqa_sample_query_set.json` ({len(spec.queries)} queries)  ",
        f"**Top-k:** {top_k}",
        "",
        "## Methodology",
        "",
        "- Compare **dense** (Qdrant + embeddings), **BM25** (lexical), and **hybrid** (RRF over both candidate lists).",
        "- No qrels in this slice: metrics are **latency**, **top-k chunk overlap** (Jaccard on `(doc_id, chunk_id)`), and **qualitative** top-hit snippets.",
        "- Re-run after re-ingest: `python src/scripts/ingest.py` then execute `notebooks/retrieval_benchmark.ipynb`.",
        "",
        "## Latency (per query, milliseconds)",
        "",
        "| Method | Mean | P50 (median) | Max |",
        "|--------|------|--------------|-----|",
    ]
    for method in METHODS:
        row = latency[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['mean_ms']:.1f} | {row['p50_ms']:.1f} | {row['max_ms']:.1f} |"
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
            f"- Mean hybrid hits **not** in dense top-{top_k}: **{overlaps['hybrid_chunks_not_in_dense_topk_mean']:.2f}**",
            f"- Mean hybrid hits **not** in BM25 top-{top_k}: **{overlaps['hybrid_chunks_not_in_bm25_topk_mean']:.2f}**",
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
            "- **Low dense–BM25 Jaccard** is expected: lexical and semantic rankings diverge on acronym-heavy or paraphrased queries.",
            "- **Hybrid** should surface chunks boosted by both lists; overlap with dense-only is often high when dense already ranks well.",
            "- This report is **not** NDCG/MRR; add BEIR qrels in Week 4 for labeled retrieval metrics.",
            "",
        ]
    )
    return "\n".join(lines)


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
