"""Offline evaluation helpers (retrieval benchmarks, metrics runners)."""

from app.evaluation.retrieval_benchmark import (
    QuerySetSpec,
    benchmark_query_set,
    chunk_keys,
    jaccard_similarity,
    load_query_set,
    render_comparison_report,
)

__all__ = [
    "QuerySetSpec",
    "benchmark_query_set",
    "chunk_keys",
    "jaccard_similarity",
    "load_query_set",
    "render_comparison_report",
]
