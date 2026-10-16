"""Offline evaluation helpers (retrieval benchmarks, metrics runners)."""

from app.evaluation.retrieval_benchmark import (
    BenchmarkTimingConfig,
    QuerySetSpec,
    TimedBenchmarkResult,
    benchmark_query_set,
    benchmark_query_set_timed,
    chunk_keys,
    jaccard_similarity,
    load_query_set,
    percentile,
    render_comparison_report,
    render_exploratory_report,
    render_latency_report,
    timed_results_to_records,
)

__all__ = [
    "BenchmarkTimingConfig",
    "QuerySetSpec",
    "TimedBenchmarkResult",
    "benchmark_query_set",
    "benchmark_query_set_timed",
    "chunk_keys",
    "jaccard_similarity",
    "load_query_set",
    "percentile",
    "render_comparison_report",
    "render_exploratory_report",
    "render_latency_report",
    "timed_results_to_records",
]
