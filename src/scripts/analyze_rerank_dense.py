"""Error analysis: dense retrieval with vs without rerank (Exp 4 vs Exp 5 artifacts)."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.evaluation.rerank_error_analysis import (
    analyze_dense_rerank,
    append_analysis_to_report,
    render_rerank_error_analysis_report,
)
from app.retrieval import QdrantChunkStore, create_embedding_provider, create_reranker_provider

BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_PATH = Path(_REPO_ROOT) / "docs" / "reports" / "week4_exp5_test_rerank.md"
PROGRAM_REPORT = Path(_REPO_ROOT) / "docs" / "reports" / "week4_retrieval_program_report.md"


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Dense rerank error analysis (Exp 4 vs Exp 5)")
    parser.add_argument("--collection", default="fiqa")
    parser.add_argument("--deep-dive", type=int, default=8)
    parser.add_argument(
        "--baseline-json",
        default=str(BENCHMARK_DIR / "week4_exp4_dense_test.json"),
    )
    parser.add_argument(
        "--rerank-json",
        default=str(BENCHMARK_DIR / "week4_exp5_dense_test.json"),
    )
    parser.add_argument(
        "--eval-dataset",
        default=str(BENCHMARK_DIR / "fiqa_qrels_test.json"),
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    rerank_path = Path(args.rerank_json)
    if not baseline_path.is_file() or not rerank_path.is_file():
        raise SystemExit("Missing Exp 4/5 dense JSON artifacts; run experiments first.")

    qdrant_path = settings.resolved_qdrant_path()
    embedder = create_embedding_provider(
        settings.embedding_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )
    reranker = create_reranker_provider(
        "cross_encoder",
        cross_encoder_model=settings.cross_encoder_rerank_model,
    )
    if reranker is None:
        raise SystemExit("cross_encoder reranker unavailable")

    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    try:
        print("Analyzing dense rerank deltas (re-fetching candidates where dNDCG@10 != 0)...")
        classified, summary = analyze_dense_rerank(
            dataset_path=Path(args.eval_dataset),
            baseline_result_path=baseline_path,
            rerank_result_path=rerank_path,
            store=store,
            embedder=embedder,
            reranker=reranker,
            candidate_limit=settings.rerank_candidate_limit,
            top_k=10,
            deep_dive_limit=args.deep_dive,
        )
    finally:
        store.close()

    out_json = BENCHMARK_DIR / "week4_exp5_dense_rerank_error_analysis.json"
    payload = {
        "summary": summary,
        "queries": [
            {
                "query_id": c.query_id,
                "direction": c.direction,
                "delta_ndcg10": c.delta_ndcg10,
                "primary_cause": c.primary_cause.value,
                "detail": c.detail,
            }
            for c in classified
            if c.direction != "unchanged"
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    section = render_rerank_error_analysis_report(
        classified,
        summary,
        deep_dive_limit=args.deep_dive,
    )
    append_analysis_to_report(REPORT_PATH, section)
    print(f"Updated {REPORT_PATH}")

    if PROGRAM_REPORT.is_file():
        prog = PROGRAM_REPORT.read_text(encoding="utf-8")
        link = (
            "- Dense rerank error analysis (Exp 4 vs Exp 5 dense): "
            "[week4_exp5_test_rerank.md](week4_exp5_test_rerank.md) "
            "(section *Dense rerank error analysis*)"
        )
        if "Dense rerank error analysis" not in prog:
            prog = prog.rstrip() + "\n\n## Additional analysis\n\n" + link + "\n"
            PROGRAM_REPORT.write_text(prog, encoding="utf-8")
            print(f"Updated {PROGRAM_REPORT}")


if __name__ == "__main__":
    main()
