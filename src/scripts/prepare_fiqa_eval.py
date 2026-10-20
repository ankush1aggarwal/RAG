"""Download FiQA qrels + queries and write a local eval JSON for retrieval metrics."""

import argparse
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.evaluation.fiqa_qrels import (
    build_fiqa_eval_dataset,
    indexed_doc_ids_from_bm25,
    save_fiqa_eval_dataset,
)
from app.retrieval import Bm25ChunkIndex


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(
        description="Prepare FiQA qrels evaluation JSON (BEIR query + relevance labels)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="BEIR qrels split (use validation for development; test for final reporting only)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/benchmarks/fiqa_qrels_<split>.json)",
    )
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help="BM25 collection name used to filter qrels to indexed doc_ids",
    )
    parser.add_argument(
        "--no-filter-to-index",
        action="store_true",
        help="Keep all qrels queries (full corpus eval); default filters to BM25 doc_ids when index exists",
    )
    parser.add_argument(
        "--min-relevant-in-index",
        type=int,
        default=1,
        help="When filtering, drop queries with fewer than this many relevant docs in the index",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.join(
            _REPO_ROOT,
            "data",
            "benchmarks",
            f"fiqa_qrels_{args.split}.json",
        )

    indexed_doc_ids: set[str] | None = None
    bm25_path = settings.resolved_bm25_index_path(args.collection)
    if not args.no_filter_to_index and bm25_path.is_file():
        print(f"Loading indexed doc_ids from {bm25_path}...")
        indexed_doc_ids = indexed_doc_ids_from_bm25(Bm25ChunkIndex.load(bm25_path))
        print(f"Indexed documents: {len(indexed_doc_ids)}")
    elif not args.no_filter_to_index:
        print(
            f"No BM25 index at {bm25_path}; exporting full qrels split without index filter. "
            "Run ingest first to filter queries to your corpus."
        )

    print(f"Building eval dataset (qrels split={args.split})...")
    dataset = build_fiqa_eval_dataset(
        args.split,  # type: ignore[arg-type]
        indexed_doc_ids=indexed_doc_ids,
        min_relevant_in_index=args.min_relevant_in_index,
    )
    save_fiqa_eval_dataset(output, dataset)
    print(f"Wrote {dataset.num_queries} queries to {output}")


if __name__ == "__main__":
    main()
