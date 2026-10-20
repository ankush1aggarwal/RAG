"""Offline retrieval evaluation with BEIR FiQA qrels (NDCG, MRR, MAP, P/R @k)."""

import argparse
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.evaluation.retrieval_eval import (
    evaluate_retrieval_dataset,
    format_metrics_table,
    load_eval_dataset,
    save_eval_result,
)
from app.evaluation.retrieval_metrics import DEFAULT_K_VALUES
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Evaluate retrieval on FiQA qrels JSON")
    parser.add_argument(
        "--dataset",
        default=os.path.join(_REPO_ROOT, "data", "benchmarks", "fiqa_qrels_validation.json"),
        help="Prepared eval JSON from prepare_fiqa_eval.py",
    )
    parser.add_argument(
        "--method",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieval depth (at least 10 for @10 metrics)",
    )
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--qdrant-path", default=settings.qdrant_path)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for full per-query results",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries (smoke test)",
    )
    args = parser.parse_args()

    dataset = load_eval_dataset(args.dataset)
    if args.limit_queries is not None:
        limited = dataset.queries[: args.limit_queries]
        dataset = type(dataset)(
            version=dataset.version,
            corpus_id=dataset.corpus_id,
            qrels_split=dataset.qrels_split,
            queries=tuple(limited),
        )

    qdrant_path = args.qdrant_path.strip() or None
    if qdrant_path is not None and not os.path.isabs(qdrant_path):
        qdrant_path = os.path.join(_REPO_ROOT, qdrant_path)

    bm25_index = None
    if args.method in ("bm25", "hybrid"):
        bm25_path = settings.resolved_bm25_index_path(args.collection)
        bm25_index = Bm25ChunkIndex.load(bm25_path)

    embedder = create_embedding_provider(
        settings.embedding_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )

    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    try:
        print(
            f"Evaluating method={args.method} on {dataset.num_queries} queries "
            f"(qrels split={dataset.qrels_split}, top_k>={max(DEFAULT_K_VALUES)})..."
        )
        result = evaluate_retrieval_dataset(
            dataset,
            method=args.method,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=args.top_k,
            eval_dataset_path=args.dataset,
        )
    finally:
        store.close()

    print()
    print(format_metrics_table(result.aggregates, result.k_values))

    if args.output:
        save_eval_result(args.output, result)
        print(f"\nWrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
