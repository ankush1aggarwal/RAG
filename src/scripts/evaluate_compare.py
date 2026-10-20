"""Run dense / BM25 / hybrid qrels evaluation and write comparison report."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.evaluation.retrieval_eval import (
    evaluate_all_methods,
    load_eval_dataset,
    render_week4_comparison_report,
    save_eval_result,
)
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import embedding_model_for_method


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(
        description="Compare dense, BM25, and hybrid on FiQA qrels"
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(_REPO_ROOT, "data", "benchmarks", "fiqa_qrels_validation.json"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--qdrant-path", default=settings.qdrant_path)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(_REPO_ROOT, "data", "benchmarks"),
        help="Directory for per-method JSON results",
    )
    parser.add_argument(
        "--report",
        default=os.path.join(_REPO_ROOT, "docs", "reports", "week4_retrieval_eval.md"),
    )
    parser.add_argument("--limit-queries", type=int, default=None)
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
        results = evaluate_all_methods(
            dataset,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=args.top_k,
            eval_dataset_path=args.dataset,
        )
    finally:
        store.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, result in results.items():
        out = output_dir / f"fiqa_qrels_eval_{method}_{dataset.qrels_split}.json"
        save_eval_result(out, result)
        print(f"Wrote {out}")

    corpus_limit = settings.resolved_fiqa_corpus_limit()
    corpus_note = (
        "`BeIR/fiqa` full corpus (~57k docs)"
        if corpus_limit is None
        else f"`BeIR/fiqa` first `{corpus_limit}` documents"
    )
    report_md = render_week4_comparison_report(
        results,
        collection_name=args.collection,
        embedding_method=settings.embedding_method,
        embedding_model=embedding_model_for_method(
            settings.embedding_method,  # type: ignore[arg-type]
            sentence_transformer_model=settings.embedding_model,
            cohere_embed_model=settings.cohere_embed_model,
        ),
        corpus_note=corpus_note,
        week3_report_path=Path(_REPO_ROOT) / "docs" / "reports" / "week3_retrieval_comparison.md",
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nWrote comparison report to {report_path}")


if __name__ == "__main__":
    main()
