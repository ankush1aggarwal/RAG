"""
Week 5 — E3 candidate diversity (dense + cross-encoder rerank).

Prerequisites:
  - Full (or matching) Qdrant collection + BM25 index for ``--collection`` (default ``fiqa``)
  - ``data/benchmarks/fiqa_qrels_{validation,test}.json`` (from prepare_fiqa_eval or Week 4)

Example:
  python src/scripts/run_week5_e3.py --split validation
  python src/scripts/run_week5_e3.py --split test --arm overfetch_cap1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.evaluation.e3_experiments import default_arms, render_e3_report
from app.evaluation.retrieval_eval import (
    evaluate_retrieval_dataset,
    load_eval_dataset,
    save_eval_result,
)
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.candidate_diversity import E3_ARMS, e3_arm_by_id
from app.retrieval.embeddings import embedding_model_for_method
from app.retrieval.reranker import create_reranker_provider

DEFAULT_COLLECTION = "fiqa"
BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_DIR = Path(_REPO_ROOT) / "docs" / "reports"
E3_REPORT = "week5_e3_candidate_diversity.md"


def _corpus_note(settings) -> str:
    limit = settings.resolved_fiqa_corpus_limit()
    if limit is None:
        return "`BeIR/fiqa` **full corpus** (~57k documents)"
    return f"`BeIR/fiqa` first `{limit}` documents (set FIQA_CORPUS_LIMIT=0 for full corpus)"


def _embedding_model_label(settings) -> str:
    return embedding_model_for_method(
        settings.embedding_method,  # type: ignore[arg-type]
        sentence_transformer_model=settings.embedding_model,
        cohere_embed_model=settings.cohere_embed_model,
    )


def _eval_path(split: str) -> Path:
    return BENCHMARK_DIR / f"fiqa_qrels_{split}.json"


def _prepare_eval(split: str, collection: str) -> None:
    subprocess.run(
        [
            sys.executable,
            os.path.join(_REPO_ROOT, "src", "scripts", "prepare_fiqa_eval.py"),
            "--split",
            split,
            "--collection",
            collection,
        ],
        check=True,
        cwd=_REPO_ROOT,
    )


def _open_store(settings, collection: str, qdrant_path: str | None):
    path = qdrant_path
    if path is not None and not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    bm25 = Bm25ChunkIndex.load(settings.resolved_bm25_index_path(collection))
    embedder = create_embedding_provider(
        settings.embedding_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )
    store = QdrantChunkStore.from_path(path, collection)
    return store, embedder, bm25


def _maybe_limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    limited = dataset.queries[:limit]
    return type(dataset)(
        version=dataset.version,
        corpus_id=dataset.corpus_id,
        qrels_split=dataset.qrels_split,
        queries=tuple(limited),
    )


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Week 5 E3 — candidate diversity arms")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-path", default=settings.qdrant_path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--overfetch-limit", type=int, default=150)
    parser.add_argument("--rerank-budget", type=int, default=50)
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help=f"Run subset of arms (default: all). Choices: {', '.join(a.arm_id for a in E3_ARMS)}",
    )
    parser.add_argument("--prepare", action="store_true", help="Regenerate qrels JSON before eval")
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    eval_path = _eval_path(args.split)
    if args.prepare:
        _prepare_eval(args.split, args.collection)
    if not eval_path.is_file():
        raise SystemExit(f"Missing {eval_path}; run prepare_fiqa_eval or Week 4 exp2 first.")

    if args.arms:
        arms = tuple(e3_arm_by_id(arm_id) for arm_id in args.arms)
    else:
        arms = default_arms()

    arms = tuple(
        replace(
            arm,
            rerank_budget=args.rerank_budget,
            overfetch_limit=args.overfetch_limit,
        )
        for arm in arms
    )

    reranker = create_reranker_provider(
        "cross_encoder",
        cross_encoder_model=settings.cross_encoder_rerank_model,
    )
    if reranker is None:
        raise SystemExit("cross_encoder reranker could not be created")

    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), args.limit_queries)
    store, embedder, bm25 = _open_store(settings, args.collection, args.qdrant_path)
    results: dict[str, object] = {}
    try:
        for arm in arms:
            print(f"\n=== E3 arm: {arm.arm_id} ({arm.label}) ===")
            result = evaluate_retrieval_dataset(
                dataset,
                method="dense",
                store=store,
                embedder=embedder,
                bm25_index=bm25,
                top_k=args.top_k,
                eval_dataset_path=str(eval_path),
                progress_every=args.progress_every,
                use_rerank=True,
                reranker=reranker,
                rerank_candidate_limit=args.rerank_budget,
                include_rag_responses=False,
                llm=None,
                e3_arm=arm,
            )
            out_path = BENCHMARK_DIR / f"week5_e3_{arm.arm_id}_{args.split}.json"
            save_eval_result(out_path, result)
            print(f"Wrote {out_path}")
            results[arm.arm_id] = result
    finally:
        store.close()

    report = render_e3_report(
        results,  # type: ignore[arg-type]
        arms,
        collection_name=args.collection,
        qrels_split=args.split,
        embedding_model=_embedding_model_label(settings),
        corpus_note=_corpus_note(settings),
        reranker_name=reranker.name,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / E3_REPORT
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
