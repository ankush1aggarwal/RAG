"""
Week 4 experiment runner.

Experiment 1 (manual): 1k-doc slice vs Week 3 baselines — see week4_retrieval_eval.md.

Experiment 2: Full-corpus index + validation qrels — dense / BM25 / hybrid.
Experiment 3: Tune RRF weights on validation; report weighted hybrid.
Experiment 4: Test qrels — dense / BM25 / weighted hybrid (frozen weights, no rerank).
Experiment 5: Same as experiment 4 + local cross-encoder rerank.

Prerequisites for 2–5:
  python src/scripts/ingest.py --full-corpus --collection fiqa --recreate
  python src/scripts/prepare_fiqa_eval.py --split validation --collection fiqa
  python src/scripts/prepare_fiqa_eval.py --split test --collection fiqa
"""

import argparseimport jsonimport osimport subprocessimport sysfrom pathlib import Pathfrom dotenv import load_dotenv_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settingsfrom app.evaluation.eval_llm import CitationPreservingEvalLLMfrom app.evaluation.retrieval_eval import (    RetrievalEvalResult,    evaluate_all_methods,    load_eval_dataset,    save_eval_result,)from app.evaluation.week4_experiments import (    EXPERIMENT2_REPORT,    EXPERIMENT3_REPORT,    EXPERIMENT4_REPORT,    EXPERIMENT5_REPORT,    MASTER_REPORT,    PROTOCOL_ARTIFACT,    TEST_METHODS,    render_experiment2_report,    render_experiment3_report,    render_experiment4_report,    render_experiment5_report,    render_master_program_report,    standard_eval_methods,)from app.evaluation.weight_tuning import prefetch_hybrid_list_cache, tune_fusion_weightsfrom app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_providerfrom app.retrieval.embeddings import embedding_model_for_methodfrom app.retrieval.fusion_weights import load_fusion_weights, save_fusion_weightsfrom app.retrieval.reranker import create_reranker_providerDEFAULT_COLLECTION = "fiqa"
WEIGHTS_PATH = Path(_REPO_ROOT) / "data" / "benchmarks" / "week4_fusion_weights.json"
BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_DIR = Path(_REPO_ROOT) / "docs" / "reports"

EVAL_LLM = CitationPreservingEvalLLM()


def _corpus_note(settings) -> str:
    limit = settings.resolved_fiqa_corpus_limit()
    if limit is None:
        return "`BeIR/fiqa` **full corpus** (~57k documents)"
    return f"`BeIR/fiqa` first `{limit}` documents (not full corpus — set FIQA_CORPUS_LIMIT=0)"


def _embedding_model_label(settings) -> str:
    return embedding_model_for_method(
        settings.embedding_method,  # type: ignore[arg-type]
        sentence_transformer_model=settings.embedding_model,
        cohere_embed_model=settings.cohere_embed_model,
    )


def _report_kwargs(args: argparse.Namespace, settings) -> dict:
    return {
        "collection_name": args.collection,
        "embedding_method": settings.embedding_method,
        "embedding_model": _embedding_model_label(settings),
        "corpus_note": _corpus_note(settings),
        "example_count": args.example_count,
    }


def _eval_paths(split: str) -> Path:
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


def _common_eval_kwargs(args: argparse.Namespace, eval_path: Path) -> dict:
    return {
        "top_k": args.top_k,
        "eval_dataset_path": str(eval_path),
        "progress_every": args.progress_every,
        "generation_top_k": args.generation_top_k,
        "llm": EVAL_LLM,
    }


def _persist_protocol(result: RetrievalEvalResult) -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_DIR / PROTOCOL_ARTIFACT).write_text(
        json.dumps(result.protocol.to_dict(), indent=2),
        encoding="utf-8",
    )


def _write_master_report(args: argparse.Namespace, settings) -> None:
    report = render_master_program_report(
        benchmark_dir=BENCHMARK_DIR,
        report_dir=REPORT_DIR,
        collection_name=args.collection,
        embedding_method=settings.embedding_method,
        embedding_model=_embedding_model_label(settings),
        corpus_note=_corpus_note(settings),
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / MASTER_REPORT).write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / MASTER_REPORT}")


def run_experiment2(args: argparse.Namespace, settings) -> None:
    eval_path = _eval_paths("validation")
    if args.prepare:
        _prepare_eval("validation", args.collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), args.limit_queries)
    store, embedder, bm25 = _open_store(settings, args.collection, args.qdrant_path)
    try:
        results = evaluate_all_methods(
            dataset,
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            methods=standard_eval_methods(include_weighted=False),
            **_common_eval_kwargs(args, eval_path),
        )
    finally:
        store.close()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    for method, result in results.items():
        save_eval_result(BENCHMARK_DIR / f"week4_exp2_{method}_validation.json", result)
    _persist_protocol(next(iter(results.values())))

    report = render_experiment2_report(results, **_report_kwargs(args, settings))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / EXPERIMENT2_REPORT).write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / EXPERIMENT2_REPORT}")


def run_experiment3(args: argparse.Namespace, settings) -> None:
    eval_path = _eval_paths("validation")
    if args.prepare:
        _prepare_eval("validation", args.collection)
    if not eval_path.is_file():
        raise SystemExit(f"Missing {eval_path}; run experiment 2 or --prepare first.")

    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), args.limit_queries)
    store, embedder, bm25 = _open_store(settings, args.collection, args.qdrant_path)
    try:
        print("Prefetching dense/BM25 lists for weight tuning...")
        cache = prefetch_hybrid_list_cache(
            dataset,
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            top_k=args.top_k,
            progress_every=args.progress_every,
        )
        print(f"Tuning weights (metric={args.tune_metric}, grid_steps={args.grid_steps})...")
        weights = tune_fusion_weights(
            dataset,
            cache,
            top_k=max(args.top_k, 10),
            metric_name=args.tune_metric,
            grid_steps=args.grid_steps,
        )
        save_fusion_weights(WEIGHTS_PATH, weights)
        print(f"Saved weights to {WEIGHTS_PATH}")

        fusion = weights.normalized()
        results = evaluate_all_methods(
            dataset,
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            methods=("hybrid", "weighted_hybrid"),
            fusion_weights=fusion,
            **_common_eval_kwargs(args, eval_path),
        )
    finally:
        store.close()

    for method, result in results.items():
        save_eval_result(BENCHMARK_DIR / f"week4_exp3_{method}_validation.json", result)
    _persist_protocol(next(iter(results.values())))

    report = render_experiment3_report(
        results,
        weights,
        **_report_kwargs(args, settings),
    )
    (REPORT_DIR / EXPERIMENT3_REPORT).write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / EXPERIMENT3_REPORT}")


def run_experiment4(args: argparse.Namespace, settings) -> None:
    if not WEIGHTS_PATH.is_file():
        raise SystemExit(f"Missing {WEIGHTS_PATH}; run experiment 3 first.")
    weights = load_fusion_weights(WEIGHTS_PATH)
    fusion = weights.normalized()

    eval_path = _eval_paths("test")
    if args.prepare:
        _prepare_eval("test", args.collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), args.limit_queries)

    store, embedder, bm25 = _open_store(settings, args.collection, args.qdrant_path)
    try:
        results = evaluate_all_methods(
            dataset,
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            methods=TEST_METHODS,
            fusion_weights=fusion,
            **_common_eval_kwargs(args, eval_path),
        )
    finally:
        store.close()

    for method, result in results.items():
        save_eval_result(BENCHMARK_DIR / f"week4_exp4_{method}_test.json", result)
    _persist_protocol(next(iter(results.values())))

    report = render_experiment4_report(
        results,
        weights,
        **_report_kwargs(args, settings),
    )
    (REPORT_DIR / EXPERIMENT4_REPORT).write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / EXPERIMENT4_REPORT}")


def run_experiment5(args: argparse.Namespace, settings) -> None:
    if not WEIGHTS_PATH.is_file():
        raise SystemExit(f"Missing {WEIGHTS_PATH}; run experiment 3 first.")
    weights = load_fusion_weights(WEIGHTS_PATH)
    fusion = weights.normalized()

    reranker = create_reranker_provider(
        "cross_encoder",
        cross_encoder_model=settings.cross_encoder_rerank_model,
    )
    if reranker is None:
        raise SystemExit("cross_encoder reranker could not be created")

    eval_path = _eval_paths("test")
    if args.prepare:
        _prepare_eval("test", args.collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), args.limit_queries)

    store, embedder, bm25 = _open_store(settings, args.collection, args.qdrant_path)
    try:
        results = evaluate_all_methods(
            dataset,
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            methods=TEST_METHODS,
            fusion_weights=fusion,
            use_rerank=True,
            reranker=reranker,
            rerank_candidate_limit=args.rerank_candidate_limit,
            **_common_eval_kwargs(args, eval_path),
        )
    finally:
        store.close()

    for method, result in results.items():
        save_eval_result(BENCHMARK_DIR / f"week4_exp5_{method}_test.json", result)
    _persist_protocol(next(iter(results.values())))

    report = render_experiment5_report(
        results,
        weights,
        reranker_name=reranker.name,
        **_report_kwargs(args, settings),
    )
    (REPORT_DIR / EXPERIMENT5_REPORT).write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / EXPERIMENT5_REPORT}")


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Week 4 retrieval experiments 2–5")
    parser.add_argument(
        "experiment",
        choices=["exp2", "exp3", "exp4", "exp5", "all"],
        help="exp4=test baseline; exp5=test + rerank; all=exp2→exp5 + master report",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-path", default=settings.qdrant_path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--generation-top-k",
        type=int,
        default=None,
        help="Chunks passed to stub LLM for citations (default: top-k)",
    )
    parser.add_argument("--prepare", action="store_true", help="Regenerate qrels JSON before eval")
    parser.add_argument("--tune-metric", default="ndcg@10")
    parser.add_argument("--grid-steps", type=int, default=21)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--example-count", type=int, default=3)
    parser.add_argument(
        "--rerank-candidate-limit",
        type=int,
        default=settings.rerank_candidate_limit,
    )
    args = parser.parse_args()

    if args.collection == "fiqa_sample" and settings.resolved_fiqa_corpus_limit() is None:
        print(
            "Warning: full corpus ingest usually uses COLLECTION_NAME=fiqa, not fiqa_sample.",
            file=sys.stderr,
        )

    runners = {
        "exp2": run_experiment2,
        "exp3": run_experiment3,
        "exp4": run_experiment4,
        "exp5": run_experiment5,
    }
    if args.experiment == "all":
        for name in ("exp2", "exp3", "exp4", "exp5"):
            runners[name](args, settings)
        _write_master_report(args, settings)
    else:
        runners[args.experiment](args, settings)
        if args.experiment == "exp5":
            _write_master_report(args, settings)


if __name__ == "__main__":
    main()
