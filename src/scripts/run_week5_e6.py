"""
Week 5 — E6 chunk size sensitivity (dense retrieval, fixed 20% overlap).

Creates a **separate Qdrant + BM25 collection per chunk size** (default names:
``fiqa`` for 100-word baseline if present, else ``fiqa_cs100``; ``fiqa_cs200``;
``fiqa_cs300``).

Does **not** delete collections. Remove old indexes only after explicit confirmation.

Examples:
  python src/scripts/run_week5_e6.py --phase all --split validation
  python src/scripts/run_week5_e6.py --phase ingest --sizes 200 300
  python src/scripts/run_week5_e6.py --phase eval --split validation
  python src/scripts/run_week5_e6.py --phase eval --split test --sizes 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.evaluation.e6_chunk_size import (
    DEFAULT_CHUNK_SIZES,
    DEFAULT_OVERLAP_FRACTION,
    best_chunk_size_on_validation,
    render_e6_report,
    resolve_e6_collection,
)
from app.evaluation.retrieval_eval import (
    RetrievalEvalResult,
    evaluate_retrieval_dataset,
    load_eval_dataset,
    save_eval_result,
)
from app.ingestion.ingest_manifest import load_manifest, manifest_path_for_collection
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import embedding_model_for_method

BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_DIR = Path(_REPO_ROOT) / "docs" / "reports"
E6_REPORT = "week5_e6_chunk_size.md"
INGEST_SCRIPT = os.path.join(_REPO_ROOT, "src", "scripts", "ingest.py")
PREPARE_EVAL_SCRIPT = os.path.join(_REPO_ROOT, "src", "scripts", "prepare_fiqa_eval.py")


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


def _qdrant_path(settings, override: str | None) -> str | None:
    path = (override if override is not None else settings.qdrant_path).strip()
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    return path


def _collection_map(
    sizes: tuple[int, ...],
    *,
    baseline_collection: str | None,
    prefixed_only: bool,
) -> dict[int, str]:
    return {
        size: resolve_e6_collection(
            size,
            baseline_collection=baseline_collection,
            use_prefixed_collections_only=prefixed_only,
        )
        for size in sizes
    }


def _manifest_matches_chunk_size(
    settings, collection: str, chunk_size: int, overlap: float
) -> bool:
    manifest_path = manifest_path_for_collection(settings.resolved_bm25_index_path(collection))
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError):
        return False
    return manifest.chunk_size == chunk_size and abs(manifest.overlap_fraction - overlap) < 1e-9


def _needs_ingest(
    settings,
    collection: str,
    chunk_size: int,
    overlap: float,
    qdrant_path: str | None,
    force: bool,
) -> bool:
    if force:
        return True
    store = QdrantChunkStore.from_path(qdrant_path, collection)
    try:
        if not store.collection_exists() or store.point_count() == 0:
            return True
        if not _manifest_matches_chunk_size(settings, collection, chunk_size, overlap):
            print(
                f"  [{collection}] manifest/chunk_size mismatch — re-ingest required "
                f"(use --force-ingest)."
            )
            return True
        print(f"  [{collection}] exists ({store.point_count():,} points) — skipping ingest.")
        return False
    finally:
        store.close()


def _run_ingest(
    collection: str,
    chunk_size: int,
    overlap: float,
    qdrant_path: str | None,
    force: bool,
) -> None:
    cmd = [
        sys.executable,
        INGEST_SCRIPT,
        "--full-corpus",
        "--collection",
        collection,
        "--chunk-size",
        str(chunk_size),
        "--overlap",
        str(overlap),
    ]
    if qdrant_path:
        cmd.extend(["--qdrant-path", qdrant_path])
    if force:
        cmd.append("--recreate")
    else:
        cmd.append("--no-recreate")
    print(f"\n=== Ingest: {collection} (chunk_size={chunk_size}, overlap={overlap}) ===")
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def _prepare_eval(split: str, collection: str) -> Path:
    out = BENCHMARK_DIR / f"fiqa_qrels_{split}_{collection}.json"
    subprocess.run(
        [
            sys.executable,
            PREPARE_EVAL_SCRIPT,
            "--split",
            split,
            "--collection",
            collection,
            "--output",
            str(out),
        ],
        check=True,
        cwd=_REPO_ROOT,
    )
    return out


def _open_store(settings, collection: str, qdrant_path: str | None):
    bm25 = Bm25ChunkIndex.load(settings.resolved_bm25_index_path(collection))
    embedder = create_embedding_provider(
        settings.embedding_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )
    store = QdrantChunkStore.from_path(qdrant_path, collection)
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


def _run_eval_arm(
    settings,
    *,
    chunk_size: int,
    collection: str,
    split: str,
    qdrant_path: str | None,
    top_k: int,
    limit_queries: int | None,
    progress_every: int,
) -> tuple[RetrievalEvalResult, int]:
    eval_path = _prepare_eval(split, collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), limit_queries)
    store, embedder, bm25 = _open_store(settings, collection, qdrant_path)
    try:
        points = store.point_count()
        print(f"\n=== Eval: chunk_size={chunk_size} collection={collection} split={split} ===")
        result = evaluate_retrieval_dataset(
            dataset,
            method="dense",
            store=store,
            embedder=embedder,
            bm25_index=bm25,
            top_k=top_k,
            eval_dataset_path=str(eval_path),
            progress_every=progress_every,
            use_rerank=False,
            include_rag_responses=False,
            llm=None,
        )
    finally:
        store.close()
    artifact = BENCHMARK_DIR / f"week5_e6_cs{chunk_size}_{split}.json"
    save_eval_result(artifact, result)
    print(f"Wrote {artifact}")
    return result, points


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Week 5 E6 — chunk size sweep")
    parser.add_argument(
        "--phase",
        choices=("ingest", "eval", "all"),
        default="all",
        help="ingest only, eval only (existing collections), or both",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNK_SIZES),
        help="Chunk sizes in words (default: 100 200 300)",
    )
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRACTION)
    parser.add_argument("--qdrant-path", default=None, help="Override settings.qdrant_path")
    parser.add_argument(
        "--baseline-collection",
        default="fiqa",
        help="Reuse this collection for chunk_size=100 (Week 4 index). "
        "Use '' to always use fiqa_cs100.",
    )
    parser.add_argument(
        "--prefixed-collections-only",
        action="store_true",
        help="Use fiqa_cs{N} for every size including 100 (ingest fiqa_cs100).",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Always recreate collections during ingest phase.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--run-test-for-winner",
        action="store_true",
        help="After validation eval on all sizes, run test once for best NDCG@10.",
    )
    args = parser.parse_args()

    sizes = tuple(sorted(set(args.sizes)))
    baseline = args.baseline_collection.strip() or None
    collections = _collection_map(
        sizes,
        baseline_collection=baseline,
        prefixed_only=args.prefixed_collections_only,
    )
    qdrant_path = _qdrant_path(settings, args.qdrant_path)

    if args.phase in ("ingest", "all"):
        for size in sizes:
            collection = collections[size]
            skip_baseline_reingest = (
                size == 100
                and baseline
                and collection == baseline
                and not args.prefixed_collections_only
            )
            if skip_baseline_reingest:
                if _needs_ingest(
                    settings, collection, size, args.overlap, qdrant_path, args.force_ingest
                ):
                    _run_ingest(collection, size, args.overlap, qdrant_path, args.force_ingest)
            else:
                if _needs_ingest(
                    settings, collection, size, args.overlap, qdrant_path, args.force_ingest
                ):
                    _run_ingest(collection, size, args.overlap, qdrant_path, True)

    results: dict[int, object] = {}
    point_counts: dict[int, int] = {}

    if args.phase in ("eval", "all"):
        for size in sizes:
            collection = collections[size]
            result, points = _run_eval_arm(
                settings,
                chunk_size=size,
                collection=collection,
                split=args.split,
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )
            results[size] = result
            point_counts[size] = points

        winner: int | None = None
        if args.split == "validation" and len(results) > 1:
            winner = best_chunk_size_on_validation(results)  # type: ignore[arg-type]

        report = render_e6_report(
            results,  # type: ignore[arg-type]
            sizes,
            qrels_split=args.split,
            overlap_fraction=args.overlap,
            collections=collections,
            point_counts=point_counts,
            embedding_model=_embedding_model_label(settings),
            corpus_note=_corpus_note(settings),
            winner_size=winner,
        )
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / E6_REPORT
        report_path.write_text(report, encoding="utf-8")
        print(f"Wrote {report_path}")

        meta = {
            "collections": {str(k): v for k, v in collections.items()},
            "chunk_sizes": list(sizes),
            "overlap_fraction": args.overlap,
            "split": args.split,
            "validation_winner_chunk_size": winner,
        }
        (BENCHMARK_DIR / "week5_e6_protocol.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if args.run_test_for_winner and winner is not None and args.split == "validation":
            print(f"\n=== Test eval for validation winner: chunk_size={winner} ===")
            _run_eval_arm(
                settings,
                chunk_size=winner,
                collection=collections[winner],
                split="test",
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )


if __name__ == "__main__":
    main()
