"""
Week 5 — E7 chunk overlap sensitivity at fixed chunk size (E6 winner: 200 words).

Collections:
  - 20% overlap → reuse ``fiqa_cs200`` (E6)
  - 10% → ``fiqa_cs200_ov10``
  - 30% → ``fiqa_cs200_ov30``

Example:
  python src/scripts/run_week5_e7.py --phase all --split validation --run-test-for-winner
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
from app.evaluation.e7_chunk_overlap import (
    DEFAULT_E7_CHUNK_SIZE,
    DEFAULT_OVERLAP_FRACTIONS,
    best_overlap_on_validation,
    e7_collection_name,
    overlap_artifact_tag,
    render_e7_report,
)
from app.evaluation.retrieval_eval import (
    evaluate_retrieval_dataset,
    load_eval_dataset,
    save_eval_result,
)
from app.ingestion.ingest_manifest import (
    invalidate_published_ingest_artifacts,
    load_manifest,
    manifest_path_for_collection,
)
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import embedding_model_for_method

BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_DIR = Path(_REPO_ROOT) / "docs" / "reports"
E7_REPORT = "week5_e7_chunk_overlap.md"
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


def _manifest_matches(
    settings,
    collection: str,
    chunk_size: int,
    overlap: float,
) -> bool:
    manifest_path = manifest_path_for_collection(settings.resolved_bm25_index_path(collection))
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError):
        return False
    return (
        manifest.chunk_size == chunk_size
        and abs(manifest.overlap_fraction - overlap) < 1e-9
    )


def delete_collection_bundle(
    settings,
    collection: str,
    qdrant_path: str | None,
) -> None:
    store = QdrantChunkStore.from_path(qdrant_path, collection)
    try:
        if store.drop_collection():
            print(f"Deleted Qdrant collection '{collection}'.")
        else:
            print(f"Qdrant collection '{collection}' was not present.")
    finally:
        store.close()
    bm25_path = settings.resolved_bm25_index_path(collection)
    invalidate_published_ingest_artifacts(bm25_path)
    if not bm25_path.exists():
        print(f"Removed BM25 artifacts for '{collection}'.")


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
        if not _manifest_matches(settings, collection, chunk_size, overlap):
            print(f"  [{collection}] manifest mismatch — re-ingest required.")
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
    recreate: bool,
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
    cmd.append("--recreate" if recreate else "--no-recreate")
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
    overlap: float,
    chunk_size: int,
    collection: str,
    split: str,
    qdrant_path: str | None,
    top_k: int,
    limit_queries: int | None,
    progress_every: int,
):
    eval_path = _prepare_eval(split, collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), limit_queries)
    store, embedder, bm25 = _open_store(settings, collection, qdrant_path)
    try:
        points = store.point_count()
        print(
            f"\n=== Eval: overlap={overlap} collection={collection} "
            f"chunk_size={chunk_size} split={split} ==="
        )
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
    tag = overlap_artifact_tag(overlap)
    artifact = BENCHMARK_DIR / f"week5_e7_{tag}_{split}.json"
    save_eval_result(artifact, result)
    print(f"Wrote {artifact}")
    return result, points


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Week 5 E7 — chunk overlap sweep")
    parser.add_argument("--phase", choices=("ingest", "eval", "all"), default="all")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--overlaps",
        type=float,
        nargs="+",
        default=list(DEFAULT_OVERLAP_FRACTIONS),
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_E7_CHUNK_SIZE)
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument(
        "--skip-delete-cs300",
        action="store_true",
        help="Do not remove fiqa_cs300 (default: delete before E7 ingest/eval).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--run-test-for-winner", action="store_true")
    args = parser.parse_args()

    overlaps = tuple(sorted(set(args.overlaps)))
    collections = {ov: e7_collection_name(ov) for ov in overlaps}
    qdrant_path = _qdrant_path(settings, args.qdrant_path)

    if not args.skip_delete_cs300:
        print("\n=== Removing index bundle: fiqa_cs300 (per experiment cleanup) ===")
        delete_collection_bundle(settings, "fiqa_cs300", qdrant_path)

    if args.phase in ("ingest", "all"):
        for overlap in overlaps:
            collection = collections[overlap]
            reuse_baseline = collection == "fiqa_cs200" and abs(overlap - 0.2) < 1e-9
            if reuse_baseline:
                if _needs_ingest(
                    settings,
                    collection,
                    args.chunk_size,
                    overlap,
                    qdrant_path,
                    args.force_ingest,
                ):
                    _run_ingest(
                        collection,
                        args.chunk_size,
                        overlap,
                        qdrant_path,
                        args.force_ingest,
                    )
            elif _needs_ingest(
                settings,
                collection,
                args.chunk_size,
                overlap,
                qdrant_path,
                args.force_ingest,
            ):
                _run_ingest(collection, args.chunk_size, overlap, qdrant_path, True)

    results: dict[float, object] = {}
    point_counts: dict[float, int] = {}

    if args.phase in ("eval", "all"):
        for overlap in overlaps:
            result, points = _run_eval_arm(
                settings,
                overlap=overlap,
                chunk_size=args.chunk_size,
                collection=collections[overlap],
                split=args.split,
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )
            results[overlap] = result
            point_counts[overlap] = points

        winner: float | None = None
        if args.split == "validation" and len(results) > 1:
            winner = best_overlap_on_validation(results)  # type: ignore[arg-type]

        report = render_e7_report(
            results,  # type: ignore[arg-type]
            overlaps,
            chunk_size=args.chunk_size,
            qrels_split=args.split,
            collections=collections,
            point_counts=point_counts,
            embedding_model=_embedding_model_label(settings),
            corpus_note=_corpus_note(settings),
            winner_overlap=winner,
        )
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / E7_REPORT).write_text(report, encoding="utf-8")
        print(f"Wrote {REPORT_DIR / E7_REPORT}")

        meta = {
            "chunk_size": args.chunk_size,
            "overlaps": list(overlaps),
            "collections": {str(k): v for k, v in collections.items()},
            "split": args.split,
            "validation_winner_overlap_fraction": winner,
        }
        (BENCHMARK_DIR / "week5_e7_protocol.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if args.run_test_for_winner and winner is not None and args.split == "validation":
            print(f"\n=== Test eval for validation winner: overlap={winner} ===")
            _run_eval_arm(
                settings,
                overlap=winner,
                chunk_size=args.chunk_size,
                collection=collections[winner],
                split="test",
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )


if __name__ == "__main__":
    main()
