"""
Week 5 — E8 embedding model comparison (200w / 20% overlap).

Baseline: ``all-MiniLM-L6-v2`` on ``fiqa_cs200`` (reuse if present).
Treatment: ingest alternative model to its own collection (default BGE base).

Example:
  python src/scripts/run_week5_e8.py --phase all --split validation --run-test-for-winner
  python src/scripts/run_week5_e8.py --alternative-model BAAI/bge-base-en-v1.5 --phase ingest
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
from app.evaluation.e7_chunk_overlap import DEFAULT_E7_CHUNK_SIZE
from app.evaluation.e8_embedding_model import (
    BASELINE_COLLECTION,
    BASELINE_EMBEDDING_MODEL,
    DEFAULT_ALTERNATIVE_MODEL,
    DEFAULT_OVERLAP_FRACTION,
    best_model_on_validation,
    e8_artifact_tag,
    e8_collection_for_model,
    render_e8_report,
)
from app.evaluation.retrieval_eval import (
    evaluate_retrieval_dataset,
    load_eval_dataset,
    save_eval_result,
)
from app.ingestion.ingest_manifest import load_manifest, manifest_path_for_collection
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import embedding_dimension

BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_DIR = Path(_REPO_ROOT) / "docs" / "reports"
E8_REPORT = "week5_e8_embedding_model.md"
INGEST_SCRIPT = os.path.join(_REPO_ROOT, "src", "scripts", "ingest.py")
PREPARE_EVAL_SCRIPT = os.path.join(_REPO_ROOT, "src", "scripts", "prepare_fiqa_eval.py")


def _corpus_note(settings) -> str:
    limit = settings.resolved_fiqa_corpus_limit()
    if limit is None:
        return "`BeIR/fiqa` **full corpus** (~57k documents)"
    return f"`BeIR/fiqa` first `{limit}` documents (set FIQA_CORPUS_LIMIT=0 for full corpus)"


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
    *,
    chunk_size: int,
    overlap: float,
    embedding_model: str,
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
        and manifest.embedding_model == embedding_model
    )


def _needs_ingest(
    settings,
    collection: str,
    *,
    chunk_size: int,
    overlap: float,
    embedding_model: str,
    qdrant_path: str | None,
    force: bool,
) -> bool:
    if force:
        return True
    store = QdrantChunkStore.from_path(qdrant_path, collection)
    try:
        if not store.collection_exists() or store.point_count() == 0:
            return True
        if not _manifest_matches(
            settings,
            collection,
            chunk_size=chunk_size,
            overlap=overlap,
            embedding_model=embedding_model,
        ):
            print(f"  [{collection}] manifest mismatch — re-ingest required.")
            return True
        print(f"  [{collection}] exists ({store.point_count():,} points) — skipping ingest.")
        return False
    finally:
        store.close()


def _run_ingest(
    collection: str,
    *,
    chunk_size: int,
    overlap: float,
    embedding_model: str,
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
        "--method",
        "sentence_transformer",
        "--embedding-model",
        embedding_model,
    ]
    if qdrant_path:
        cmd.extend(["--qdrant-path", qdrant_path])
    cmd.append("--recreate" if recreate else "--no-recreate")
    print(f"\n=== Ingest: {collection} model={embedding_model} ===")
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


def _open_store(settings, collection: str, embedding_model: str, qdrant_path: str | None):
    bm25 = Bm25ChunkIndex.load(settings.resolved_bm25_index_path(collection))
    embedder = create_embedding_provider(
        "sentence_transformer",
        model_name=embedding_model,
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


def _manifest_dim(settings, collection: str) -> int:
    path = manifest_path_for_collection(settings.resolved_bm25_index_path(collection))
    if path.is_file():
        return load_manifest(path).embedding_dimension
    return 0


def _run_eval_arm(
    settings,
    *,
    embedding_model: str,
    collection: str,
    split: str,
    qdrant_path: str | None,
    top_k: int,
    limit_queries: int | None,
    progress_every: int,
):
    eval_path = _prepare_eval(split, collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), limit_queries)
    store, embedder, bm25 = _open_store(settings, collection, embedding_model, qdrant_path)
    try:
        store.assert_collection_vector_size(embedding_dimension(embedder))
        points = store.point_count()
        print(f"\n=== Eval: model={embedding_model} collection={collection} split={split} ===")
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
    tag = e8_artifact_tag(embedding_model)
    artifact = BENCHMARK_DIR / f"week5_e8_{tag}_{split}.json"
    save_eval_result(artifact, result)
    print(f"Wrote {artifact}")
    return result, points


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Week 5 E8 — embedding model comparison")
    parser.add_argument("--phase", choices=("ingest", "eval", "all"), default="all")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--baseline-model", default=BASELINE_EMBEDDING_MODEL)
    parser.add_argument("--alternative-model", default=DEFAULT_ALTERNATIVE_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_E7_CHUNK_SIZE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRACTION)
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--run-test-for-winner", action="store_true")
    args = parser.parse_args()

    models = (args.baseline_model, args.alternative_model)
    collections = {
        m: e8_collection_for_model(m, chunk_size=args.chunk_size) for m in models
    }
    qdrant_path = _qdrant_path(settings, args.qdrant_path)

    if args.phase in ("ingest", "all"):
        for model in models:
            collection = collections[model]
            reuse_baseline = model == args.baseline_model and collection == BASELINE_COLLECTION
            needs = _needs_ingest(
                settings,
                collection,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                embedding_model=model,
                qdrant_path=qdrant_path,
                force=args.force_ingest,
            )
            if not needs:
                continue
            recreate = not reuse_baseline or args.force_ingest
            _run_ingest(
                collection,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                embedding_model=model,
                qdrant_path=qdrant_path,
                recreate=recreate,
            )

    results: dict[str, object] = {}
    point_counts: dict[str, int] = {}
    dimensions: dict[str, int] = {}

    if args.phase in ("eval", "all"):
        for model in models:
            collection = collections[model]
            dimensions[model] = _manifest_dim(settings, collection)
            result, points = _run_eval_arm(
                settings,
                embedding_model=model,
                collection=collection,
                split=args.split,
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )
            results[model] = result
            point_counts[model] = points

        winner: str | None = None
        if args.split == "validation" and len(results) > 1:
            winner = best_model_on_validation(results)  # type: ignore[arg-type]

        report = render_e8_report(
            results,  # type: ignore[arg-type]
            models,
            collections=collections,
            dimensions=dimensions,
            point_counts=point_counts,
            chunk_size=args.chunk_size,
            overlap_fraction=args.overlap,
            qrels_split=args.split,
            corpus_note=_corpus_note(settings),
            winner_model=winner,
        )
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / E8_REPORT).write_text(report, encoding="utf-8")
        print(f"Wrote {REPORT_DIR / E8_REPORT}")

        meta = {
            "chunk_size": args.chunk_size,
            "overlap_fraction": args.overlap,
            "models": list(models),
            "collections": collections,
            "split": args.split,
            "validation_winner_model": winner,
        }
        (BENCHMARK_DIR / "week5_e8_protocol.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if args.run_test_for_winner and winner is not None and args.split == "validation":
            print(f"\n=== Test eval for validation winner: {winner} ===")
            _run_eval_arm(
                settings,
                embedding_model=winner,
                collection=collections[winner],
                split="test",
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )


if __name__ == "__main__":
    main()
