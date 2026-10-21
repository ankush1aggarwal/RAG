"""
Week 5 — E9 paragraph-aware chunking vs E8 fixed 200w (BGE).

Baseline: ``fiqa_cs200_bge_base_en_v1_5`` (reuse).
Treatment: ingest ``fiqa_para_bge_base_en_v1_5`` with --chunking-strategy paragraph.

Example:
  python src/scripts/run_week5_e9.py --phase probe
  python src/scripts/run_week5_e9.py --phase cleanup --execute
  python src/scripts/run_week5_e9.py --phase all --split validation --run-test-for-winner
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
from app.evaluation.e9_paragraph_chunking import (
    BASELINE_ARM,
    BASELINE_COLLECTION,
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_FRACTION,
    EMBEDDING_MODEL,
    TREATMENT_ARM,
    TREATMENT_COLLECTION,
    best_arm_on_validation,
    collection_for_arm,
    e9_artifact_tag,
    render_e9_report,
)
from app.evaluation.retrieval_eval import (
    evaluate_retrieval_dataset,
    load_eval_dataset,
    save_eval_result,
)
from app.ingestion import chunks_from_documents, documents_from_corpus, load_fiqa_corpus
from app.ingestion.ingest_manifest import (
    invalidate_published_ingest_artifacts,
    load_manifest,
    manifest_path_for_collection,
)
from app.retrieval import Bm25ChunkIndex, QdrantChunkStore, create_embedding_provider
from app.retrieval.embeddings import embedding_dimension

BENCHMARK_DIR = Path(_REPO_ROOT) / "data" / "benchmarks"
REPORT_DIR = Path(_REPO_ROOT) / "docs" / "reports"
E9_REPORT = "week5_e9_paragraph_chunking.md"
INGEST_SCRIPT = os.path.join(_REPO_ROOT, "src", "scripts", "ingest.py")
PREPARE_EVAL_SCRIPT = os.path.join(_REPO_ROOT, "src", "scripts", "prepare_fiqa_eval.py")

KEEP_COLLECTIONS = frozenset({BASELINE_COLLECTION, "fiqa"})


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


def _list_qdrant_collections(qdrant_path: str | None) -> list[str]:
    if not qdrant_path:
        return []
    root = Path(qdrant_path) / "collection"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _list_bm25_collections(settings) -> list[str]:
    directory = settings.resolved_bm25_index_path("_probe").parent
    if not directory.is_dir():
        return []
    names: list[str] = []
    for path in directory.glob("*.pkl"):
        names.append(path.stem)
    return sorted(names)


def delete_collection_bundle(settings, collection: str, qdrant_path: str | None) -> None:
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
    print(f"Removed BM25/manifest artifacts for '{collection}'.")


def run_cleanup(settings, qdrant_path: str | None, execute: bool) -> None:
    qdrant_names = set(_list_qdrant_collections(qdrant_path))
    bm25_names = set(_list_bm25_collections(settings))
    to_remove = sorted((qdrant_names | bm25_names) - KEEP_COLLECTIONS)
    if not to_remove:
        print("No legacy collections to remove.")
        return
    print("Collections to remove (keeping fiqa + fiqa_cs200_bge_base_en_v1_5):")
    for name in to_remove:
        print(f"  - {name}")
    if not execute:
        print("Dry run only — pass --execute to delete.")
        return
    for name in to_remove:
        delete_collection_bundle(settings, name, qdrant_path)


def run_probe(settings, max_words: int, overlap: float) -> int:
    limit = settings.resolved_fiqa_corpus_limit()
    label = "full" if limit is None else str(limit)
    print(f"Probe: loading corpus (limit={label}), paragraph chunking...")
    dataset = load_fiqa_corpus(limit=limit)
    documents = documents_from_corpus(dataset, preserve_paragraphs=True)
    chunks = chunks_from_documents(
        documents,
        chunk_size=max_words,
        overlap_fraction=overlap,
        chunking_strategy="paragraph",
    )
    print(f"Documents: {len(documents):,}, paragraph chunks: {len(chunks):,}")
    out = BENCHMARK_DIR / "week5_e9_probe.json"
    out.write_text(
        json.dumps(
            {
                "num_documents": len(documents),
                "num_chunks": len(chunks),
                "max_words": max_words,
                "overlap_fraction": overlap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return len(chunks)


def _manifest_matches_paragraph(
    settings,
    collection: str,
    max_words: int,
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
        manifest.chunking_strategy == "paragraph"
        and manifest.chunk_size == max_words
        and abs(manifest.overlap_fraction - overlap) < 1e-9
        and manifest.embedding_model == embedding_model
    )


def _needs_treatment_ingest(
    settings,
    max_words: int,
    overlap: float,
    qdrant_path: str | None,
    force: bool,
) -> bool:
    if force:
        return True
    store = QdrantChunkStore.from_path(qdrant_path, TREATMENT_COLLECTION)
    try:
        if not store.collection_exists() or store.point_count() == 0:
            return True
        if not _manifest_matches_paragraph(
            settings, TREATMENT_COLLECTION, max_words, overlap, EMBEDDING_MODEL
        ):
            print(f"  [{TREATMENT_COLLECTION}] manifest mismatch — re-ingest required.")
            return True
        n = store.point_count()
        print(f"  [{TREATMENT_COLLECTION}] exists ({n:,} points) — skipping ingest.")
        return False
    finally:
        store.close()


def _run_treatment_ingest(
    max_words: int,
    overlap: float,
    qdrant_path: str | None,
    recreate: bool,
) -> None:
    cmd = [
        sys.executable,
        INGEST_SCRIPT,
        "--full-corpus",
        "--collection",
        TREATMENT_COLLECTION,
        "--chunking-strategy",
        "paragraph",
        "--chunk-size",
        str(max_words),
        "--overlap",
        str(overlap),
        "--method",
        "sentence_transformer",
        "--embedding-model",
        EMBEDDING_MODEL,
    ]
    if qdrant_path:
        cmd.extend(["--qdrant-path", qdrant_path])
    cmd.append("--recreate" if recreate else "--no-recreate")
    print(f"\n=== Ingest: {TREATMENT_COLLECTION} (paragraph, max_words={max_words}) ===")
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
        "sentence_transformer",
        model_name=EMBEDDING_MODEL,
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
    arm: str,
    split: str,
    qdrant_path: str | None,
    top_k: int,
    limit_queries: int | None,
    progress_every: int,
):
    collection = collection_for_arm(arm)
    eval_path = _prepare_eval(split, collection)
    dataset = _maybe_limit_dataset(load_eval_dataset(eval_path), limit_queries)
    store, embedder, bm25 = _open_store(settings, collection, qdrant_path)
    try:
        store.assert_collection_vector_size(embedding_dimension(embedder))
        points = store.point_count()
        print(f"\n=== Eval: arm={arm} collection={collection} split={split} ===")
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
    tag = e9_artifact_tag(arm)
    artifact = BENCHMARK_DIR / f"week5_e9_{tag}_{split}.json"
    save_eval_result(artifact, result)
    print(f"Wrote {artifact}")
    return result, points


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Week 5 E9 — paragraph-aware chunking")
    parser.add_argument(
        "--phase",
        choices=("probe", "cleanup", "ingest", "eval", "all"),
        default="all",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRACTION)
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="With cleanup: actually delete indexes",
    )
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--run-test-for-winner", action="store_true")
    args = parser.parse_args()

    qdrant_path = _qdrant_path(settings, args.qdrant_path)
    arms = (BASELINE_ARM, TREATMENT_ARM)
    collections = {arm: collection_for_arm(arm) for arm in arms}
    probe_count: int | None = None

    if args.phase == "probe":
        run_probe(settings, args.max_words, args.overlap)
        return

    if args.phase in ("cleanup", "all") and not args.skip_cleanup:
        run_cleanup(settings, qdrant_path, execute=args.execute or args.phase == "all")

    if args.phase in ("ingest", "all"):
        probe_count = run_probe(settings, args.max_words, args.overlap)
        if _needs_treatment_ingest(
            settings,
            args.max_words,
            args.overlap,
            qdrant_path,
            args.force_ingest,
        ):
            _run_treatment_ingest(
                args.max_words,
                args.overlap,
                qdrant_path,
                recreate=True,
            )

    results: dict[str, object] = {}
    point_counts: dict[str, int] = {}

    if args.phase in ("eval", "all"):
        for arm in arms:
            result, points = _run_eval_arm(
                settings,
                arm=arm,
                split=args.split,
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )
            results[arm] = result
            point_counts[arm] = points

        winner: str | None = None
        if args.split == "validation" and len(results) > 1:
            winner = best_arm_on_validation(results)  # type: ignore[arg-type]

        if probe_count is None:
            probe_path = BENCHMARK_DIR / "week5_e9_probe.json"
            if probe_path.is_file():
                probe_count = json.loads(probe_path.read_text(encoding="utf-8")).get("num_chunks")

        report = render_e9_report(
            results,  # type: ignore[arg-type]
            arms,
            collections=collections,
            point_counts=point_counts,
            max_words=args.max_words,
            overlap_fraction=args.overlap,
            qrels_split=args.split,
            corpus_note=_corpus_note(settings),
            winner_arm=winner,
            probe_chunk_count=probe_count,
        )
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / E9_REPORT).write_text(report, encoding="utf-8")
        print(f"Wrote {REPORT_DIR / E9_REPORT}")

        meta = {
            "max_words": args.max_words,
            "overlap_fraction": args.overlap,
            "embedding_model": EMBEDDING_MODEL,
            "arms": list(arms),
            "collections": collections,
            "split": args.split,
            "validation_winner_arm": winner,
        }
        (BENCHMARK_DIR / "week5_e9_protocol.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        if args.run_test_for_winner and winner is not None and args.split == "validation":
            print(f"\n=== Test eval for validation winner: {winner} ===")
            _run_eval_arm(
                settings,
                arm=winner,
                split="test",
                qdrant_path=qdrant_path,
                top_k=args.top_k,
                limit_queries=args.limit_queries,
                progress_every=args.progress_every,
            )


if __name__ == "__main__":
    main()
