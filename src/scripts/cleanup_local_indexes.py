"""Remove local Qdrant + BM25 indexes except named collections (dry-run by default)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.ingestion.ingest_manifest import invalidate_published_ingest_artifacts
from app.retrieval.qdrant_store import QdrantChunkStore

DEFAULT_KEEP = ("fiqa", "fiqa_cs200_bge_base_en_v1_5")


def _qdrant_collections_on_disk(qdrant_path: str) -> list[str]:
    root = os.path.join(qdrant_path, "collection")
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
    )


def _bm25_collections(settings) -> list[str]:
    directory = settings.resolved_bm25_index_path("_x").parent
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.pkl"))


def _prune_meta_json(qdrant_path: str, keep: set[str]) -> None:
    meta_path = os.path.join(qdrant_path, "meta.json")
    if not os.path.isfile(meta_path):
        return
    data = json.loads(open(meta_path, encoding="utf-8").read())
    collections = data.get("collections", {})
    removed = [k for k in collections if k not in keep]
    for k in removed:
        del collections[k]
    if removed:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"Pruned meta.json collections: {removed}")


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Delete local indexes not in --keep list")
    parser.add_argument(
        "--keep",
        nargs="+",
        default=list(DEFAULT_KEEP),
        help="Collection names to retain",
    )
    parser.add_argument("--execute", action="store_true", help="Perform deletes")
    args = parser.parse_args()
    keep = set(args.keep)
    qdrant_path = settings.resolved_qdrant_path()
    if not qdrant_path:
        raise SystemExit("In-memory Qdrant — nothing to clean on disk.")

    qdrant_names = set(_qdrant_collections_on_disk(qdrant_path))
    bm25_names = set(_bm25_collections(settings))
    to_remove = sorted((qdrant_names | bm25_names) - keep)
    if not to_remove:
        print("Nothing to remove.")
        return
    print("Will remove (not in keep list):")
    for name in to_remove:
        print(f"  - {name}")
    print("Keeping:", ", ".join(sorted(keep)))
    if not args.execute:
        print("Dry run — pass --execute to delete.")
        return

    for name in to_remove:
        store = QdrantChunkStore.from_path(qdrant_path, name)
        try:
            if store.drop_collection():
                print(f"Qdrant dropped: {name}")
        finally:
            store.close()
        coll_dir = os.path.join(qdrant_path, "collection", name)
        if os.path.isdir(coll_dir):
            import shutil

            shutil.rmtree(coll_dir)
            print(f"Removed orphan dir: {coll_dir}")
        invalidate_published_ingest_artifacts(settings.resolved_bm25_index_path(name))

    _prune_meta_json(qdrant_path, keep)
    print("Done.")


if __name__ == "__main__":
    main()
