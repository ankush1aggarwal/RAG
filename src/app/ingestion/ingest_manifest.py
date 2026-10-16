"""Persisted metadata for safe ``ingest.py --no-recreate`` reuse."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from app.ingestion.corpus import FIQA_HF_ID

MANIFEST_VERSION = 1


class IngestManifest(BaseModel):
    version: int = MANIFEST_VERSION
    corpus_id: str = FIQA_HF_ID
    corpus_limit: int
    chunk_size: int
    overlap_fraction: float
    embedding_method: str
    embedding_model: str
    embedding_dimension: int
    collection_name: str
    num_documents: int
    num_chunks: int


class IngestReuseValidationError(ValueError):
    """Raised when an existing index does not match the current ingest configuration."""


def manifest_path_for_collection(bm25_pickle_path: Path) -> Path:
    return bm25_pickle_path.with_suffix(".manifest.json")


def save_manifest(path: Path, manifest: IngestManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(path: Path) -> IngestManifest | None:
    if not path.is_file():
        return None
    return IngestManifest.model_validate_json(path.read_text(encoding="utf-8"))


def manifests_compatible(stored: IngestManifest, current: IngestManifest) -> list[str]:
    mismatches: list[str] = []
    for field in (
        "corpus_id",
        "corpus_limit",
        "chunk_size",
        "overlap_fraction",
        "embedding_method",
        "embedding_model",
        "embedding_dimension",
        "collection_name",
        "num_documents",
        "num_chunks",
    ):
        stored_val = getattr(stored, field)
        current_val = getattr(current, field)
        if stored_val != current_val:
            mismatches.append(
                f"{field}: indexed {stored_val!r} vs current {current_val!r}"
            )
    return mismatches


def validate_reuse(
    stored: IngestManifest | None,
    current: IngestManifest,
    *,
    indexed_point_count: int | None,
) -> None:
    if stored is None:
        raise IngestReuseValidationError(
            "No ingest manifest found beside the BM25 index. "
            "Run once with default --recreate to build a fresh index and manifest, "
            "or use --recreate when corpus or chunking settings change."
        )
    mismatches = manifests_compatible(stored, current)
    if mismatches:
        detail = "; ".join(mismatches)
        raise IngestReuseValidationError(
            f"Ingest configuration does not match the existing index ({detail}). "
            "Use --recreate to delete and rebuild Qdrant + BM25."
        )
    if indexed_point_count is not None and indexed_point_count != current.num_chunks:
        raise IngestReuseValidationError(
            f"Qdrant collection has {indexed_point_count} points but this ingest produces "
            f"{current.num_chunks} chunks. Use --recreate to rebuild."
        )
