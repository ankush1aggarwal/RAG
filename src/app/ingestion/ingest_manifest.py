"""Persisted metadata for safe ``ingest.py --no-recreate`` reuse."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from app.ingestion.corpus import FIQA_HF_ID

if TYPE_CHECKING:
    from app.retrieval.bm25_store import Bm25ChunkIndex

MANIFEST_VERSION = 1

RUNTIME_CONFIG_FIELDS = (
    "corpus_id",
    "corpus_limit",
    "chunk_size",
    "overlap_fraction",
    "embedding_method",
    "embedding_model",
    "embedding_dimension",
    "collection_name",
)


class IngestManifest(BaseModel):
    version: int = MANIFEST_VERSION
    ingest_generation_id: str = ""
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


class IndexAlignmentError(ValueError):
    """Raised when Qdrant, BM25, and manifest disagree."""


def manifest_path_for_collection(bm25_pickle_path: Path) -> Path:
    return bm25_pickle_path.with_suffix(".manifest.json")


def save_manifest(path: Path, manifest: IngestManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(path: Path) -> IngestManifest | None:
    if not path.is_file():
        return None
    return IngestManifest.model_validate_json(path.read_text(encoding="utf-8"))


def invalidate_published_ingest_artifacts(bm25_path: Path) -> None:
    """Remove on-disk BM25 + manifest so a failed rebuild cannot satisfy readiness."""
    manifest_path = manifest_path_for_collection(bm25_path)
    manifest_path.unlink(missing_ok=True)
    bm25_path.unlink(missing_ok=True)


def runtime_config_mismatches(
    manifest: IngestManifest,
    *,
    corpus_limit: int,
    chunk_size: int,
    overlap_fraction: float,
    embedding_method: str,
    embedding_model: str,
    embedding_dimension: int,
    collection_name: str,
    corpus_id: str = FIQA_HF_ID,
) -> list[str]:
    """Compare manifest to the API / ingest configuration (not chunk counts)."""
    expected = {
        "corpus_id": corpus_id,
        "corpus_limit": corpus_limit,
        "chunk_size": chunk_size,
        "overlap_fraction": overlap_fraction,
        "embedding_method": embedding_method,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "collection_name": collection_name,
    }
    mismatches: list[str] = []
    for field in RUNTIME_CONFIG_FIELDS:
        stored_val = getattr(manifest, field)
        current_val = expected[field]
        if stored_val != current_val:
            mismatches.append(
                f"{field}: manifest {stored_val!r} vs runtime {current_val!r}"
            )
    return mismatches


def validate_manifest_runtime_config(
    manifest: IngestManifest,
    *,
    corpus_limit: int,
    chunk_size: int,
    overlap_fraction: float,
    embedding_method: str,
    embedding_model: str,
    embedding_dimension: int,
    collection_name: str,
    corpus_id: str = FIQA_HF_ID,
) -> None:
    mismatches = runtime_config_mismatches(
        manifest,
        corpus_limit=corpus_limit,
        chunk_size=chunk_size,
        overlap_fraction=overlap_fraction,
        embedding_method=embedding_method,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        collection_name=collection_name,
        corpus_id=corpus_id,
    )
    if mismatches:
        detail = "; ".join(mismatches)
        raise IndexAlignmentError(
            f"Ingest manifest does not match runtime configuration ({detail}). "
            "Re-run ingest with matching settings or update the service configuration."
        )
    if not manifest.ingest_generation_id:
        raise IndexAlignmentError(
            "Ingest manifest is missing ingest_generation_id; re-run ingest with --recreate."
        )


def manifests_compatible(stored: IngestManifest, current: IngestManifest) -> list[str]:
    mismatches: list[str] = []
    for field in (*RUNTIME_CONFIG_FIELDS, "num_documents", "num_chunks"):
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


def validate_retrieval_index_alignment(
    manifest: IngestManifest,
    *,
    collection_name: str,
    qdrant_point_count: int,
    bm25_chunk_count: int,
    qdrant_vector_size: int | None = None,
    bm25_ingest_generation_id: str | None = None,
) -> None:
    """Ensure manifest, Qdrant collection, and BM25 chunk counts are consistent."""
    if manifest.collection_name != collection_name:
        raise IndexAlignmentError(
            f"Manifest collection {manifest.collection_name!r} does not match "
            f"configured collection {collection_name!r}."
        )
    if qdrant_point_count != manifest.num_chunks:
        raise IndexAlignmentError(
            f"Qdrant has {qdrant_point_count} points but manifest expects "
            f"{manifest.num_chunks} chunks."
        )
    if bm25_chunk_count != manifest.num_chunks:
        raise IndexAlignmentError(
            f"BM25 index has {bm25_chunk_count} chunks but manifest expects "
            f"{manifest.num_chunks}."
        )
    if qdrant_vector_size is not None and qdrant_vector_size != manifest.embedding_dimension:
        raise IndexAlignmentError(
            f"Qdrant vector size is {qdrant_vector_size} but manifest expects "
            f"{manifest.embedding_dimension}."
        )
    if manifest.ingest_generation_id:
        if bm25_ingest_generation_id != manifest.ingest_generation_id:
            raise IndexAlignmentError(
                "BM25 index ingest_generation_id does not match manifest "
                f"({bm25_ingest_generation_id!r} vs {manifest.ingest_generation_id!r})."
            )


def publish_ingest_artifacts(
    bm25_index: Bm25ChunkIndex,
    manifest: IngestManifest,
    bm25_path: Path,
) -> None:
    """Write BM25 pickle and manifest to temp files, then atomically replace published paths."""
    manifest_path = manifest_path_for_collection(bm25_path)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_tmp = bm25_path.with_name(f"{bm25_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f"{manifest_path.name}.tmp")
    try:
        bm25_index.save(bm25_tmp, ingest_generation_id=manifest.ingest_generation_id)
        save_manifest(manifest_tmp, manifest)
        os.replace(bm25_tmp, bm25_path)
        os.replace(manifest_tmp, manifest_path)
    except Exception:
        bm25_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise
    finally:
        bm25_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
