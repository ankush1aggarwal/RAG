import pytest

from app.ingestion.ingest_manifest import (
    IngestManifest,
    IngestReuseValidationError,
    manifests_compatible,
    validate_reuse,
)


def _manifest(**overrides: object) -> IngestManifest:
    base = {
        "corpus_limit": 1000,
        "chunk_size": 100,
        "overlap_fraction": 0.2,
        "embedding_method": "sentence_transformer",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "collection_name": "fiqa_sample",
        "num_documents": 1000,
        "num_chunks": 1900,
    }
    base.update(overrides)
    return IngestManifest(**base)


def test_manifests_compatible_when_equal():
    assert manifests_compatible(_manifest(), _manifest()) == []


def test_manifests_compatible_detects_chunk_size_change():
    current = _manifest(chunk_size=80)
    mismatches = manifests_compatible(_manifest(), current)
    assert any("chunk_size" in item for item in mismatches)


def test_validate_reuse_requires_stored_manifest():
    with pytest.raises(IngestReuseValidationError, match="No ingest manifest"):
        validate_reuse(None, _manifest(), indexed_point_count=1900)


def test_validate_reuse_checks_point_count():
    with pytest.raises(IngestReuseValidationError, match="500 points"):
        validate_reuse(_manifest(), _manifest(num_chunks=1900), indexed_point_count=500)
