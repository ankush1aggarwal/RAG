import pytest

from app.ingestion.ingest_manifest import (
    IndexAlignmentError,
    IngestManifest,
    IngestReuseValidationError,
    invalidate_published_ingest_artifacts,
    load_manifest,
    manifest_path_for_collection,
    manifests_compatible,
    publish_ingest_artifacts,
    validate_manifest_runtime_config,
    validate_retrieval_index_alignment,
    validate_reuse,
)
from app.ingestion.schemas import ChunkRecord
from app.retrieval.bm25_store import Bm25ChunkIndex
from app.retrieval.embeddings import embedding_model_for_method


def _manifest(**overrides: object) -> IngestManifest:
    base = {
        "ingest_generation_id": "gen-test-001",
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


def test_embedding_model_for_method_sentence_transformer():
    assert (
        embedding_model_for_method(
            "sentence_transformer",
            sentence_transformer_model="all-MiniLM-L6-v2",
            cohere_embed_model="embed-v4.0",
        )
        == "all-MiniLM-L6-v2"
    )


def test_embedding_model_for_method_cohere():
    assert (
        embedding_model_for_method(
            "cohere",
            sentence_transformer_model="all-MiniLM-L6-v2",
            cohere_embed_model="embed-v4.0",
        )
        == "embed-v4.0"
    )


def test_manifest_records_cohere_embed_model_when_method_is_cohere():
    manifest = _manifest(
        embedding_method="cohere",
        embedding_model="embed-v4.0",
    )
    assert manifest.embedding_method == "cohere"
    assert manifest.embedding_model == "embed-v4.0"
    assert manifest.embedding_model != "all-MiniLM-L6-v2"


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


def test_validate_retrieval_index_alignment_requires_matching_counts():
    manifest = _manifest(num_chunks=2)
    validate_retrieval_index_alignment(
        manifest,
        collection_name="fiqa_sample",
        qdrant_point_count=2,
        bm25_chunk_count=2,
        qdrant_vector_size=384,
        bm25_ingest_generation_id="gen-test-001",
    )
    with pytest.raises(IndexAlignmentError, match="Qdrant has 1 points"):
        validate_retrieval_index_alignment(
            manifest,
            collection_name="fiqa_sample",
            qdrant_point_count=1,
            bm25_chunk_count=2,
            qdrant_vector_size=384,
            bm25_ingest_generation_id="gen-test-001",
        )


def test_invalidate_published_ingest_artifacts_removes_pickle_and_manifest(tmp_path):
    bm25_path = tmp_path / "fiqa_sample.pkl"
    bm25_path.write_bytes(b"pickle")
    manifest_path = manifest_path_for_collection(bm25_path)
    manifest_path.write_text("{}", encoding="utf-8")
    invalidate_published_ingest_artifacts(bm25_path)
    assert not bm25_path.is_file()
    assert not manifest_path.is_file()


def test_validate_manifest_runtime_config_rejects_embedding_model_mismatch():
    manifest = _manifest(embedding_model="other-model")
    with pytest.raises(IndexAlignmentError, match="embedding_model"):
        validate_manifest_runtime_config(
            manifest,
            corpus_limit=1000,
            chunk_size=100,
            overlap_fraction=0.2,
            embedding_method="sentence_transformer",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            collection_name="fiqa_sample",
        )


def test_validate_manifest_runtime_config_requires_generation_id():
    manifest = _manifest(ingest_generation_id="")
    with pytest.raises(IndexAlignmentError, match="ingest_generation_id"):
        validate_manifest_runtime_config(
            manifest,
            corpus_limit=1000,
            chunk_size=100,
            overlap_fraction=0.2,
            embedding_method="sentence_transformer",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            collection_name="fiqa_sample",
        )


def test_validate_retrieval_index_alignment_checks_generation_id():
    manifest = _manifest(ingest_generation_id="gen-a", num_chunks=2)
    validate_retrieval_index_alignment(
        manifest,
        collection_name="fiqa_sample",
        qdrant_point_count=2,
        bm25_chunk_count=2,
        qdrant_vector_size=384,
        bm25_ingest_generation_id="gen-a",
    )
    with pytest.raises(IndexAlignmentError, match="ingest_generation_id"):
        validate_retrieval_index_alignment(
            manifest,
            collection_name="fiqa_sample",
            qdrant_point_count=2,
            bm25_chunk_count=2,
            qdrant_vector_size=384,
            bm25_ingest_generation_id="gen-b",
        )


def test_publish_ingest_artifacts_writes_pickle_and_manifest(tmp_path):
    chunks = [
        ChunkRecord(
            doc_id="1",
            chunk_id=0,
            chunk_text="broker fees",
            word_len=2,
            char_len=11,
            doc_word_len=2,
        )
    ]
    index = Bm25ChunkIndex.from_chunks(chunks)
    manifest = _manifest(num_documents=1, num_chunks=1)
    bm25_path = tmp_path / "fiqa_sample.pkl"
    publish_ingest_artifacts(index, manifest, bm25_path)
    assert bm25_path.is_file()
    loaded_manifest = load_manifest(manifest_path_for_collection(bm25_path))
    assert loaded_manifest is not None
    assert loaded_manifest.num_chunks == 1
    assert loaded_manifest.ingest_generation_id == "gen-test-001"
    loaded_index = Bm25ChunkIndex.load(bm25_path)
    assert loaded_index.num_chunks == 1
    assert loaded_index.ingest_generation_id == "gen-test-001"
