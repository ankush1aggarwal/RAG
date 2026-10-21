import pytest

from app.api.resources import build_resources
from app.config.production_retrieval import (
    FROZEN_RETRIEVAL_TOP_K,
    FrozenRetrievalConfigError,
    resolve_production_query,
    validate_frozen_retrieval_settings,
)
from app.config.settings import Settings


def test_resolve_production_query_passthrough_when_not_enforced():
    resolved = resolve_production_query(
        method="hybrid",
        top_k=3,
        use_rerank=True,
        enforce_frozen=False,
    )
    assert resolved.method == "hybrid"
    assert resolved.top_k == 3
    assert resolved.use_rerank is True


def test_resolve_production_query_accepts_frozen_dense():
    resolved = resolve_production_query(
        method="dense",
        top_k=FROZEN_RETRIEVAL_TOP_K,
        use_rerank=False,
        enforce_frozen=True,
    )
    assert resolved.top_k == FROZEN_RETRIEVAL_TOP_K


def test_resolve_production_query_rejects_hybrid_when_enforced():
    with pytest.raises(ValueError, match="dense"):
        resolve_production_query(
            method="hybrid",
            top_k=FROZEN_RETRIEVAL_TOP_K,
            use_rerank=False,
            enforce_frozen=True,
        )


def test_resolve_production_query_rejects_rerank_when_enforced():
    with pytest.raises(ValueError, match="reranking"):
        resolve_production_query(
            method="dense",
            top_k=FROZEN_RETRIEVAL_TOP_K,
            use_rerank=True,
            enforce_frozen=True,
        )


def test_resolve_production_query_rejects_top_k_mismatch_when_enforced():
    with pytest.raises(ValueError, match="top_k=10"):
        resolve_production_query(
            method="dense",
            top_k=2,
            use_rerank=False,
            enforce_frozen=True,
        )


def _legacy_dev_settings() -> Settings:
    return Settings(
        _env_file=None,
        api_enforce_frozen_retrieval=True,
        embedding_model="all-MiniLM-L6-v2",
        collection_name="fiqa",
        chunk_size=100,
        chunk_overlap_fraction=0.2,
        fiqa_corpus_limit=1000,
        retrieval_top_k=10,
    )


def test_validate_frozen_settings_rejects_legacy_minilm_profile():
    with pytest.raises(FrozenRetrievalConfigError) as exc:
        validate_frozen_retrieval_settings(_legacy_dev_settings())
    message = str(exc.value)
    assert "embedding_model" in message
    assert "collection_name" in message
    assert "chunk_size" in message
    assert "fiqa_corpus_limit" in message


def test_validate_frozen_settings_skipped_when_enforcement_disabled():
    settings = _legacy_dev_settings()
    settings = settings.model_copy(update={"api_enforce_frozen_retrieval": False})
    validate_frozen_retrieval_settings(settings)


def test_validate_frozen_settings_accepts_default_production_settings():
    validate_frozen_retrieval_settings(Settings(_env_file=None))


def test_build_resources_fails_fast_on_legacy_settings_with_enforcement():
    with pytest.raises(FrozenRetrievalConfigError, match="collection_name"):
        build_resources(_legacy_dev_settings())


def test_build_resources_skips_reranker_when_frozen_enforced(monkeypatch):
    from unittest.mock import MagicMock

    from app.config.production_retrieval import FROZEN_COLLECTION_NAME

    rerank_called = False

    def _fake_rerank(*_args, **_kwargs):
        nonlocal rerank_called
        rerank_called = True
        return MagicMock()

    monkeypatch.setattr("app.api.resources.create_reranker_provider", _fake_rerank)
    embedder = MagicMock()
    embedder.dimension = 768
    monkeypatch.setattr(
        "app.api.resources.create_embedding_provider",
        lambda *_a, **_k: embedder,
    )
    monkeypatch.setattr("app.api.resources.create_llm_provider", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(
        "app.api.resources.QdrantChunkStore.from_path",
        lambda *_a, **_k: MagicMock(),
    )
    monkeypatch.setattr(
        "app.api.resources.Bm25ChunkIndex.load",
        lambda *_a: MagicMock(num_chunks=1, ingest_generation_id="gen"),
    )
    monkeypatch.setattr(
        "app.api.resources.load_manifest",
        lambda *_a: MagicMock(
            collection_name=FROZEN_COLLECTION_NAME,
            num_chunks=1,
            embedding_dimension=768,
            ingest_generation_id="gen",
            corpus_limit=None,
            chunk_size=200,
            overlap_fraction=0.2,
            embedding_method="sentence_transformer",
            embedding_model="BAAI/bge-base-en-v1.5",
            num_documents=1,
        ),
    )

    resources = build_resources(Settings(_env_file=None, api_enforce_frozen_retrieval=True))
    assert resources.reranker is None
    assert rerank_called is False
