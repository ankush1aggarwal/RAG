import pytest

from app.config.production_retrieval import (
    FROZEN_RETRIEVAL_TOP_K,
    resolve_production_query,
)


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
