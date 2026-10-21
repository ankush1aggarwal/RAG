"""Frozen production retrieval profile (Week 5 E8 eval winner)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.pipeline.rag import RetrievalMethod

if TYPE_CHECKING:
    from app.config.settings import Settings

# Locked to E8 test protocol: dense, no rerank, top_k=10, 200w/20%, BGE-base.
FROZEN_COLLECTION_NAME: str = "fiqa_cs200_bge_base_en_v1_5"
FROZEN_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
FROZEN_CHUNK_SIZE: int = 200
FROZEN_CHUNK_OVERLAP_FRACTION: float = 0.2
FROZEN_FIQA_CORPUS_LIMIT: int = 0  # 0 = full BEIR FiQA corpus (~57k docs)
FROZEN_RETRIEVAL_TOP_K: int = 10
FROZEN_RETRIEVAL_METHOD: RetrievalMethod = "dense"
FROZEN_EMBEDDING_METHOD: str = "sentence_transformer"
FROZEN_RERANKER_PROVIDER: str = "none"


class FrozenRetrievalConfigError(ValueError):
    """Resolved settings do not match the frozen E8 production retrieval profile."""


def validate_frozen_retrieval_settings(settings: Settings) -> None:
    """
    When ``api_enforce_frozen_retrieval`` is true, require env/settings to match E8.

    Query-time enforcement alone is not enough: a legacy ``.env`` (MiniLM + ``fiqa``)
    must not start while claiming the frozen profile.
    """
    if not settings.api_enforce_frozen_retrieval:
        return
    mismatches: list[str] = []

    def expect(label: str, expected: object, actual: object) -> None:
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - expected) > 1e-9:
                mismatches.append(f"{label}: expected {expected!r}, got {actual!r}")
            return
        if actual != expected:
            mismatches.append(f"{label}: expected {expected!r}, got {actual!r}")

    expect("embedding_method", FROZEN_EMBEDDING_METHOD, settings.embedding_method)
    expect("embedding_model", FROZEN_EMBEDDING_MODEL, settings.embedding_model)
    expect("collection_name", FROZEN_COLLECTION_NAME, settings.collection_name)
    expect("chunk_size", FROZEN_CHUNK_SIZE, settings.chunk_size)
    expect("chunk_overlap_fraction", FROZEN_CHUNK_OVERLAP_FRACTION, settings.chunk_overlap_fraction)
    expect("fiqa_corpus_limit", FROZEN_FIQA_CORPUS_LIMIT, settings.fiqa_corpus_limit)
    expect("retrieval_top_k", FROZEN_RETRIEVAL_TOP_K, settings.retrieval_top_k)
    expect("reranker_provider", FROZEN_RERANKER_PROVIDER, settings.reranker_provider)

    if mismatches:
        lines = "\n".join(f"  - {line}" for line in mismatches)
        raise FrozenRetrievalConfigError(
            "API_ENFORCE_FROZEN_RETRIEVAL=true requires the frozen E8 retrieval "
            f"settings; fix `.env` or set API_ENFORCE_FROZEN_RETRIEVAL=false for dev:\n{lines}"
        )


@dataclass(frozen=True)
class ResolvedQueryRetrieval:
    method: RetrievalMethod
    top_k: int
    use_rerank: bool


def resolve_production_query(
    *,
    method: RetrievalMethod,
    top_k: int,
    use_rerank: bool,
    enforce_frozen: bool,
    frozen_top_k: int = FROZEN_RETRIEVAL_TOP_K,
) -> ResolvedQueryRetrieval:
    """Map HTTP request fields to the retrieval call (frozen profile when enforced)."""
    if not enforce_frozen:
        return ResolvedQueryRetrieval(method=method, top_k=top_k, use_rerank=use_rerank)
    if method != FROZEN_RETRIEVAL_METHOD:
        raise ValueError(
            f"Production API serves {FROZEN_RETRIEVAL_METHOD} retrieval only; got method={method!r}."
        )
    if use_rerank:
        raise ValueError("Production API does not enable reranking (frozen E8 profile).")
    if top_k != frozen_top_k:
        raise ValueError(
            f"Production API uses top_k={frozen_top_k} (frozen E8 profile); got top_k={top_k}."
        )
    return ResolvedQueryRetrieval(
        method=FROZEN_RETRIEVAL_METHOD,
        top_k=frozen_top_k,
        use_rerank=False,
    )
