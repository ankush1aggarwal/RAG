"""Frozen production retrieval profile (Week 5 E8 eval winner)."""

from __future__ import annotations

from dataclasses import dataclass

from app.pipeline.rag import RetrievalMethod

# Locked to E8 test protocol: dense, no rerank, top_k=10, 200w/20%, BGE-base.
FROZEN_COLLECTION_NAME: str = "fiqa_cs200_bge_base_en_v1_5"
FROZEN_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
FROZEN_CHUNK_SIZE: int = 200
FROZEN_CHUNK_OVERLAP_FRACTION: float = 0.2
FROZEN_FIQA_CORPUS_LIMIT: int = 0  # 0 = full BEIR FiQA corpus (~57k docs)
FROZEN_RETRIEVAL_TOP_K: int = 10
FROZEN_RETRIEVAL_METHOD: RetrievalMethod = "dense"


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
