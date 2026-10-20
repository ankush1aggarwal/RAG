"""Rerank retrieval candidates (local cross-encoder or Cohere Rerank)."""

from __future__ import annotations

import os
from typing import Literal, Protocol

from app.ingestion.schemas import SearchHit

RerankerMethod = Literal["cross_encoder", "cohere", "none"]


class RerankerProvider(Protocol):
    @property
    def name(self) -> str: ...

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._encoder = CrossEncoder(model_name)

    @property
    def name(self) -> str:
        return f"cross_encoder:{self._model_name}"

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not hits:
            return []
        pairs = [(query, hit.chunk_text) for hit in hits]
        scores = self._encoder.predict(pairs)
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [
            hit.model_copy(update={"score": float(score)})
            for hit, score in ranked
        ]


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        import cohere

        self._client = cohere.ClientV2(api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not hits:
            return []
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[hit.chunk_text for hit in hits],
            top_n=min(top_k, len(hits)),
        )
        out: list[SearchHit] = []
        for item in response.results:
            hit = hits[item.index]
            out.append(hit.model_copy(update={"score": float(item.relevance_score)}))
        return out


def create_reranker_provider(
    method: RerankerMethod,
    *,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cohere_api_key: str | None = None,
    cohere_rerank_model: str = "rerank-english-v3.0",
) -> RerankerProvider | None:
    if method == "none":
        return None
    if method == "cross_encoder":
        return CrossEncoderReranker(model_name=cross_encoder_model)
    if method == "cohere":
        key = cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is required for cohere reranker")
        return CohereReranker(api_key=key, model=cohere_rerank_model)
    raise ValueError(f"Invalid reranker method: {method}")


def resolve_rerank_candidate_limit(
    top_k: int,
    *,
    candidate_limit: int | None = None,
    default_candidate_limit: int = 50,
) -> int:
    """How many chunks to retrieve before reranking down to ``top_k``."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if candidate_limit is not None:
        return max(candidate_limit, top_k)
    return max(default_candidate_limit, top_k)
