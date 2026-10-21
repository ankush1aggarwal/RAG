"""HTTP request/response models (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.production_retrieval import FROZEN_RETRIEVAL_METHOD, FROZEN_RETRIEVAL_TOP_K
from app.pipeline.rag import RetrievalMethod, SourceCitation


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str = "ready"
    qdrant_collection: bool
    bm25_index: bool
    retrieval_index_aligned: bool
    llm_provider: str
    llm_ready: bool


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for RAG")
    top_k: int = Field(
        default=FROZEN_RETRIEVAL_TOP_K,
        ge=1,
        le=20,
        description=(
            "Number of chunks to retrieve (production default matches frozen E8 eval: top_k=10)"
        ),
    )
    method: RetrievalMethod = Field(
        default=FROZEN_RETRIEVAL_METHOD,
        description=(
            "Retrieval method (production API accepts dense only when frozen profile is enforced)"
        ),
    )
    use_rerank: bool = Field(
        default=False,
        description=(
            "Rerank retrieval candidates before generation "
            "(disabled on production API when frozen profile is enforced)"
        ),
    )


class RetrievalMeta(BaseModel):
    method: str = "dense"
    top_k: int
    reranker: str | None = None
    candidates_retrieved: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    retrieval: RetrievalMeta
