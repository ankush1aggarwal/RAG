"""HTTP request/response models (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.config.production_retrieval import FROZEN_RETRIEVAL_METHOD, FROZEN_RETRIEVAL_TOP_K
from app.pipeline.rag import RetrievalMethod, SourceCitation


class ErrorResponse(BaseModel):
    """Stable error body for non-2xx responses (see ``docs/api_contract.md``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error": "validation_error",
                    "message": "Request validation failed.",
                    "detail": [{"loc": ["body", "question"], "msg": "String should have at least 1 character"}],
                    "request_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ]
        }
    )

    error: str
    message: str
    detail: object | None = None
    request_id: str | None = None


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
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What is APR on a credit card?",
                    "top_k": 10,
                    "method": "dense",
                    "use_rerank": False,
                }
            ]
        }
    )

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
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "APR is the annual percentage rate...",
                    "citations": [
                        {"document_id": "7411", "chunk_id": 0, "title": ""},
                    ],
                    "retrieval": {
                        "method": "dense",
                        "top_k": 10,
                        "reranker": None,
                        "candidates_retrieved": None,
                    },
                }
            ]
        }
    )

    answer: str
    citations: list[SourceCitation]
    retrieval: RetrievalMeta
