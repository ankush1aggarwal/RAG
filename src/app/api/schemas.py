"""HTTP request/response models (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field

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
    top_k: int = Field(default=3, ge=1, le=20, description="Number of chunks to retrieve")
    method: RetrievalMethod = Field(
        default="dense",
        description="Retrieval: dense (Qdrant), bm25, or hybrid (RRF)",
    )


class RetrievalMeta(BaseModel):
    method: str = "dense"
    top_k: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    retrieval: RetrievalMeta
