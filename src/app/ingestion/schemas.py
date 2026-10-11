"""Data structures for FiQA documents and chunks."""

from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator


class DocumentRecord(BaseModel):
    doc_id: str
    doc_title: str = ""
    doc_text: str
    word_len: int
    char_len: int


class ChunkRecord(BaseModel):
    doc_id: str
    doc_title: str = ""
    chunk_id: int
    chunk_text: str
    word_len: int
    char_len: int
    doc_word_len: int
    chunk_embed: list[float] | None = None

    def set_embedding(self, vector: np.ndarray | list[float]) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        self.chunk_embed = arr.tolist()


class SearchHit(BaseModel):
    doc_id: str
    doc_title: str = ""
    chunk_id: int
    chunk_text: str
    score: float

    @field_validator("doc_id", mode="before")
    @classmethod
    def _coerce_doc_id(cls, value: Any) -> str:
        return str(value)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> "SearchHit":
        return cls(
            doc_id=payload["doc_id"],
            doc_title=payload.get("doc_title") or "",
            chunk_id=payload["chunk_id"],
            chunk_text=payload["chunk_text"],
            score=score,
        )
