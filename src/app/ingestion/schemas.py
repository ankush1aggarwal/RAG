"""Data structures for FiQA documents and chunks."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    doc_id: int
    doc_text: str
    word_len: int
    char_len: int


class ChunkRecord(BaseModel):
    doc_id: int
    chunk_id: int
    chunk_text: str
    word_len: int
    char_len: int
    chunk_embed: list[float] | None = None

    def set_embedding(self, vector: np.ndarray | list[float]) -> None:
        arr = np.asarray(vector, dtype=np.float32)
        self.chunk_embed = arr.tolist()


class SearchHit(BaseModel):
    doc_id: int
    chunk_id: int
    chunk_text: str
    score: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> "SearchHit":
        return cls(
            doc_id=payload["doc_id"],
            chunk_id=payload["chunk_id"],
            chunk_text=payload["chunk_text"],
            score=score,
        )
