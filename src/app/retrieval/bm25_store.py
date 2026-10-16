"""BM25 lexical index over the same chunk records used for dense retrieval."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.ingestion.schemas import ChunkRecord, SearchHit


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class Bm25ChunkIndex:
    """In-memory BM25 index aligned with ``ChunkRecord`` order and metadata."""

    def __init__(self, bm25: BM25Okapi, chunks: list[ChunkRecord]) -> None:
        if len(chunks) != bm25.corpus_size:
            raise ValueError("BM25 corpus size must match chunk metadata length")
        self._bm25 = bm25
        self._chunks = chunks

    @classmethod
    def from_chunks(cls, chunks: list[ChunkRecord]) -> Bm25ChunkIndex:
        if not chunks:
            raise ValueError("cannot build BM25 index from empty chunks")
        tokenized_corpus = [tokenize(chunk.chunk_text) for chunk in chunks]
        return cls(BM25Okapi(tokenized_corpus), list(chunks))

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        hits: list[SearchHit] = []
        for index in ranked_indices:
            chunk = self._chunks[index]
            hits.append(
                SearchHit(
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.chunk_text,
                    score=float(scores[index]),
                )
            )
        return hits

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "chunks": [chunk.model_dump() for chunk in self._chunks],
            "bm25": self._bm25,
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Bm25ChunkIndex:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"BM25 index not found: {target}")
        with target.open("rb") as handle:
            payload = pickle.load(handle)
        chunks = [ChunkRecord(**item) for item in payload["chunks"]]
        return cls(payload["bm25"], chunks)
