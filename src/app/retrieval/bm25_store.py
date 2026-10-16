"""BM25 lexical index over the same chunk records used for dense retrieval.

Indexes are persisted with :mod:`pickle`. Only load files you created locally via
``ingest.py`` (or otherwise trust); untrusted pickle files can execute arbitrary code
during ``pickle.load``.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.ingestion.schemas import ChunkRecord, SearchHit

# Alphanumeric tokens plus parenthetical forms (e.g. 401(k)); ignores surrounding punctuation.
_TOKEN_PATTERN = re.compile(
    r"\d+\([a-z0-9]+\)|[a-z0-9]+(?:'[a-z]+)?",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """Tokenizer for BM25 corpus and queries (regex-based, not whitespace split)."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class Bm25ChunkIndex:
    """In-memory BM25 index aligned with ``ChunkRecord`` order and metadata."""

    def __init__(
        self,
        bm25: BM25Okapi,
        chunks: list[ChunkRecord],
        corpus_tokens: list[list[str]],
    ) -> None:
        if len(chunks) != bm25.corpus_size:
            raise ValueError("BM25 corpus size must match chunk metadata length")
        if len(corpus_tokens) != len(chunks):
            raise ValueError("corpus token rows must match chunk metadata length")
        self._bm25 = bm25
        self._chunks = chunks
        self._corpus_tokens = corpus_tokens
        self._corpus_token_sets = tuple(frozenset(tokens) for tokens in corpus_tokens)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def unique_doc_count(self) -> int:
        return len({chunk.doc_id for chunk in self._chunks})

    @classmethod
    def from_chunks(cls, chunks: list[ChunkRecord]) -> Bm25ChunkIndex:
        if not chunks:
            raise ValueError("cannot build BM25 index from empty chunks")
        corpus_tokens = [tokenize(chunk.chunk_text) for chunk in chunks]
        return cls(BM25Okapi(corpus_tokens), list(chunks), corpus_tokens)

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        overlaps = [
            len(query_token_set & self._corpus_token_sets[index])
            for index in range(len(scores))
        ]

        eligible = [index for index, count in enumerate(overlaps) if count > 0]
        if not eligible:
            return []

        ranked_indices = sorted(
            eligible,
            key=lambda index: (scores[index], overlaps[index]),
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
        """Load a pickle written by :meth:`save`.

        **Security:** uses ``pickle.load``. Do not open indexes from untrusted sources.
        """
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"BM25 index not found: {target}")
        with target.open("rb") as handle:
            payload = pickle.load(handle)
        chunks = [ChunkRecord(**item) for item in payload["chunks"]]
        corpus_tokens = [tokenize(chunk.chunk_text) for chunk in chunks]
        bm25 = BM25Okapi(corpus_tokens)
        return cls(bm25, chunks, corpus_tokens)
