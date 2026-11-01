"""Ingestion schemas and chunking (API-safe: no optional ``datasets`` import here).

Import ``corpus`` / ``normalize`` from their modules in ingest scripts only.
Loading them in ``__init__`` would run whenever any ``app.ingestion.*`` submodule
is imported (e.g. ``schemas`` on the API path).
"""

from app.ingestion.chunking import (
    chunk_text_fixed_size_with_overlap,
    chunk_text_paragraph_aware,
    chunks_from_documents,
    split_paragraphs,
)
from app.ingestion.schemas import ChunkRecord, DocumentRecord, SearchHit

__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "SearchHit",
    "chunk_text_fixed_size_with_overlap",
    "chunk_text_paragraph_aware",
    "chunks_from_documents",
    "split_paragraphs",
]
