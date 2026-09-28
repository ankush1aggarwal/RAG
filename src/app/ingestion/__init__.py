"""Corpus loading, normalization, and chunking."""

from app.ingestion.chunking import chunk_text_fixed_size_with_overlap, chunks_from_documents
from app.ingestion.corpus import load_fiqa_corpus
from app.ingestion.normalize import documents_from_corpus, normalize_whitespace
from app.ingestion.schemas import ChunkRecord, DocumentRecord, SearchHit

__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "SearchHit",
    "chunk_text_fixed_size_with_overlap",
    "chunks_from_documents",
    "documents_from_corpus",
    "load_fiqa_corpus",
    "normalize_whitespace",
]
