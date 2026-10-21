"""Fixed-size and paragraph-aware chunking."""

from typing import Literal

from app.ingestion.schemas import ChunkRecord, DocumentRecord

ChunkingStrategy = Literal["fixed_words", "paragraph"]


def chunk_text_fixed_size_with_overlap(
    text: str,
    chunk_size: int = 100,
    overlap_fraction: float = 0.2,
) -> list[str]:
    text_words = text.split()

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be between 0 and 1")

    overlap_int = int(chunk_size * overlap_fraction)
    step = chunk_size - overlap_int
    if step < 1:
        raise ValueError("chunk_size must be greater than overlap window")

    chunks: list[str] = []

    for i in range(0, len(text_words), step):
        chunk_words = text_words[i : i + chunk_size]
        # Skip only trailing windows that are overlap-sized (not the first window).
        if chunks and len(chunk_words) <= overlap_int:
            continue
        chunks.append(" ".join(chunk_words))

    return chunks


def split_paragraphs(text: str) -> list[str]:
    if not text.strip():
        return []
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text_paragraph_aware(
    text: str,
    max_words: int = 200,
    overlap_fraction: float = 0.2,
) -> list[str]:
    """Split on paragraph boundaries, merge up to max_words, sub-split long paragraphs."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be between 0 and 1")

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        flat = " ".join(text.split())
        if not flat:
            return []
        paragraphs = [flat]

    output: list[str] = []
    buffer_parts: list[str] = []
    buffer_words = 0

    def flush_buffer() -> None:
        nonlocal buffer_parts, buffer_words
        if buffer_parts:
            output.append(" ".join(buffer_parts))
            buffer_parts = []
            buffer_words = 0

    for para in paragraphs:
        word_count = len(para.split())
        if word_count > max_words:
            flush_buffer()
            output.extend(
                chunk_text_fixed_size_with_overlap(
                    para,
                    chunk_size=max_words,
                    overlap_fraction=overlap_fraction,
                )
            )
            continue
        if buffer_words == 0:
            buffer_parts = [para]
            buffer_words = word_count
        elif buffer_words + word_count <= max_words:
            buffer_parts.append(para)
            buffer_words += word_count
        else:
            flush_buffer()
            buffer_parts = [para]
            buffer_words = word_count

    flush_buffer()
    return output


def chunks_from_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 100,
    overlap_fraction: float = 0.2,
    *,
    chunking_strategy: ChunkingStrategy = "fixed_words",
) -> list[ChunkRecord]:
    chunk_records: list[ChunkRecord] = []
    for doc in documents:
        if chunking_strategy == "paragraph":
            pieces = chunk_text_paragraph_aware(
                doc.doc_text,
                max_words=chunk_size,
                overlap_fraction=overlap_fraction,
            )
        else:
            pieces = chunk_text_fixed_size_with_overlap(
                doc.doc_text,
                chunk_size=chunk_size,
                overlap_fraction=overlap_fraction,
            )
        for chunk_id, chunk_text in enumerate(pieces):
            chunk_records.append(
                ChunkRecord(
                    doc_id=doc.doc_id,
                    doc_title=doc.doc_title,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    word_len=len(chunk_text.split()),
                    char_len=len(chunk_text),
                    doc_word_len=doc.word_len,
                )
            )
    return chunk_records
