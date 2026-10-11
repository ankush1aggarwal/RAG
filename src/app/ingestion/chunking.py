"""Fixed-size word chunking with overlap."""

from app.ingestion.schemas import ChunkRecord, DocumentRecord


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


def chunks_from_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 100,
    overlap_fraction: float = 0.2,
) -> list[ChunkRecord]:
    chunk_records: list[ChunkRecord] = []
    for doc in documents:
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
