"""Fixed-size word chunking with overlap."""

from app.ingestion.schemas import ChunkRecord, DocumentRecord


def chunk_text_fixed_size_with_overlap(
    text: str,
    chunk_size: int = 100,
    overlap_fraction: float = 0.2,
) -> list[str]:
    text_words = text.split()
    overlap_int = int(chunk_size * overlap_fraction)
    chunks: list[str] = []

    for i in range(0, len(text_words), chunk_size):
        chunk_words = text_words[max(i - overlap_int, 0) : i + chunk_size]
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
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    word_len=len(chunk_text.split()),
                    char_len=len(chunk_text),
                )
            )
    return chunk_records
