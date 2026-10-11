"""Normalize raw BEIR FiQA corpus rows into document records."""

from datasets import Dataset

from app.ingestion.schemas import DocumentRecord


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def documents_from_corpus(dataset: Dataset) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for i in range(len(dataset)):
        row = dataset[i]
        raw_text = row["text"]
        raw_title = row.get("title") or ""
        doc_id = str(row["_id"])
        doc_text = normalize_whitespace(raw_text)
        doc_title = normalize_whitespace(raw_title) if raw_title else ""
        records.append(
            DocumentRecord(
                doc_id=doc_id,
                doc_title=doc_title,
                doc_text=doc_text,
                word_len=len(raw_text.split()),
                char_len=len(raw_text),
            )
        )
    return records
