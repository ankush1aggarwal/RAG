"""Normalize raw FiQA corpus rows into document records."""

from datasets import Dataset

from app.ingestion.schemas import DocumentRecord


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def documents_from_corpus(dataset: Dataset) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for i in range(len(dataset)):
        raw = dataset[i]["doc"]
        doc_text = normalize_whitespace(raw)
        records.append(
            DocumentRecord(
                doc_id=i,
                doc_text=doc_text,
                word_len=len(doc_text.split()),
                char_len=len(doc_text),
            )
        )
    return records
