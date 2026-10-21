"""Normalize raw BEIR FiQA corpus rows into document records."""

import re

from datasets import Dataset

from app.ingestion.schemas import DocumentRecord

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_whitespace_preserve_paragraphs(text: str) -> str:
    """Collapse whitespace within paragraphs; keep blank-line paragraph boundaries."""
    paragraphs: list[str] = []
    for block in _PARAGRAPH_SPLIT.split(text):
        normalized = " ".join(block.split())
        if normalized:
            paragraphs.append(normalized)
    if not paragraphs:
        return normalize_whitespace(text)
    return "\n\n".join(paragraphs)


def documents_from_corpus(
    dataset: Dataset,
    *,
    preserve_paragraphs: bool = False,
) -> list[DocumentRecord]:
    records: list[DocumentRecord] = []
    for i in range(len(dataset)):
        row = dataset[i]
        raw_text = row["text"]
        raw_title = row.get("title") or ""
        doc_id = str(row["_id"])
        if preserve_paragraphs:
            doc_text = normalize_whitespace_preserve_paragraphs(raw_text)
        else:
            doc_text = normalize_whitespace(raw_text)
        doc_title = normalize_whitespace(raw_title) if raw_title else ""
        records.append(
            DocumentRecord(
                doc_id=doc_id,
                doc_title=doc_title,
                doc_text=doc_text,
                word_len=len(doc_text.split()),
                char_len=len(doc_text),
            )
        )
    return records
