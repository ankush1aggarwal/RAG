"""Load BEIR FiQA qrels and join with query text for retrieval evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import load_dataset

from app.ingestion.corpus import FIQA_HF_ID

FIQA_QRELS_HF_ID = "BeIR/fiqa-qrels"
FIQA_QUERIES_CONFIG = "queries"
FIQA_QUERIES_SPLIT = "queries"

QrelsSplit = Literal["train", "validation", "test"]
EVAL_DATASET_VERSION = 1


@dataclass(frozen=True)
class FiqaEvalQuery:
    """One FiQA query with document-level relevance labels (BEIR corpus-id)."""

    query_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    relevance_by_doc_id: dict[str, int]


@dataclass(frozen=True)
class FiqaEvalDataset:
    version: int
    corpus_id: str
    qrels_split: QrelsSplit
    queries: tuple[FiqaEvalQuery, ...]

    @property
    def num_queries(self) -> int:
        return len(self.queries)


def load_fiqa_queries() -> dict[str, str]:
    """Map BEIR query ``_id`` to query text."""
    dataset = load_dataset(FIQA_HF_ID, FIQA_QUERIES_CONFIG, split=FIQA_QUERIES_SPLIT)
    return {str(row["_id"]): str(row["text"]) for row in dataset}


def load_fiqa_qrels_rows(split: QrelsSplit) -> list[dict[str, object]]:
    dataset = load_dataset(FIQA_QRELS_HF_ID, "default", split=split)
    return [dict(row) for row in dataset]


def build_fiqa_eval_dataset(
    split: QrelsSplit,
    *,
    indexed_doc_ids: set[str] | None = None,
    min_relevant_in_index: int = 1,
) -> FiqaEvalDataset:
    """
    Join qrels with query strings.

    When ``indexed_doc_ids`` is set, keep only queries that have at least
    ``min_relevant_in_index`` relevant documents present in the index (for partial corpora).
    """
    query_text = load_fiqa_queries()
    relevance: dict[str, dict[str, int]] = defaultdict(dict)
    for row in load_fiqa_qrels_rows(split):
        query_id = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score = int(row["score"])
        if score <= 0:
            continue
        relevance[query_id][doc_id] = max(score, relevance[query_id].get(doc_id, 0))

    eval_queries: list[FiqaEvalQuery] = []
    for query_id, doc_scores in sorted(relevance.items(), key=lambda item: item[0]):
        question = query_text.get(query_id)
        if not question:
            continue
        relevant = tuple(sorted(doc_scores.keys()))
        if indexed_doc_ids is not None:
            in_index = [doc_id for doc_id in relevant if doc_id in indexed_doc_ids]
            if len(in_index) < min_relevant_in_index:
                continue
            relevant = tuple(in_index)
            doc_scores = {doc_id: doc_scores[doc_id] for doc_id in relevant}
        eval_queries.append(
            FiqaEvalQuery(
                query_id=query_id,
                question=question,
                relevant_doc_ids=relevant,
                relevance_by_doc_id=doc_scores,
            )
        )

    return FiqaEvalDataset(
        version=EVAL_DATASET_VERSION,
        corpus_id=FIQA_HF_ID,
        qrels_split=split,
        queries=tuple(eval_queries),
    )


def eval_dataset_to_dict(dataset: FiqaEvalDataset) -> dict[str, object]:
    return {
        "version": dataset.version,
        "corpus_id": dataset.corpus_id,
        "qrels_split": dataset.qrels_split,
        "num_queries": dataset.num_queries,
        "queries": [
            {
                "query_id": query.query_id,
                "question": query.question,
                "relevant_doc_ids": list(query.relevant_doc_ids),
                "relevance_by_doc_id": query.relevance_by_doc_id,
            }
            for query in dataset.queries
        ],
    }


def save_fiqa_eval_dataset(path: str | Path, dataset: FiqaEvalDataset) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(eval_dataset_to_dict(dataset), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_fiqa_eval_dataset(path: str | Path) -> FiqaEvalDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = tuple(
        FiqaEvalQuery(
            query_id=str(item["query_id"]),
            question=str(item["question"]),
            relevant_doc_ids=tuple(str(doc_id) for doc_id in item["relevant_doc_ids"]),
            relevance_by_doc_id={
                str(doc_id): int(score)
                for doc_id, score in item["relevance_by_doc_id"].items()
            },
        )
        for item in payload["queries"]
    )
    return FiqaEvalDataset(
        version=int(payload["version"]),
        corpus_id=str(payload["corpus_id"]),
        qrels_split=payload["qrels_split"],  # type: ignore[arg-type]
        queries=queries,
    )


def indexed_doc_ids_from_bm25(bm25_index: object) -> set[str]:
    """Unique ``doc_id`` values from a loaded ``Bm25ChunkIndex``."""
    chunks = getattr(bm25_index, "_chunks", None)
    if chunks is None:
        raise TypeError("expected Bm25ChunkIndex with _chunks")
    return {str(chunk.doc_id) for chunk in chunks}
