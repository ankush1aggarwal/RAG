"""Load FiQA document corpus from Hugging Face (BEIR layout)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.ingestion.constants import FIQA_CORPUS_CONFIG, FIQA_CORPUS_SPLIT, FIQA_HF_ID

if TYPE_CHECKING:
    from datasets import Dataset


def load_fiqa_corpus(
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Any:
    """
    Load FiQA corpus documents.

    Args:
        limit: Max number of documents (e.g. 1000 for local dev). None loads full ~57k.
        shuffle: Whether to shuffle before taking ``limit`` rows.
        seed: Random seed when shuffle is True.
    """
    from datasets import load_dataset

    if limit is None:
        return load_dataset(FIQA_HF_ID, FIQA_CORPUS_CONFIG, split=FIQA_CORPUS_SPLIT)

    if shuffle:
        full = load_dataset(FIQA_HF_ID, FIQA_CORPUS_CONFIG, split=FIQA_CORPUS_SPLIT)
        return full.shuffle(seed=seed).select(range(min(limit, len(full))))

    return load_dataset(FIQA_HF_ID, FIQA_CORPUS_CONFIG, split=f"{FIQA_CORPUS_SPLIT}[:{limit}]")
