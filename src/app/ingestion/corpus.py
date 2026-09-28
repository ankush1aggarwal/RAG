"""Load FiQA document corpus from Hugging Face."""

from datasets import Dataset, load_dataset

FIQA_HF_ID = "explodinggradients/fiqa"
FIQA_CORPUS_CONFIG = "corpus"
FIQA_CORPUS_SPLIT = "corpus"


def load_fiqa_corpus(
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Dataset:
    """
    Load FiQA corpus documents.

    Args:
        limit: Max number of documents (e.g. 1000 for local dev). None loads full ~57k.
        shuffle: Whether to shuffle before taking ``limit`` rows.
        seed: Random seed when shuffle is True.
    """
    if limit is None:
        return load_dataset(FIQA_HF_ID, FIQA_CORPUS_CONFIG, split=FIQA_CORPUS_SPLIT)

    if shuffle:
        full = load_dataset(FIQA_HF_ID, FIQA_CORPUS_CONFIG, split=FIQA_CORPUS_SPLIT)
        return full.shuffle(seed=seed).select(range(min(limit, len(full))))

    return load_dataset(FIQA_HF_ID, FIQA_CORPUS_CONFIG, split=f"{FIQA_CORPUS_SPLIT}[:{limit}]")
