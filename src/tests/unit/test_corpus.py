from unittest.mock import MagicMock, patch

from datasets import Dataset

from app.ingestion.corpus import load_fiqa_corpus


@patch("app.ingestion.corpus.load_dataset")
def test_load_fiqa_corpus_with_limit_uses_slice_split(mock_load_dataset: MagicMock):
    mock_load_dataset.return_value = Dataset.from_dict({"doc": ["a", "b"]})

    result = load_fiqa_corpus(limit=1000, shuffle=False)

    mock_load_dataset.assert_called_once_with(
        "explodinggradients/fiqa",
        "corpus",
        split="corpus[:1000]",
    )
    assert len(result) == 2


@patch("app.ingestion.corpus.load_dataset")
def test_load_fiqa_corpus_shuffle_selects_subset(mock_load_dataset: MagicMock):
    full = Dataset.from_dict({"doc": [f"doc{i}" for i in range(10)]})
    mock_load_dataset.return_value = full

    result = load_fiqa_corpus(limit=3, shuffle=True, seed=99)

    mock_load_dataset.assert_called_once_with(
        "explodinggradients/fiqa",
        "corpus",
        split="corpus",
    )
    assert len(result) == 3


@patch("app.ingestion.corpus.load_dataset")
def test_load_fiqa_corpus_full_load(mock_load_dataset: MagicMock):
    mock_load_dataset.return_value = Dataset.from_dict({"doc": ["only"]})

    result = load_fiqa_corpus(limit=None)

    mock_load_dataset.assert_called_once_with(
        "explodinggradients/fiqa",
        "corpus",
        split="corpus",
    )
    assert len(result) == 1
