from datasets import Dataset

from app.ingestion.normalize import documents_from_corpus, normalize_whitespace


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("  hello   world \n\t foo  ") == "hello world foo"


def test_documents_from_corpus_builds_records():
    dataset = Dataset.from_dict(
        {
            "doc": [
                "  first   doc  ",
                "second doc",
            ]
        }
    )
    records = documents_from_corpus(dataset)

    assert len(records) == 2
    assert records[0].doc_id == 0
    assert records[0].doc_text == "first doc"
    assert records[0].word_len == 2
    assert records[0].char_len == len("first doc")
    assert records[1].doc_id == 1
    assert records[1].doc_text == "second doc"
