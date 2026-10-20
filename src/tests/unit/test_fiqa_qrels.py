from unittest.mock import MagicMock, patch

from app.evaluation.fiqa_qrels import (
    build_fiqa_eval_dataset,
    eval_dataset_to_dict,
    load_fiqa_eval_dataset,
    save_fiqa_eval_dataset,
)


@patch("app.evaluation.fiqa_qrels.load_fiqa_queries")
@patch("app.evaluation.fiqa_qrels.load_fiqa_qrels_rows")
def test_build_fiqa_eval_dataset_joins_queries_and_filters_index(
    mock_qrels: MagicMock,
    mock_queries: MagicMock,
):
    mock_qrels.return_value = [
        {"query-id": 8, "corpus-id": 100, "score": 1},
        {"query-id": 8, "corpus-id": 200, "score": 1},
        {"query-id": 9, "corpus-id": 300, "score": 1},
    ]
    mock_queries.return_value = {"8": "fees?", "9": "tax?"}

    dataset = build_fiqa_eval_dataset(
        "validation",
        indexed_doc_ids={"100"},
        min_relevant_in_index=1,
    )
    assert dataset.num_queries == 1
    assert dataset.queries[0].query_id == "8"
    assert dataset.queries[0].relevant_doc_ids == ("100",)


def test_eval_dataset_round_trip(tmp_path):
    from app.evaluation.fiqa_qrels import FiqaEvalDataset, FiqaEvalQuery

    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="test",
        queries=(
            FiqaEvalQuery(
                query_id="1",
                question="q",
                relevant_doc_ids=("42",),
                relevance_by_doc_id={"42": 1},
            ),
        ),
    )
    path = tmp_path / "fiqa_qrels_test.json"
    save_fiqa_eval_dataset(path, dataset)
    loaded = load_fiqa_eval_dataset(path)
    assert loaded.num_queries == 1
    assert eval_dataset_to_dict(loaded)["num_queries"] == 1
