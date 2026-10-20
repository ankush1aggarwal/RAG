from app.evaluation.fiqa_qrels import FiqaEvalQuery
from app.evaluation.rerank_error_analysis import (
    RerankCause,
    classify_dense_rerank_query,
)
from app.ingestion.schemas import SearchHit


def _hit(doc_id: str, text: str, chunk_id: int = 0) -> SearchHit:
    return SearchHit(doc_id=doc_id, chunk_id=chunk_id, chunk_text=text, score=1.0)


def test_classify_helped_by_ordering():
    query = FiqaEvalQuery(
        query_id="1",
        question="broker fees and commissions",
        relevant_doc_ids=("42",),
        relevance_by_doc_id={"42": 2},
    )
    dense = [_hit("99", "unrelated"), _hit("42", "broker fees vary widely")]
    reranked = [_hit("42", "broker fees vary widely"), _hit("99", "unrelated")]
    result = classify_dense_rerank_query(
        query,
        baseline_ranked=("99", "42"),
        rerank_ranked=("42", "99"),
        delta_ndcg10=0.2,
        dense_candidate_hits=dense,
        reranked_hits=reranked,
        candidate_doc_limit=50,
    )
    assert result.direction == "helped"
    assert result.primary_cause == RerankCause.IRRELEVANT_ORDERING


def test_classify_hurt_missing_candidates():
    query = FiqaEvalQuery(
        query_id="2",
        question="tax on etf dividends",
        relevant_doc_ids=("7",),
        relevance_by_doc_id={"7": 1},
    )
    dense = [_hit(str(i), f"chunk {i}") for i in range(10) if str(i) != "7"]
    reranked = dense[:2]
    result = classify_dense_rerank_query(
        query,
        baseline_ranked=tuple(str(i) for i in range(10) if str(i) != "7"),
        rerank_ranked=("0", "1"),
        delta_ndcg10=-0.1,
        dense_candidate_hits=dense,
        reranked_hits=reranked,
        candidate_doc_limit=50,
    )
    assert result.direction == "hurt"
    assert result.primary_cause == RerankCause.MISSING_CANDIDATES
