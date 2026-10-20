from app.evaluation.fiqa_qrels import FiqaEvalDataset, FiqaEvalQuery
from app.evaluation.weight_tuning import HybridListCache, tune_fusion_weights
from app.ingestion.schemas import SearchHit


def _hit(doc_id: str) -> SearchHit:
    return SearchHit(doc_id=doc_id, chunk_id=0, chunk_text="t", score=1.0)


def test_tune_fusion_weights_selects_dense_when_only_dense_relevant():
    dataset = FiqaEvalDataset(
        version=1,
        corpus_id="BeIR/fiqa",
        qrels_split="validation",
        queries=(
            FiqaEvalQuery(
                query_id="1",
                question="q",
                relevant_doc_ids=("d",),
                relevance_by_doc_id={"d": 1},
            ),
        ),
    )
    cache = HybridListCache(
        by_query_id={
            "1": ([_hit("d")], [_hit("wrong")]),
        }
    )
    weights = tune_fusion_weights(
        dataset,
        cache,
        top_k=10,
        metric_name="ndcg@1",
        grid_steps=11,
    )
    assert weights.dense >= weights.bm25
