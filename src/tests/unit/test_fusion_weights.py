import json

from app.retrieval.fusion_weights import FusionWeights, load_fusion_weights, save_fusion_weights


def test_fusion_weights_normalized():
    w = FusionWeights(dense=3.0, bm25=1.0)
    assert w.normalized() == (0.75, 0.25)
    assert w.as_list() == [0.75, 0.25]


def test_fusion_weights_zero_sum_falls_back_to_equal():
    w = FusionWeights(dense=0.0, bm25=0.0)
    assert w.normalized() == (1.0, 1.0)


def test_fusion_weights_save_load_roundtrip(tmp_path):
    path = tmp_path / "weights.json"
    weights = FusionWeights(
        dense=0.6,
        bm25=0.4,
        metric_optimized="ndcg@10",
        qrels_split="validation",
        num_queries=100,
        score_at_selection=0.42,
    )
    save_fusion_weights(path, weights)
    loaded = load_fusion_weights(path)
    assert loaded == weights
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["score_at_selection"] == 0.42
