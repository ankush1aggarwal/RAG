from app.evaluation.e6_chunk_size import e6_collection_name, resolve_e6_collection


def test_e6_collection_name():
    assert e6_collection_name(200) == "fiqa_cs200"


def test_resolve_baseline_100():
    assert resolve_e6_collection(100, baseline_collection="fiqa") == "fiqa"
    assert resolve_e6_collection(200, baseline_collection="fiqa") == "fiqa_cs200"


def test_resolve_prefixed_only():
    assert resolve_e6_collection(100, use_prefixed_collections_only=True) == "fiqa_cs100"
