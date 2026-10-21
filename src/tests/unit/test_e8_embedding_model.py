from app.evaluation.e8_embedding_model import (
    e8_collection_for_model,
    model_slug,
)


def test_model_slug():
    assert model_slug("BAAI/bge-base-en-v1.5") == "bge_base_en_v1_5"


def test_e8_collection_baseline():
    assert e8_collection_for_model("all-MiniLM-L6-v2") == "fiqa_cs200"


def test_e8_collection_alternative():
    assert e8_collection_for_model("BAAI/bge-base-en-v1.5") == "fiqa_cs200_bge_base_en_v1_5"
