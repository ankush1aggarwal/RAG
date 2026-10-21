from app.evaluation.e9_paragraph_chunking import (
    BASELINE_COLLECTION,
    TREATMENT_COLLECTION,
    collection_for_arm,
    e9_artifact_tag,
)


def test_e9_collections():
    assert collection_for_arm("fixed_200w") == BASELINE_COLLECTION
    assert collection_for_arm("paragraph") == TREATMENT_COLLECTION
    assert TREATMENT_COLLECTION == "fiqa_para_bge_base_en_v1_5"


def test_e9_artifact_tags():
    assert e9_artifact_tag("paragraph") == "paragraph"
    assert e9_artifact_tag("fixed_200w") == "fixed_200w_bge"
