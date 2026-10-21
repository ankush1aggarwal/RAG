from app.evaluation.e7_chunk_overlap import e7_collection_name, overlap_artifact_tag


def test_e7_baseline_reuses_fiqa_cs200():
    assert e7_collection_name(0.2) == "fiqa_cs200"


def test_e7_new_collections():
    assert e7_collection_name(0.1) == "fiqa_cs200_ov10"
    assert e7_collection_name(0.3) == "fiqa_cs200_ov30"


def test_overlap_artifact_tag():
    assert overlap_artifact_tag(0.1) == "ov10"
