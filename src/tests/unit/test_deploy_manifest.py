from app.config.settings import Settings, _REPO_ROOT
from app.ingestion.ingest_manifest import load_manifest, manifest_path_for_collection


def test_deploy_manifest_path_layout():
    settings = Settings(
        _env_file=None,
        bm25_index_dir="deploy/index_metadata",
        collection_name="fiqa_cs200_bge_base_en_v1_5",
    )
    manifest_path = manifest_path_for_collection(settings.resolved_bm25_index_path())
    assert manifest_path == (
        _REPO_ROOT / "deploy" / "index_metadata" / "fiqa_cs200_bge_base_en_v1_5.manifest.json"
    )


def test_packaged_deploy_manifest_loads():
    settings = Settings(
        _env_file=None,
        bm25_index_dir="deploy/index_metadata",
        collection_name="fiqa_cs200_bge_base_en_v1_5",
    )
    path = manifest_path_for_collection(settings.resolved_bm25_index_path())
    manifest = load_manifest(path)
    assert manifest is not None
    assert manifest.collection_name == "fiqa_cs200_bge_base_en_v1_5"
    assert manifest.embedding_model == "BAAI/bge-base-en-v1.5"
    assert manifest.num_chunks == 72851
