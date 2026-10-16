from pathlib import Path

from app.config.settings import Settings, _REPO_ROOT


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "all-MiniLM-L6-v2"
    assert settings.embedding_method == "sentence_transformer"
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "mistral:latest"
    assert settings.ollama_timeout_seconds == 120.0
    assert settings.cohere_chat_model == "command-r-08-2024"
    assert settings.fiqa_corpus_limit == 1000
    assert settings.chunk_size == 100
    assert settings.collection_name == "fiqa_sample"
    assert settings.bm25_index_dir == "data/bm25_indexes"


def test_resolved_bm25_index_path():
    settings = Settings(collection_name="fiqa_sample", bm25_index_dir="data/bm25_indexes", _env_file=None)
    assert settings.resolved_bm25_index_path() == _REPO_ROOT / "data" / "bm25_indexes" / "fiqa_sample.pkl"


def test_resolved_qdrant_path_relative_to_repo_root():
    settings = Settings(qdrant_path="data/qdrant_local", _env_file=None)
    resolved = settings.resolved_qdrant_path()
    assert resolved == str(_REPO_ROOT / "data" / "qdrant_local")


def test_resolved_qdrant_path_empty_means_in_memory():
    settings = Settings(qdrant_path="", _env_file=None)
    assert settings.resolved_qdrant_path() is None


def test_resolved_qdrant_path_keeps_absolute_path():
    abs_path = Path("D:/custom/qdrant")
    settings = Settings(qdrant_path=str(abs_path), _env_file=None)
    assert Path(settings.resolved_qdrant_path()) == abs_path
