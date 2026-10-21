from pathlib import Path

from app.config.production_retrieval import (
    FROZEN_CHUNK_SIZE,
    FROZEN_COLLECTION_NAME,
    FROZEN_EMBEDDING_MODEL,
    FROZEN_FIQA_CORPUS_LIMIT,
    FROZEN_RETRIEVAL_TOP_K,
)
from app.config.settings import _REPO_ROOT, Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.embedding_model == FROZEN_EMBEDDING_MODEL
    assert settings.embedding_method == "sentence_transformer"
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "mistral:latest"
    assert settings.ollama_timeout_seconds == 120.0
    assert settings.cohere_chat_model == "command-r-08-2024"
    assert settings.fiqa_corpus_limit == FROZEN_FIQA_CORPUS_LIMIT
    assert settings.chunk_size == FROZEN_CHUNK_SIZE
    assert settings.collection_name == FROZEN_COLLECTION_NAME
    assert settings.retrieval_top_k == FROZEN_RETRIEVAL_TOP_K
    assert settings.api_enforce_frozen_retrieval is True
    assert settings.bm25_index_dir == "data/bm25_indexes"


def test_resolved_bm25_index_path():
    settings = Settings(
        collection_name="fiqa_sample",
        bm25_index_dir="data/bm25_indexes",
        _env_file=None,
    )
    expected = _REPO_ROOT / "data" / "bm25_indexes" / "fiqa_sample.pkl"
    assert settings.resolved_bm25_index_path() == expected


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
