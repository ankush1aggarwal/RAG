"""Application settings from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cohere_api_key: str | None = None
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_method: str = "sentence_transformer"  # or "cohere"
    cohere_embed_model: str = "embed-v4.0"
    cohere_embed_batch_size: int = 96
    cohere_embed_batch_sleep_seconds: float = 60.0

    llm_provider: str = "ollama"  # or "cohere" -- overridden in .env
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_timeout_seconds: float = 120.0
    cohere_chat_model: str = "command-r-08-2024"

    # Use 0 (or negative) for the full BEIR FiQA corpus (~57k documents).
    fiqa_corpus_limit: int = 1000
    chunk_size: int = 100
    chunk_overlap_fraction: float = 0.2

    # Relative paths resolve from repository root. Use empty string for in-memory Qdrant.
    qdrant_path: str = "data/qdrant_local"
    collection_name: str = "fiqa_sample"
    bm25_index_dir: str = "data/bm25_indexes"

    # Reranker (optional at query time; default provider is local cross-encoder)
    reranker_provider: str = "cross_encoder"  # cross_encoder | cohere | none
    cross_encoder_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cohere_rerank_model: str = "rerank-english-v3.0"
    rerank_candidate_limit: int = 50

    def resolved_fiqa_corpus_limit(self) -> int | None:
        """``None`` means load the full FiQA corpus (no document cap)."""
        if self.fiqa_corpus_limit <= 0:
            return None
        return self.fiqa_corpus_limit

    def resolved_qdrant_path(self) -> str | None:
        if not self.qdrant_path:
            return None
        path = Path(self.qdrant_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return str(path)

    def resolved_bm25_index_path(self, collection_name: str | None = None) -> Path:
        name = collection_name or self.collection_name
        directory = Path(self.bm25_index_dir)
        if not directory.is_absolute():
            directory = _REPO_ROOT / directory
        return directory / f"{name}.pkl"

    hf_hub_disable_symlinks_warning: bool = True


def get_settings() -> Settings:
    return Settings()
