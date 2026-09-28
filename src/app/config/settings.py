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

    fiqa_corpus_limit: int = 1000
    chunk_size: int = 100
    chunk_overlap_fraction: float = 0.2

    # Relative paths resolve from repository root. Use empty string for in-memory Qdrant.
    qdrant_path: str = "data/qdrant_local"
    collection_name: str = "fiqa_sample"

    def resolved_qdrant_path(self) -> str | None:
        if not self.qdrant_path:
            return None
        path = Path(self.qdrant_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return str(path)

    hf_hub_disable_symlinks_warning: bool = True


def get_settings() -> Settings:
    return Settings()
