"""Open a Qdrant store from application settings (local path or Qdrant Cloud)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from app.retrieval.qdrant_store import QdrantChunkStore

if TYPE_CHECKING:
    from app.config.settings import Settings


def resolve_local_qdrant_path(raw: str | None) -> str | None:
    """Resolve CLI/env path strings relative to the repo root (local disk only)."""
    from app.config.settings import _REPO_ROOT, _is_windows_drive_path

    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    if _is_windows_drive_path(stripped):
        return str(Path(stripped))
    path = Path(stripped)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path)


def open_qdrant_chunk_store(
    settings: Settings,
    collection_name: str,
    *,
    path_override: str | None = None,
) -> QdrantChunkStore:
    """
    Connect using ``QDRANT_URL`` when set; otherwise local ``QDRANT_PATH`` / override.

    ``path_override`` is ignored when a remote URL is configured.
    """
    url = settings.resolved_qdrant_url()
    if url:
        return QdrantChunkStore.from_connection(
            collection_name=collection_name,
            url=url,
            api_key=settings.resolved_qdrant_api_key(),
        )
    if path_override is not None:
        path = resolve_local_qdrant_path(path_override)
    else:
        path = settings.resolved_qdrant_path()
    return QdrantChunkStore.from_connection(collection_name=collection_name, path=path)
