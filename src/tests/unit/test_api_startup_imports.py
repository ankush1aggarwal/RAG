"""Ensure API startup paths do not require optional ``datasets``."""

from __future__ import annotations

import importlib
import sys


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_ingest_manifest_import_avoids_corpus_module():
    _clear_modules("app.ingestion.corpus", "app.ingestion.normalize")
    importlib.import_module("app.ingestion.ingest_manifest")
    assert "app.ingestion.corpus" not in sys.modules


def test_importing_ingestion_schemas_does_not_load_corpus():
    """API code imports ``app.ingestion.schemas``; package ``__init__`` must stay light."""
    _clear_modules(
        "app.ingestion.schemas",
        "app.ingestion",
        "app.ingestion.corpus",
        "app.ingestion.normalize",
    )
    importlib.import_module("app.ingestion.schemas")
    assert "app.ingestion.corpus" not in sys.modules


def test_api_main_import_does_not_load_corpus(monkeypatch):
    """Importing the FastAPI app module must not pull in ``corpus`` (needs ``datasets``)."""
    _clear_modules(
        "app.api.main",
        "app.api.routes",
        "app.api.resources",
        "app.ingestion",
        "app.ingestion.schemas",
        "app.ingestion.corpus",
        "app.ingestion.normalize",
    )
    monkeypatch.setenv("API_ENFORCE_FROZEN_RETRIEVAL", "false")
    importlib.import_module("app.api.main")
    assert "app.ingestion.corpus" not in sys.modules
