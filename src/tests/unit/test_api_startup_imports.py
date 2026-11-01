"""Ensure API startup paths do not require optional ``datasets``."""

from __future__ import annotations

import importlib
import sys


def test_ingest_manifest_import_avoids_corpus_module():
    """``ingest_manifest`` is loaded at API startup; it must not import ``corpus`` (datasets)."""
    for name in list(sys.modules):
        if name == "app.ingestion.corpus" or name == "datasets":
            del sys.modules[name]
    importlib.import_module("app.ingestion.ingest_manifest")
    assert "app.ingestion.corpus" not in sys.modules
