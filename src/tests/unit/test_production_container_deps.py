"""Production container dependency surface (matches Dockerfile extras)."""

import importlib.util


def test_dockerfile_hosted_extra_cohere_available():
    """GHCR image uses pip install '.[api,embeddings,hosted]' for Cloud Run chat."""
    spec = importlib.util.find_spec("cohere")
    assert spec is not None, "install pyproject [hosted] or requirements.txt cohere for prod parity"


def test_dockerfile_embeddings_extra_available():
    spec = importlib.util.find_spec("sentence_transformers")
    assert spec is not None
