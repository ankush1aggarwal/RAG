"""Download embedding weights into HF_HOME (Docker image build step)."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config.production_retrieval import FROZEN_EMBEDDING_MODEL
from app.retrieval.embeddings import SentenceTransformerEmbedding


def main() -> None:
    model_name = os.environ.get("EMBEDDING_MODEL", FROZEN_EMBEDDING_MODEL).strip()
    if not model_name:
        raise SystemExit("EMBEDDING_MODEL must not be empty")
    cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "(hub default)"
    print(f"Prefetching sentence-transformers model {model_name!r} into {cache}...")
    provider = SentenceTransformerEmbedding(model_name=model_name)
    print(f"Cached model dimension={provider.dimension}")


if __name__ == "__main__":
    main()
