"""Dense retrieval against indexed FiQA chunks."""

import argparse
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.retrieval import QdrantChunkStore, create_embedding_provider


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Query FiQA Qdrant index")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument(
        "--method",
        choices=["sentence_transformer", "cohere"],
        default=settings.embedding_method,
    )
    parser.add_argument(
        "--qdrant-path",
        default=settings.qdrant_path,
        help="Local Qdrant storage path (empty string for in-memory)",
    )
    args = parser.parse_args()

    qdrant_path = args.qdrant_path.strip() or None
    if qdrant_path is not None and not os.path.isabs(qdrant_path):
        qdrant_path = os.path.join(_REPO_ROOT, qdrant_path)

    embedder = create_embedding_provider(
        args.method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )

    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    hits = store.search_text(args.query, embedder, limit=args.top_k)

    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk_text[:200] + ("..." if len(hit.chunk_text) > 200 else "")
        print(
            f"{rank}. score={hit.score:.4f} doc_id={hit.doc_id} chunk_id={hit.chunk_id}\n   {snippet}\n"
        )


if __name__ == "__main__":
    main()
