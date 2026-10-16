"""Retrieval CLI: dense (Qdrant) or BM25 over indexed FiQA chunks."""

import argparse
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.ingestion.schemas import SearchHit
from app.retrieval import (
    Bm25ChunkIndex,
    QdrantChunkStore,
    create_embedding_provider,
    hybrid_search_text,
)


def _print_hits(hits: list[SearchHit]) -> None:
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk_text[:200] + ("..." if len(hit.chunk_text) > 200 else "")
        title = f" title={hit.doc_title!r}" if hit.doc_title else ""
        print(
            f"{rank}. score={hit.score:.4f} doc_id={hit.doc_id} chunk_id={hit.chunk_id}{title}\n"
            f"   {snippet}\n"
        )


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Query FiQA retrieval index")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument(
        "--method",
        choices=["sentence_transformer", "cohere", "bm25", "hybrid"],
        default=settings.embedding_method,
        help="Dense embedding provider, bm25, or hybrid (RRF over dense + bm25)",
    )
    parser.add_argument(
        "--qdrant-path",
        default=settings.qdrant_path,
        help="Local Qdrant storage path (dense methods only; empty string for in-memory)",
    )
    parser.add_argument(
        "--bm25-path",
        default=None,
        help="BM25 index file (default: data/bm25_indexes/<collection>.pkl)",
    )
    args = parser.parse_args()

    if args.method == "bm25":
        bm25_path = args.bm25_path or str(settings.resolved_bm25_index_path(args.collection))
        index = Bm25ChunkIndex.load(bm25_path)
        hits = index.search(args.query, limit=args.top_k)
        _print_hits(hits)
        return

    bm25_path = args.bm25_path or str(settings.resolved_bm25_index_path(args.collection))
    bm25_index = Bm25ChunkIndex.load(bm25_path) if args.method == "hybrid" else None

    qdrant_path = args.qdrant_path.strip() or None
    if qdrant_path is not None and not os.path.isabs(qdrant_path):
        qdrant_path = os.path.join(_REPO_ROOT, qdrant_path)

    dense_method = settings.embedding_method if args.method == "hybrid" else args.method
    embedder = create_embedding_provider(
        dense_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )

    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    try:
        if args.method == "hybrid":
            assert bm25_index is not None
            hits = hybrid_search_text(
                args.query,
                store=store,
                bm25_index=bm25_index,
                embedder=embedder,
                limit=args.top_k,
            )
        else:
            hits = store.search_text(args.query, embedder, limit=args.top_k)
        _print_hits(hits)
    finally:
        store.close()


if __name__ == "__main__":
    main()
