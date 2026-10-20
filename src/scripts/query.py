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
from app.pipeline.rag import RetrievalMethod, retrieve_with_optional_rerank
from app.retrieval import (
    Bm25ChunkIndex,
    QdrantChunkStore,
    create_embedding_provider,
    create_reranker_provider,
)


def _print_hits(hits: list[SearchHit], *, reranker: str | None, candidates: int | None) -> None:
    if reranker:
        print(f"Reranker: {reranker} (candidates={candidates})")
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.chunk_text[:200] + ("..." if len(hit.chunk_text) > 200 else "")
        title = f" title={hit.doc_title!r}" if hit.doc_title else ""
        print(
            f"{rank}. score={hit.score:.4f} doc_id={hit.doc_id} chunk_id={hit.chunk_id}{title}\n"
            f"   {snippet}\n"
        )


def _cli_method_to_retrieval(method: str) -> RetrievalMethod:
    if method in ("sentence_transformer", "cohere"):
        return "dense"
    if method == "bm25":
        return "bm25"
    if method == "hybrid":
        return "hybrid"
    raise ValueError(f"Unsupported method: {method}")


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
        "--rerank",
        action="store_true",
        help="Rerank candidates before displaying (uses RERANKER_PROVIDER from settings)",
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

    retrieval_method = _cli_method_to_retrieval(args.method)
    bm25_path = args.bm25_path or str(settings.resolved_bm25_index_path(args.collection))
    bm25_index = None
    if retrieval_method in ("bm25", "hybrid"):
        bm25_index = Bm25ChunkIndex.load(bm25_path)

    qdrant_path = args.qdrant_path.strip() or None
    if qdrant_path is not None and not os.path.isabs(qdrant_path):
        qdrant_path = os.path.join(_REPO_ROOT, qdrant_path)

    dense_method = settings.embedding_method if args.method == "hybrid" else args.method
    if retrieval_method == "dense" and args.method in ("sentence_transformer", "cohere"):
        dense_method = args.method

    embedder = create_embedding_provider(
        dense_method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )

    reranker = None
    if args.rerank:
        reranker = create_reranker_provider(
            settings.reranker_provider,  # type: ignore[arg-type]
            cross_encoder_model=settings.cross_encoder_rerank_model,
            cohere_api_key=settings.cohere_api_key,
            cohere_rerank_model=settings.cohere_rerank_model,
        )
        if reranker is None:
            raise SystemExit("RERANKER_PROVIDER=none; cannot use --rerank")

    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    try:
        hits, reranker_name, candidate_count = retrieve_with_optional_rerank(
            args.query,
            retrieval_method=retrieval_method,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            top_k=args.top_k,
            use_rerank=args.rerank,
            reranker=reranker,
            rerank_candidate_limit=settings.rerank_candidate_limit,
        )
        _print_hits(hits, reranker=reranker_name, candidates=candidate_count)
    finally:
        store.close()


if __name__ == "__main__":
    main()
