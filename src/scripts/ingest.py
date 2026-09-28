"""Build FiQA chunk index in Qdrant (Week 1 baseline pipeline)."""

import argparse
import os
import sys

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.ingestion import chunks_from_documents, documents_from_corpus, load_fiqa_corpus
from app.retrieval import QdrantChunkStore, attach_embeddings, create_embedding_provider


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Ingest FiQA corpus into Qdrant")
    parser.add_argument("--limit", type=int, default=settings.fiqa_corpus_limit)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument(
        "--method",
        choices=["sentence_transformer", "cohere"],
        default=settings.embedding_method,
    )
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=float, default=settings.chunk_overlap_fraction)
    parser.add_argument(
        "--qdrant-path",
        default=settings.qdrant_path,
        help="Local Qdrant storage path (empty string for in-memory)",
    )
    args = parser.parse_args()

    qdrant_path = args.qdrant_path.strip() or None
    if qdrant_path is not None and not os.path.isabs(qdrant_path):
        qdrant_path = os.path.join(_REPO_ROOT, qdrant_path)

    print(f"Loading FiQA corpus (limit={args.limit})...")
    dataset = load_fiqa_corpus(limit=args.limit)
    documents = documents_from_corpus(dataset)
    chunks = chunks_from_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap_fraction=args.overlap,
    )
    print(f"Documents: {len(documents)}, chunks: {len(chunks)}")

    embedder = create_embedding_provider(
        args.method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )

    print("Embedding chunks...")
    attach_embeddings(chunks, embedder)

    vector_size = len(chunks[0].chunk_embed or [])
    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    print(f"Creating collection '{args.collection}' (dim={vector_size})...")
    store.create_collection(vector_size)
    store.upsert_chunks(chunks)
    print("Ingest complete.")


if __name__ == "__main__":
    main()
