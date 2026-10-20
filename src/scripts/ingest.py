"""Build FiQA chunk index in Qdrant (Week 1 baseline pipeline)."""

import argparse
import os
import sys
import uuid

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from app.config import get_settings
from app.ingestion import chunks_from_documents, documents_from_corpus, load_fiqa_corpus
from app.ingestion.ingest_manifest import (
    IngestManifest,
    IngestReuseValidationError,
    invalidate_published_ingest_artifacts,
    load_manifest,
    manifest_path_for_collection,
    publish_ingest_artifacts,
    validate_reuse,
)
from app.retrieval import (
    Bm25ChunkIndex,
    QdrantChunkStore,
    attach_embeddings,
    create_embedding_provider,
)
from app.retrieval.embeddings import (
    embedding_dimension,
    embedding_model_for_method,
    validate_chunk_embedding_dimensions,
)


def main() -> None:
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    settings = get_settings()

    if settings.hf_hub_disable_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    parser = argparse.ArgumentParser(description="Ingest FiQA corpus into Qdrant")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max corpus documents (default: FIQA_CORPUS_LIMIT from settings; 0 = full corpus)",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Ingest the full BEIR FiQA corpus (~57k docs). Same as --limit 0.",
    )
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
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete and rebuild Qdrant + BM25 if the collection exists (default: True). "
        "With --no-recreate, reuse only when corpus/chunking/embedding settings match "
        "the saved ingest manifest and point count.",
    )
    args = parser.parse_args()

    qdrant_path = args.qdrant_path.strip() or None
    if qdrant_path is not None and not os.path.isabs(qdrant_path):
        qdrant_path = os.path.join(_REPO_ROOT, qdrant_path)

    bm25_path = settings.resolved_bm25_index_path(args.collection)
    manifest_path = manifest_path_for_collection(bm25_path)
    if args.recreate:
        invalidate_published_ingest_artifacts(bm25_path)

    if args.full_corpus:
        corpus_limit: int | None = None
    elif args.limit is not None:
        corpus_limit = None if args.limit <= 0 else args.limit
    else:
        corpus_limit = settings.resolved_fiqa_corpus_limit()

    limit_label = "full" if corpus_limit is None else str(corpus_limit)
    print(f"Loading BEIR FiQA corpus (limit={limit_label})...")
    dataset = load_fiqa_corpus(limit=corpus_limit)
    documents = documents_from_corpus(dataset)
    chunks = chunks_from_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap_fraction=args.overlap,
    )
    print(f"Documents: {len(documents)}, chunks: {len(chunks)}")
    if not chunks:
        raise SystemExit("No chunks produced; check corpus sample and chunking settings.")

    embedder = create_embedding_provider(
        args.method,
        model_name=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
        cohere_embed_model=settings.cohere_embed_model,
        cohere_embed_batch_size=settings.cohere_embed_batch_size,
        cohere_embed_batch_sleep_seconds=settings.cohere_embed_batch_sleep_seconds,
    )
    expected_dim = embedding_dimension(embedder)
    print(f"Embedding model dimension: {expected_dim}")

    print("Embedding chunks...")
    attach_embeddings(chunks, embedder)
    validate_chunk_embedding_dimensions(chunks, expected_dim)

    manifest = IngestManifest(
        ingest_generation_id=str(uuid.uuid4()),
        corpus_limit=corpus_limit,
        chunk_size=args.chunk_size,
        overlap_fraction=args.overlap,
        embedding_method=args.method,
        embedding_model=embedding_model_for_method(
            args.method,
            sentence_transformer_model=settings.embedding_model,
            cohere_embed_model=settings.cohere_embed_model,
        ),
        embedding_dimension=expected_dim,
        collection_name=args.collection,
        num_documents=len(documents),
        num_chunks=len(chunks),
    )

    store = QdrantChunkStore.from_path(qdrant_path, args.collection)
    try:
        if not args.recreate and store.collection_exists():
            print(f"Validating existing collection '{args.collection}' for --no-recreate...")
            try:
                validate_reuse(
                    load_manifest(manifest_path),
                    manifest,
                    indexed_point_count=store.point_count(),
                )
            except IngestReuseValidationError as exc:
                raise SystemExit(str(exc)) from exc
            print("Configuration matches saved ingest manifest.")

        print(f"Building BM25 index for {len(chunks)} chunks (in memory until Qdrant upsert)...")
        bm25_index = Bm25ChunkIndex.from_chunks(chunks)

        store.prepare_collection(expected_dim, recreate=args.recreate)
        store.assert_collection_vector_size(expected_dim)
        print(f"Indexing {len(chunks)} chunks into '{args.collection}' (dim={expected_dim})...")
        store.upsert_chunks(chunks)

        publish_ingest_artifacts(bm25_index, manifest, bm25_path)
        print(f"BM25 index published to {bm25_path}")
        print(f"Ingest manifest published to {manifest_path}")
        print("Ingest complete.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
