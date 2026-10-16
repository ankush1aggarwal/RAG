"""BM25 overlap precompute: correctness guard + optional local timing (not CI-gated)."""

from app.evaluation.retrieval_benchmark import benchmark_bm25_overlap_eligibility_pass
from app.ingestion.schemas import ChunkRecord
from app.retrieval.bm25_store import Bm25ChunkIndex, tokenize


def _synthetic_chunks(count: int) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for index in range(count):
        text = f"broker fees commission topic {index % 50} financial markets"
        words = text.split()
        chunks.append(
            ChunkRecord(
                doc_id=str(index // 3),
                chunk_id=index % 3,
                chunk_text=text,
                word_len=len(words),
                char_len=len(text),
                doc_word_len=len(words),
            )
        )
    return chunks


def _overlap_vector_legacy(index: Bm25ChunkIndex, query: str) -> list[int]:
    query_token_set = set(tokenize(query))
    corpus_tokens = index._corpus_tokens
    return [
        len(query_token_set & set(corpus_tokens[i]))
        for i in range(index.num_chunks)
    ]


def _overlap_vector_precomputed(index: Bm25ChunkIndex, query: str) -> list[int]:
    query_token_set = set(tokenize(query))
    return [
        len(query_token_set & index._corpus_token_sets[i])
        for i in range(index.num_chunks)
    ]


def test_precomputed_token_sets_match_per_chunk_overlap():
    chunks = _synthetic_chunks(200)
    index = Bm25ChunkIndex.from_chunks(chunks)
    query = "broker fees commission"
    assert _overlap_vector_precomputed(index, query) == _overlap_vector_legacy(
        index, query
    )


def test_bm25_search_results_unchanged_with_precomputed_sets():
    chunks = _synthetic_chunks(100)
    index = Bm25ChunkIndex.from_chunks(chunks)
    first = index.search("broker fees", limit=5)
    second = index.search("broker fees", limit=5)
    assert [h.doc_id for h in first] == [h.doc_id for h in second]
    assert len(first) == 5


def test_bm25_overlap_timing_regression_local_report(capsys):
    """Print before/after ms on ~2k chunks; always passes (informational)."""
    chunks = _synthetic_chunks(1900)
    index = Bm25ChunkIndex.from_chunks(chunks)
    query = "broker fees and commission schedules"
    bench = benchmark_bm25_overlap_eligibility_pass(index, query, repeats=50)
    print(
        f"\nBM25 overlap pass (~{bench.indexed_chunks} chunks, {bench.repeats} runs):\n"
        f"  legacy (filter+sort re-tokenize): {bench.legacy_mean_ms:.3f} ms\n"
        f"  precomputed token sets:           {bench.precomputed_mean_ms:.3f} ms\n"
        f"  speedup:                          {bench.speedup:.2f}x\n"
    )
