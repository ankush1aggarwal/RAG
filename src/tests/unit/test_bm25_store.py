from pathlib import Path

import pytest

from app.ingestion.schemas import ChunkRecord
from app.retrieval.bm25_store import Bm25ChunkIndex, tokenize


def _chunk(doc_id: str, chunk_id: int, text: str, title: str = "") -> ChunkRecord:
    words = text.split()
    return ChunkRecord(
        doc_id=doc_id,
        doc_title=title,
        chunk_id=chunk_id,
        chunk_text=text,
        word_len=len(words),
        char_len=len(text),
        doc_word_len=len(words),
    )


def test_tokenize_lowercases():
    assert tokenize("Broker FEES") == ["broker", "fees"]


def test_tokenize_strips_surrounding_punctuation():
    assert tokenize("What are broker fees?") == ["what", "are", "broker", "fees"]
    assert tokenize("fees?") == ["fees"]


def test_tokenize_preserves_financial_tokens():
    assert tokenize("401(k) employer match") == ["401(k)", "employer", "match"]
    assert tokenize("APR on credit cards") == ["apr", "on", "credit", "cards"]


def test_tokenize_is_regex_not_whitespace_split():
    assert tokenize("fees?") == ["fees"]
    assert tokenize("fees?") != "fees?".lower().split()


def test_corpus_and_query_use_same_tokenizer():
    text = "Broker fees? 401(k) and APR."
    chunks = [_chunk("1", 0, text)]
    index = Bm25ChunkIndex.from_chunks(chunks)
    assert index._corpus_tokens[0] == tokenize(text)
    assert index.search("fees? 401(k)")[0].doc_id == "1"


def test_from_chunks_rejects_empty():
    with pytest.raises(ValueError, match="empty chunks"):
        Bm25ChunkIndex.from_chunks([])


def test_search_prefers_lexical_match():
    chunks = [
        _chunk("1", 0, "broker fees commission retail trading"),
        _chunk("2", 0, "pasta recipes italian cooking"),
    ]
    index = Bm25ChunkIndex.from_chunks(chunks)
    hits = index.search("broker fees", limit=1)

    assert len(hits) == 1
    assert hits[0].doc_id == "1"
    assert hits[0].chunk_id == 0
    assert hits[0].chunk_text == chunks[0].chunk_text

    all_hits = index.search("broker fees", limit=2)
    assert len(all_hits) == 1
    assert all_hits[0].doc_id == "1"
    assert all_hits[0].score >= 0


def test_bm25_ranking_orders_by_lexical_relevance():
    chunks = [
        _chunk("1", 0, "broker fees explained for retail investors"),
        _chunk("2", 0, "broker commission schedule and fee tables"),
        _chunk("3", 0, "unrelated pasta recipes and cooking tips"),
    ]
    index = Bm25ChunkIndex.from_chunks(chunks)
    hits = index.search("broker fees", limit=3)

    assert [h.doc_id for h in hits] == ["1", "2"]
    assert hits[0].score >= hits[1].score
    assert all(hit.score > 0 for hit in hits)


def test_bm25_same_doc_id_different_chunk_id_ranked_separately():
    chunks = [
        _chunk("5", 0, "general market commentary"),
        _chunk("5", 1, "broker fees and commission details"),
    ]
    chunks.extend(
        _chunk(str(i), 0, f"unrelated financial news and market summary {i}")
        for i in range(10)
    )
    index = Bm25ChunkIndex.from_chunks(chunks)
    hits = index.search("broker fees", limit=3)

    assert hits[0].doc_id == "5"
    assert hits[0].chunk_id == 1
    assert hits[0].score > 0
    assert all(hit.score >= 0 for hit in hits)


def test_bm25_empty_query_returns_no_hits():
    index = Bm25ChunkIndex.from_chunks([_chunk("1", 0, "broker fees")])
    assert index.search("   ", limit=3) == []


def test_bm25_no_lexical_match_returns_no_hits():
    index = Bm25ChunkIndex.from_chunks([_chunk("1", 0, "broker fees commission")])
    assert index.search("zzzznoterm", limit=3) == []


def test_bm25_returns_overlap_when_bm25_score_is_zero_on_tiny_corpus():
    chunks = [
        _chunk("1", 0, "broker fees commission"),
        _chunk("2", 0, "pasta recipes italian cooking"),
    ]
    index = Bm25ChunkIndex.from_chunks(chunks)
    hits = index.search("broker fees", limit=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "1"


def test_bm25_punctuation_query_matches_corpus_terms():
    chunks = [_chunk("1", 0, "broker fees vary by account type")]
    index = Bm25ChunkIndex.from_chunks(chunks)
    hits = index.search("broker fees?", limit=3)
    assert len(hits) == 1
    assert hits[0].doc_id == "1"


def test_bm25_parenthetical_token_matches():
    chunks = [_chunk("1", 0, "employer 401(k) matching and vesting rules")]
    index = Bm25ChunkIndex.from_chunks(chunks)
    hits = index.search("401(k) vesting", limit=1)
    assert hits[0].doc_id == "1"


def test_bm25_search_respects_limit():
    chunks = [_chunk(str(i), 0, f"broker fees topic number {i}") for i in range(5)]
    index = Bm25ChunkIndex.from_chunks(chunks)
    assert len(index.search("broker fees", limit=2)) == 2


def test_save_and_load_round_trip(tmp_path: Path):
    chunks = [_chunk("42", 2, "annual percentage rate APR borrowing", title="Rates")]
    index = Bm25ChunkIndex.from_chunks(chunks)
    path = tmp_path / "fiqa_sample.pkl"
    index.save(path)

    loaded = Bm25ChunkIndex.load(path)
    hits = loaded.search("APR", limit=1)
    assert hits[0].doc_id == "42"
    assert hits[0].chunk_id == 2
    assert hits[0].doc_title == "Rates"
