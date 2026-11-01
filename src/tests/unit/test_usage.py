"""Tests for LLM usage parsing and cost helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.generation.usage import (
    estimate_chat_cost_usd,
    usage_from_cohere_chat_response,
    usage_from_ollama_chat_payload,
    usage_log_fields,
)


def test_usage_from_ollama_chat_payload():
    usage = usage_from_ollama_chat_payload(
        {"prompt_eval_count": 120, "eval_count": 45, "message": {"content": "x"}}
    )
    assert usage is not None
    assert usage.provider == "ollama"
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 45
    assert usage.total_tokens == 165


def test_usage_from_ollama_missing_counts_returns_none():
    assert usage_from_ollama_chat_payload({"message": {"content": "x"}}) is None


def test_usage_from_cohere_chat_response_billed_units():
    billed = MagicMock(input_tokens=1000, output_tokens=250)
    usage_obj = MagicMock(billed_units=billed, tokens=None)
    response = MagicMock(usage=usage_obj)

    usage = usage_from_cohere_chat_response(
        response,
        input_usd_per_1m=3.0,
        output_usd_per_1m=15.0,
    )
    assert usage is not None
    assert usage.provider == "cohere"
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 250
    assert usage.total_tokens == 1250
    assert usage.estimated_cost_usd == estimate_chat_cost_usd(
        1000, 250, input_usd_per_1m=3.0, output_usd_per_1m=15.0
    )


def test_usage_log_fields_omits_none():
    from app.generation.usage import LlmUsage

    fields = usage_log_fields(LlmUsage(provider="ollama", prompt_tokens=1))
    assert fields == {"provider": "ollama", "prompt_tokens": 1}
