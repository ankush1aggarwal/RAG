"""LLM token usage and optional cost estimates for observability."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from pydantic import BaseModel

_last_llm_usage: ContextVar[LlmUsage | None] = ContextVar("last_llm_usage", default=None)


class LlmUsage(BaseModel):
    """Token counts from the provider when exposed; cost is optional."""

    provider: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


def set_last_llm_usage(usage: LlmUsage | None) -> Token:
    return _last_llm_usage.set(usage)


def reset_last_llm_usage(token: Token) -> None:
    _last_llm_usage.reset(token)


def take_llm_usage() -> LlmUsage | None:
    """Return usage recorded for the latest LLM call in this context, then clear."""
    usage = _last_llm_usage.get()
    _last_llm_usage.set(None)
    return usage


def usage_from_ollama_chat_payload(data: dict[str, Any]) -> LlmUsage | None:
    if not isinstance(data, dict):
        return None
    prompt = _coerce_int(data.get("prompt_eval_count"))
    completion = _coerce_int(data.get("eval_count"))
    if prompt is None and completion is None:
        return None
    total = (prompt or 0) + (completion or 0) if prompt is not None or completion is not None else None
    if prompt is None and completion is not None:
        total = completion
    elif completion is None and prompt is not None:
        total = prompt
    return LlmUsage(
        provider="ollama",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def usage_from_cohere_chat_response(
    response: Any,
    *,
    input_usd_per_1m: float | None = None,
    output_usd_per_1m: float | None = None,
) -> LlmUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    billed = getattr(usage, "billed_units", None)
    prompt = _coerce_int(getattr(billed, "input_tokens", None) if billed else None)
    completion = _coerce_int(getattr(billed, "output_tokens", None) if billed else None)
    if prompt is None and completion is None:
        tokens = getattr(usage, "tokens", None)
        if tokens is not None:
            prompt = _coerce_int(getattr(tokens, "input_tokens", None))
            completion = _coerce_int(getattr(tokens, "output_tokens", None))
    if prompt is None and completion is None:
        return None
    total: int | None
    if prompt is not None and completion is not None:
        total = prompt + completion
    elif prompt is not None:
        total = prompt
    else:
        total = completion
    cost = estimate_chat_cost_usd(
        prompt,
        completion,
        input_usd_per_1m=input_usd_per_1m,
        output_usd_per_1m=output_usd_per_1m,
    )
    return LlmUsage(
        provider="cohere",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        estimated_cost_usd=cost,
    )


def estimate_chat_cost_usd(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    input_usd_per_1m: float | None,
    output_usd_per_1m: float | None,
) -> float | None:
    if input_usd_per_1m is None and output_usd_per_1m is None:
        return None
    cost = 0.0
    counted = False
    if prompt_tokens is not None and input_usd_per_1m is not None:
        cost += (prompt_tokens / 1_000_000.0) * input_usd_per_1m
        counted = True
    if completion_tokens is not None and output_usd_per_1m is not None:
        cost += (completion_tokens / 1_000_000.0) * output_usd_per_1m
        counted = True
    return round(cost, 8) if counted else None


def usage_log_fields(usage: LlmUsage | None) -> dict[str, Any]:
    """Flatten usage for ``emit`` (omit unset fields)."""
    if usage is None:
        return {}
    data = usage.model_dump(exclude_none=True)
    if not data:
        return {}
    # ``provider`` is metadata, not a secret.
    return data


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
