"""Redact secrets and limit sensitive text in log fields."""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|credential)",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"

# Never log raw user questions or LLM prompts by default.
_NEVER_LOG_KEYS = frozenset(
    {
        "question",
        "prompt",
        "prompt_user",
        "prompt_system",
        "chunk_text",
        "answer",
        "messages",
    }
)
# Token *counts* from providers (not secrets).
_SAFE_METRIC_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_cost_usd",
        "provider",
    }
)


def sanitize_log_value(key: str, value: Any, *, max_text_len: int) -> Any:
    key_lower = key.lower()
    if key_lower not in _SAFE_METRIC_KEYS and _SENSITIVE_KEY_RE.search(key):
        return _REDACTED
    if key_lower in _NEVER_LOG_KEYS:
        if isinstance(value, str):
            return f"[omitted len={len(value)}]"
        return _REDACTED
    if isinstance(value, str) and len(value) > max_text_len:
        return value[:max_text_len] + "…"
    if isinstance(value, dict):
        return {k: sanitize_log_value(str(k), v, max_text_len=max_text_len) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_log_value(key, item, max_text_len=max_text_len) for item in value[:20]]
    return value


def sanitize_fields(fields: dict[str, Any], *, max_text_len: int) -> dict[str, Any]:
    return {k: sanitize_log_value(k, v, max_text_len=max_text_len) for k, v in fields.items()}
