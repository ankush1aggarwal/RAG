"""Structured log events and stage timings."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.observability.context import current_request_id
from app.observability.redaction import sanitize_fields

_logger = logging.getLogger("fiqa.rag")
_DEFAULT_MAX_FIELD_CHARS = 120


def _max_field_chars() -> int:
    from app.config.settings import get_settings

    return get_settings().log_max_field_chars


def emit(event: str, level: int = logging.INFO, **fields: Any) -> None:
    safe = sanitize_fields(fields, max_text_len=_max_field_chars())
    payload: dict[str, Any] = {"event": event, **safe}
    request_id = current_request_id()
    if request_id:
        payload["request_id"] = request_id
    _logger.log(level, event, extra={"structured": payload})


@contextmanager
def stage_timer(stage: str, **fields: Any) -> Iterator[None]:
    """Log ``stage_start`` / ``stage_end`` with ``duration_ms`` (or ``stage_error``)."""
    start = time.perf_counter()
    emit("stage_start", stage=stage, **fields)
    try:
        yield
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        emit(
            "stage_error",
            level=logging.ERROR,
            stage=stage,
            duration_ms=round(elapsed_ms, 2),
            error_type=type(exc).__name__,
            **fields,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        emit("stage_end", stage=stage, duration_ms=round(elapsed_ms, 2), **fields)
