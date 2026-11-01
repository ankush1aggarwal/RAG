"""Structured JSON logging and stage timings (Week 7 — no OTel/Langfuse)."""

from app.observability.context import bind_request_id, current_request_id, reset_request_id
from app.observability.events import emit, stage_timer

__all__ = [
    "bind_request_id",
    "current_request_id",
    "emit",
    "reset_request_id",
    "stage_timer",
]


def configure_structured_logging(settings: object) -> None:
    """Lazy import avoids circular load with ``app.config.settings``."""
    from app.observability.logging import configure_structured_logging as _configure

    _configure(settings)
