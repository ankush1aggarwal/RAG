"""JSON log lines to stderr (stdlib only)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config.settings import Settings

_CONFIGURED = False


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_structured_logging(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if not settings.structured_logging_enabled:
        logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
        _CONFIGURED = True
        return

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


def reset_structured_logging_config() -> None:
    """Test helper: allow re-running ``configure_structured_logging``."""
    global _CONFIGURED
    _CONFIGURED = False
