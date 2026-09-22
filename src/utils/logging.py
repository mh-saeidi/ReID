"""Structured logging setup.

INFO stays operator-readable (model loads, device selection, gallery summary,
source start/stop). Per-detection and per-match detail is DEBUG only, and raw
embedding values are never logged -- they are biometric-like data.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False

_RESET = "\033[0m"
_COLORS = {
    "DEBUG": "\033[38;5;245m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;196m",
}

_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
    "taskName",
}


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class KeyValueFormatter(logging.Formatter):
    """Human-readable formatter that appends structured ``extra`` fields."""

    def __init__(self, *, color: bool = False) -> None:
        super().__init__("%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = _extras(record)
        if extras:
            base += " | " + " ".join(f"{k}={_render(v)}" for k, v in sorted(extras.items()))
        if self.color:
            prefix = _COLORS.get(record.levelname, "")
            if prefix:
                base = f"{prefix}{base}{_RESET}"
        return base


class JsonFormatter(logging.Formatter):
    """One JSON object per line -- for shipping logs to a collector."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


def setup_logging(
    level: str = "INFO",
    *,
    log_file: str | Path | None = None,
    json_logs: bool = False,
    color: bool | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging once per process."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)

    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric)

    use_color = sys.stderr.isatty() if color is None else color
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter() if json_logs else KeyValueFormatter(color=use_color))
    root.addHandler(stream)

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter() if json_logs else KeyValueFormatter(color=False))
        root.addHandler(file_handler)

    # Ultralytics is chatty at INFO; keep it aligned with our own verbosity.
    logging.getLogger("ultralytics").setLevel(max(numeric, logging.WARNING))
    _CONFIGURED = True
    return root


# Attribute names ``logging.LogRecord`` owns. Passing any of them through
# ``extra=`` raises KeyError, which would turn a log line into a crash -- so the
# adapter below renames them instead of letting that happen at runtime.
_RESERVED_EXTRA: frozenset[str] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)


class StructuredAdapter(logging.LoggerAdapter):
    """Logger that accepts any ``extra`` key without colliding with LogRecord."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"{key}_" if key in _RESERVED_EXTRA else key): value
                for key, value in extra.items()
            }
        return msg, kwargs

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802 - logging API
        return self.logger.isEnabledFor(level)


def get_logger(name: str) -> StructuredAdapter:
    """Return a module logger (``src.reid.yolo26_reid`` style names)."""
    return StructuredAdapter(logging.getLogger(name), {})
