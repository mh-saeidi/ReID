"""Event dispatch and persistence."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path

from src.core.exceptions import OutputError
from src.events.types import Event, EventType
from src.utils.logging import get_logger

logger = get_logger(__name__)

EventHandler = Callable[[Event], None]


class EventManager:
    """Publishes events to subscribers and, optionally, to a JSONL file.

    Subscriber failures are logged and swallowed: a broken integration must not
    take down video processing.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        log_path: Path | None = None,
        history_size: int = 1000,
        console: bool = False,
    ) -> None:
        self._enabled = enabled
        self._log_path = log_path
        self._console = console
        self._handlers: dict[EventType | None, list[EventHandler]] = {}
        self._history: deque[Event] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._file = None

        if enabled and log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._file = log_path.open("a", encoding="utf-8")
            except OSError as exc:
                raise OutputError(
                    f"cannot open the event log {log_path}: {exc}. "
                    "Check output.directory / events.directory permissions."
                ) from exc

    @property
    def enabled(self) -> bool:
        return self._enabled

    def subscribe(self, handler: EventHandler, event_type: EventType | None = None) -> None:
        """Register a handler for one event type, or for all when ``None``."""
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, handler: EventHandler, event_type: EventType | None = None) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: Event) -> Event:
        """Record and dispatch an event."""
        if not self._enabled:
            return event
        with self._lock:
            self._history.append(event)
            if self._file is not None:
                try:
                    self._file.write(json.dumps(event.to_dict(), default=str) + "\n")
                    self._file.flush()
                except OSError as exc:  # pragma: no cover - disk-full territory
                    logger.error("Cannot write the event log: %s", exc)

        if self._console:
            logger.info("Event", extra={"event": event.type.value, **_console_fields(event)})
        else:
            logger.debug("Event", extra={"event": event.type.value, **_console_fields(event)})

        for handler in (*self._handlers.get(event.type, ()), *self._handlers.get(None, ())):
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - isolate subscriber failures
                logger.error(
                    "Event handler failed",
                    extra={"event": event.type.value, "error": str(exc)},
                )
        return event

    def history(self, limit: int | None = None, event_type: EventType | None = None) -> list[Event]:
        events: Iterable[Event] = list(self._history)
        if event_type is not None:
            events = [e for e in events if e.type is event_type]
        events = list(events)
        return events[-limit:] if limit else events

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def __enter__(self) -> EventManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _console_fields(event: Event) -> dict[str, object]:
    fields: dict[str, object] = {"source": event.source_id, "frame": event.frame_index}
    if event.track_id is not None:
        fields["track_id"] = event.track_id
    if event.identity_name:
        fields["identity"] = event.identity_name
    if event.similarity is not None:
        fields["similarity"] = round(event.similarity, 3)
    return fields
