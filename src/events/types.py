"""Event vocabulary.

A generic event layer keeps integrations (webhooks, message queues, a UI, an
access-control system) out of the processing code: subscribers attach to the
manager instead of the pipeline growing new responsibilities.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    PERSON_DETECTED = "person_detected"
    PERSON_RECOGNIZED = "person_recognized"
    UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
    PERSON_LOST = "person_lost"
    IDENTITY_CHANGED = "identity_changed"
    TRACK_STARTED = "track_started"
    TRACK_ENDED = "track_ended"
    SNAPSHOT_SAVED = "snapshot_saved"
    VIDEO_RECORDING_STARTED = "video_recording_started"
    VIDEO_RECORDING_STOPPED = "video_recording_stopped"
    SOURCE_STARTED = "source_started"
    SOURCE_ENDED = "source_ended"
    GALLERY_UPDATED = "gallery_updated"
    ERROR = "error"


@dataclass(slots=True)
class Event:
    """One thing that happened, with enough context to act on it."""

    type: EventType
    source_id: str
    timestamp: float = field(default_factory=time.time)
    frame_index: int = 0
    track_id: int | None = None
    identity_id: str | None = None
    identity_name: str | None = None
    identity_title: str | None = None
    similarity: float | None = None
    bbox: list[float] | None = None
    detector_confidence: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event_id": self.event_id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "source_id": self.source_id,
            "frame_index": self.frame_index,
        }
        for key in (
            "track_id",
            "identity_id",
            "identity_name",
            "identity_title",
            "detector_confidence",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.similarity is not None:
            data["similarity"] = round(self.similarity, 4)
        if self.bbox is not None:
            data["bbox"] = [round(v, 2) for v in self.bbox]
        if self.payload:
            data["payload"] = self.payload
        return data
