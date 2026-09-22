"""Tracker interface and per-track identity state.

A track id answers "is this the same moving object as last frame". It is
temporary and source-local, and is deliberately kept separate from the
registered identity, which only ever comes from a gallery match.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.types import Detection, RecognitionResult, RecognitionStatus
from src.tracking.recognition_cache import RecognitionCache
from src.tracking.stabilizer import IdentityHistory


@dataclass(slots=True)
class TrackState:
    """Everything the pipeline remembers about one tracked person."""

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float
    last_seen_time: float

    frames_seen: int = 0
    frames_recognized: int = 0
    frames_unknown: int = 0
    frames_no_face: int = 0
    """Frames where the person was tracked but no usable face was visible."""
    reid_runs: int = 0

    identity: IdentityHistory = field(default_factory=IdentityHistory)
    recognition_cache: RecognitionCache = field(default_factory=RecognitionCache)
    """Scheduler bookkeeping. Holds no embedding: see recognition_cache.py."""
    last_embedding: np.ndarray | None = None
    """Only used in person_reid mode, where a transiently bad person box may
    legitimately reuse the previous crop's embedding. Never used in face mode."""
    last_reid_frame: int = -1_000_000
    last_result: RecognitionResult | None = None
    last_bbox: Any = None
    snapshot_taken_at: float = 0.0
    events_emitted: set[str] = field(default_factory=set)

    @property
    def identity_id(self) -> str | None:
        return self.identity.current_identity

    @property
    def identity_similarity(self) -> float:
        return self.identity.current_similarity

    @property
    def recognition_state(self) -> RecognitionStatus:
        if self.last_result is None:
            return RecognitionStatus.PENDING
        return self.last_result.status

    @property
    def age_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "frames_seen": self.frames_seen,
            "frames_recognized": self.frames_recognized,
            "frames_unknown": self.frames_unknown,
            "frames_no_face": self.frames_no_face,
            "reid_runs": self.reid_runs,
            "identity_id": self.identity_id,
            "identity_similarity": round(self.identity_similarity, 4),
            "recognition_state": self.recognition_state.value,
            "identity_switches": self.identity.switches,
            "recognition_cache": self.recognition_cache.to_dict(),
            "history": [
                {
                    "frame": o.frame_index,
                    "identity": o.identity_id,
                    "similarity": round(o.similarity, 4),
                }
                for o in self.identity.history
            ],
        }


class Tracker(ABC):
    """Assigns temporally stable ids to detections."""

    @abstractmethod
    def update(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Return detections annotated with track ids."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all tracker state."""

    @property
    @abstractmethod
    def name(self) -> str: ...
