"""Explicit, typed internal data model shared by every component.

Nothing in this project passes undocumented dictionaries around: every
boundary between the detector, the encoder, the gallery, the tracker and the
output layer speaks in terms of the structures defined here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Height / width. Full-body person crops are typically > 1.5."""
        return self.height / self.width if self.width > 0 else 0.0

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    def clip(self, width: int, height: int) -> BBox:
        return BBox(
            x1=float(min(max(self.x1, 0.0), width)),
            y1=float(min(max(self.y1, 0.0), height)),
            x2=float(min(max(self.x2, 0.0), width)),
            y2=float(min(max(self.y2, 0.0), height)),
        )

    def touches_border(self, width: int, height: int, tolerance: int = 2) -> bool:
        return (
            self.x1 <= tolerance
            or self.y1 <= tolerance
            or self.x2 >= width - tolerance
            or self.y2 >= height - tolerance
        )

    @classmethod
    def from_xyxy(cls, values: np.ndarray | list[float] | tuple[float, ...]) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in values[:4])
        return cls(x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# Detection / recognition
# --------------------------------------------------------------------------- #


class RecognitionStatus(str, Enum):
    """Outcome of comparing a query embedding against the identity gallery."""

    RECOGNIZED = "recognized"
    """Similarity >= recognition_threshold (and the margin rule passed)."""

    LOW_CONFIDENCE = "low_confidence"
    """Above the recognition threshold but below high_confidence_threshold."""

    UNKNOWN = "unknown"
    """Best similarity below the recognition threshold -- open-set rejection."""

    REJECTED = "rejected"
    """Above threshold but rejected by an explicit rule (e.g. ambiguous match)."""

    PENDING = "pending"
    """No ReID was run for this detection yet (skipped by reid_interval)."""

    NO_FACE = "no_face"
    """Face mode: the person was detected but their face is not usable.

    Deliberately distinct from UNKNOWN. "Unknown" means a face was seen and
    matched nobody registered; "no face" means there was nothing to compare.
    Collapsing the two would either invent rejections for people facing away,
    or tempt the system into quietly falling back to clothing.
    """


RECOGNIZED_STATUSES: frozenset[RecognitionStatus] = frozenset(
    {RecognitionStatus.RECOGNIZED, RecognitionStatus.LOW_CONFIDENCE}
)


@dataclass(slots=True)
class Detection:
    """A single person detected by the object detector in one frame."""

    bbox: BBox
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: int | None = None
    detection_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    face: Any | None = None
    """The FaceDetection used for recognition, in face mode (None otherwise)."""


@dataclass(slots=True)
class PersonIdentity:
    """A registered (enrolled) identity backed by one or more reference embeddings."""

    id: str
    name: str
    title: str = ""
    image_paths: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    embedding_dimension: int | None = None
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""
    quality: dict[str, Any] = field(default_factory=dict)
    source_model: str = ""

    @property
    def image_path(self) -> str:
        """Primary reference image (the one-shot enrollment image)."""
        return self.image_paths[0] if self.image_paths else ""

    @property
    def display_label(self) -> str:
        return f"{self.name} ({self.title})" if self.title else self.name


@dataclass(slots=True)
class RecognitionResult:
    """Result of matching one query embedding against the gallery."""

    status: RecognitionStatus
    identity_id: str | None = None
    identity_name: str | None = None
    identity_title: str | None = None
    similarity: float = 0.0
    runner_up_id: str | None = None
    runner_up_similarity: float = 0.0
    all_similarities: dict[str, float] = field(default_factory=dict)
    detail: str = ""
    """Human-readable reason, used for NO_FACE and REJECTED outcomes."""

    @property
    def is_recognized(self) -> bool:
        return self.status in RECOGNIZED_STATUSES

    @property
    def margin(self) -> float:
        """Gap between the best and second-best gallery identity."""
        return self.similarity - self.runner_up_similarity

    @classmethod
    def unknown(cls, unknown_label: str = "Unknown", **kwargs: Any) -> RecognitionResult:
        return cls(status=RecognitionStatus.UNKNOWN, identity_name=unknown_label, **kwargs)

    @classmethod
    def no_face(cls, label: str = "No face", detail: str = "") -> RecognitionResult:
        """A person with no usable face: neither a match nor a rejection."""
        return cls(status=RecognitionStatus.NO_FACE, identity_name=label, detail=detail)


@dataclass(slots=True)
class DetectionResult:
    """Detection + identity decision: the unit of output for the whole pipeline."""

    detection_id: str
    bbox: BBox
    detector_confidence: float
    source_id: str
    timestamp: float
    frame_index: int = 0
    track_id: int | None = None

    face: Any | None = None
    """The FaceDetection behind this decision, in face mode."""

    # Per-frame (instantaneous) recognition, before temporal stabilization.
    recognition: RecognitionResult = field(default_factory=RecognitionResult.unknown)
    # Stabilized recognition from the track's identity history (video only).
    stabilized: RecognitionResult | None = None

    @property
    def effective(self) -> RecognitionResult:
        """The recognition the UI/events should use: stabilized when available."""
        return self.stabilized if self.stabilized is not None else self.recognition

    @property
    def identity_id(self) -> str | None:
        return self.effective.identity_id

    @property
    def identity_name(self) -> str | None:
        return self.effective.identity_name

    @property
    def identity_title(self) -> str | None:
        return self.effective.identity_title

    @property
    def reid_similarity(self) -> float:
        return self.effective.similarity

    @property
    def recognition_status(self) -> RecognitionStatus:
        return self.effective.status

    @property
    def face_bbox(self) -> BBox | None:
        """Box of the face the identity came from, in face mode."""
        return getattr(self.face, "bbox", None) if self.face is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "track_id": self.track_id,
            "bbox": [round(v, 2) for v in self.bbox.to_list()],
            "detector_confidence": round(self.detector_confidence, 4),
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "identity_title": self.identity_title,
            "reid_similarity": round(self.reid_similarity, 4),
            "recognition_status": self.recognition_status.value,
            "recognition_detail": self.effective.detail,
            "face": self.face.to_dict() if self.face is not None else None,
            "instantaneous_status": self.recognition.status.value,
            "instantaneous_similarity": round(self.recognition.similarity, 4),
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
        }


@dataclass(slots=True)
class FrameResult:
    """Everything the pipeline produced for a single frame."""

    frame_index: int
    timestamp: float
    source_id: str
    width: int
    height: int
    detections: list[DetectionResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def recognized(self) -> list[DetectionResult]:
        return [d for d in self.detections if d.recognition_status in RECOGNIZED_STATUSES]

    @property
    def unknown(self) -> list[DetectionResult]:
        return [d for d in self.detections if d.recognition_status not in RECOGNIZED_STATUSES]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "width": self.width,
            "height": self.height,
            "num_detections": len(self.detections),
            "num_recognized": len(self.recognized),
            "num_unknown": len(self.unknown),
            "timings_ms": {k: round(v * 1000.0, 3) for k, v in self.timings.items()},
            "detections": [d.to_dict() for d in self.detections],
        }


# --------------------------------------------------------------------------- #
# Frames / sources
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Frame:
    """One image pulled from an input source."""

    image: np.ndarray
    index: int
    timestamp: float = field(default_factory=time.time)
    source_id: str = "unknown"
    path: str | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Static metadata describing an input source."""

    source_id: str
    kind: str
    width: int
    height: int
    fps: float
    frame_count: int | None = None
    is_stream: bool = False
