"""Face-specific data types.

Face recognition answers the identity question from facial appearance alone,
which is what makes it invariant to clothing. It also introduces a state that
whole-body ReID never has: *the face is not visible*. That is neither a match
nor a non-match, and it is represented explicitly rather than being collapsed
into "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.core.types import BBox


class FaceQuality(str, Enum):
    """Why a detected face was or was not usable for recognition."""

    OK = "ok"
    TOO_SMALL = "too_small"
    LOW_CONFIDENCE = "low_confidence"
    EXTREME_POSE = "extreme_pose"
    BLURRY = "blurry"


@dataclass(slots=True)
class FaceDetection:
    """One detected face with the landmarks needed for alignment.

    Args:
        bbox: Face box in absolute frame coordinates.
        score: Detector confidence.
        landmarks: ``(5, 2)`` array -- right eye, left eye, nose, right mouth
            corner, left mouth corner -- in absolute frame coordinates. These
            drive the similarity transform that normalises pose before
            embedding, which is where most of face recognition's accuracy
            comes from.
        quality: Why this face was accepted or rejected.
    """

    bbox: BBox
    score: float
    landmarks: np.ndarray | None = None
    quality: FaceQuality = FaceQuality.OK
    detail: str = ""

    @property
    def size(self) -> float:
        """Shorter side of the face box, the usual resolution proxy."""
        return min(self.bbox.width, self.bbox.height)

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) >= 5

    @property
    def eye_distance(self) -> float:
        """Inter-ocular distance -- a pose-robust measure of face resolution."""
        if not self.has_landmarks:
            return 0.0
        return float(np.linalg.norm(self.landmarks[1] - self.landmarks[0]))

    def offset(self, dx: float, dy: float) -> FaceDetection:
        """Shift into another coordinate frame (crop space -> frame space)."""
        landmarks = None
        if self.landmarks is not None:
            landmarks = self.landmarks + np.array([dx, dy], dtype=np.float32)
        return FaceDetection(
            bbox=BBox(
                self.bbox.x1 + dx, self.bbox.y1 + dy, self.bbox.x2 + dx, self.bbox.y2 + dy
            ),
            score=self.score,
            landmarks=landmarks,
            quality=self.quality,
            detail=self.detail,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bbox": [round(v, 2) for v in self.bbox.to_list()],
            "score": round(self.score, 4),
            "size": round(self.size, 1),
            "eye_distance": round(self.eye_distance, 1),
            "quality": self.quality.value,
        }


@dataclass(frozen=True, slots=True)
class FaceDetectorInfo:
    """What face detector was loaded."""

    name: str
    model_path: str
    backend: str
    device: str
    input_size: tuple[int, int]
    load_time_s: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
