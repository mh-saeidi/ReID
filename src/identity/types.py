"""Identity data model.

The central rule of this module: the quantities below are *different things*
and are never collapsed into one "confidence". They answer different questions,
they live on different scales, and only one of them is a probability.

``detector_confidence``       YOLO26's belief that a person is present.
``face_detection_confidence`` the face detector's belief that a face is present.
``face_quality``              how reliable this face's embedding will be, [0, 1].
``face_similarity``           raw cosine between two L2-normalised embeddings.
                              NOT a probability and NOT a percentage.
``track_id``                  a temporary, source-local object handle.
``identity_id``               a registered person.
``identity_confidence``       a calibrated probability that the identity claim
                              is correct -- and only populated when calibration
                              data exists to support that reading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.face.quality import FaceQualityScore
from src.face.visibility import FaceVisibility


class IdentityStatus(str, Enum):
    """The outcome of an identity decision."""

    RECOGNIZED = "recognized"
    """Enough evidence to name this person."""

    UNCERTAIN = "uncertain"
    """A candidate leads, but the evidence does not yet support naming them.
    Distinct from UNKNOWN: this says "not yet", not "nobody"."""

    UNKNOWN = "unknown"
    """Compared against the gallery and matched nobody. Open-set rejection."""

    TEMPORARILY_MAINTAINED = "temporarily_maintained"
    """No usable face this frame, but the track was already confirmed and the
    identity is being carried within its bounded budget."""

    NO_FACE = "no_face"
    """A person with nothing to recognise from. Not a rejection."""

    PENDING = "pending"
    """No recognition has been attempted for this detection yet."""

    @property
    def is_named(self) -> bool:
        return self in (IdentityStatus.RECOGNIZED, IdentityStatus.TEMPORARILY_MAINTAINED)


class TrackIdentityPhase(str, Enum):
    """Where a track sits in its identity lifecycle.

    Separate from :class:`IdentityStatus`, which describes a single decision.
    This describes the track's history.
    """

    NEW_TRACK = "new_track"
    UNCONFIRMED = "unconfirmed"
    RECOGNIZED = "recognized"
    TEMPORARILY_OCCLUDED = "temporarily_occluded"
    UNKNOWN = "unknown"
    LOST = "lost"


class FailureReason(str, Enum):
    """Which layer failed, when the system could not name someone.

    Returning a bare "Unknown" for every failure destroys the information
    needed to fix the system: a person missed by the detector and a person
    whose face was too small require completely different remedies.
    """

    # Person detection
    PERSON_NOT_DETECTED = "person_not_detected"
    LOW_PERSON_DETECTION_CONFIDENCE = "low_person_detection_confidence"

    # Face stage
    FACE_NOT_FOUND = "face_not_found"
    FACE_TOO_SMALL = "face_too_small"
    FACE_TOO_BLURRY = "face_too_blurry"
    FACE_OCCLUDED = "face_occluded"
    FACE_POSE_TOO_EXTREME = "face_pose_too_extreme"
    FACE_LOW_QUALITY = "face_low_quality"
    FACE_ALIGNMENT_FAILED = "face_alignment_failed"
    FACE_EMBEDDING_FAILED = "face_embedding_failed"

    # Recognition
    NO_IDENTITY_MATCH = "no_identity_match"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    LOW_SIMILARITY = "low_similarity"
    LOW_CALIBRATED_CONFIDENCE = "low_calibrated_confidence"
    EMPTY_GALLERY = "empty_gallery"

    # Tracking
    TRACK_LOST = "track_lost"
    TRACK_SWITCH = "track_switch"
    TRACK_TIMEOUT = "track_timeout"

    # Scheduling
    NOT_EVALUATED_THIS_FRAME = "not_evaluated_this_frame"

    @property
    def layer(self) -> str:
        name = self.value
        if name.startswith("person") or name.startswith("low_person"):
            return "person_detection"
        if name.startswith("face"):
            return "face"
        if name.startswith("track"):
            return "tracking"
        if name == "not_evaluated_this_frame":
            return "scheduling"
        return "recognition"


@dataclass(slots=True)
class FaceMatchResult:
    """One query embedding compared against the gallery.

    Carries the raw similarity and the candidate ranking. It deliberately does
    NOT carry a confidence: turning a similarity into a probability requires
    calibration data, which lives in a later stage.
    """

    best_identity_id: str | None = None
    best_similarity: float = 0.0
    runner_up_id: str | None = None
    runner_up_similarity: float = 0.0
    matched_embedding_key: str = ""
    """Which stored embedding matched -- 'reference', 'live_003', ..."""
    ranked: list[tuple[str, float]] = field(default_factory=list)
    gallery_size: int = 0

    @property
    def margin(self) -> float:
        """Gap to the next best identity. A small margin means ambiguity."""
        return self.best_similarity - self.runner_up_similarity

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_identity_id": self.best_identity_id,
            "best_similarity": round(self.best_similarity, 6),
            "runner_up_id": self.runner_up_id,
            "runner_up_similarity": round(self.runner_up_similarity, 6),
            "margin": round(self.margin, 6),
            "matched_embedding": self.matched_embedding_key,
            "gallery_size": self.gallery_size,
            "top_k": [(i, round(s, 6)) for i, s in self.ranked[:5]],
        }


@dataclass(slots=True)
class IdentityObservation:
    """A single frame's worth of identity evidence for one track."""

    frame_index: int
    timestamp: float
    identity_id: str | None
    similarity: float
    quality: float
    visibility: FaceVisibility
    margin: float = 0.0

    @property
    def weight(self) -> float:
        """How much this observation should count.

        Quality-weighted: a clean frontal face at close range is worth far more
        than a small, blurred, half-turned one, and weighting by quality is what
        stops a run of poor frames outvoting a good one.
        """
        return float(np.clip(self.quality, 0.0, 1.0))


@dataclass(slots=True)
class IdentityEvidence:
    """Accumulated evidence for one track, across frames."""

    track_id: int
    observations: list[IdentityObservation] = field(default_factory=list)
    scores_by_identity: dict[str, float] = field(default_factory=dict)
    weights_by_identity: dict[str, float] = field(default_factory=dict)
    confirmations: int = 0
    unknown_count: int = 0
    no_face_count: int = 0

    def weighted_score(self, identity_id: str) -> float:
        """Quality-weighted mean similarity for one candidate."""
        weight = self.weights_by_identity.get(identity_id, 0.0)
        if weight <= 0.0:
            return 0.0
        return self.scores_by_identity.get(identity_id, 0.0) / weight

    def ranked_candidates(self) -> list[tuple[str, float]]:
        return sorted(
            ((i, self.weighted_score(i)) for i in self.scores_by_identity),
            key=lambda pair: -pair[1],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "observations": len(self.observations),
            "confirmations": self.confirmations,
            "unknown_count": self.unknown_count,
            "no_face_count": self.no_face_count,
            "candidates": [
                {"identity_id": i, "weighted_similarity": round(s, 6)}
                for i, s in self.ranked_candidates()[:5]
            ],
        }


@dataclass(slots=True)
class IdentityDecision:
    """The system's answer for one detection, with everything behind it.

    Every quantity is kept separate and named for what it is.
    """

    status: IdentityStatus
    identity_id: str | None = None
    name: str | None = None
    title: str | None = None

    # --- separate, non-interchangeable quantities --------------------------
    detector_confidence: float = 0.0
    face_detection_confidence: float = 0.0
    face_similarity: float = 0.0
    face_quality: float = 0.0
    identity_confidence: float | None = None
    """Calibrated probability the identity claim is correct. ``None`` means no
    calibration exists, and the caller must not invent one from similarity."""

    track_id: int | None = None
    track_stability: float = 0.0
    phase: TrackIdentityPhase = TrackIdentityPhase.NEW_TRACK
    visibility: FaceVisibility = FaceVisibility.NO_USABLE_FACE
    threshold_used: float = 0.0
    threshold_family: str = "full"

    failure: FailureReason | None = None
    reason: str = ""
    quality_detail: FaceQualityScore | None = None
    match: FaceMatchResult | None = None
    evidence: IdentityEvidence | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_named(self) -> bool:
        return self.status.is_named and self.identity_id is not None

    @property
    def display_label(self) -> str:
        if not self.is_named:
            return {
                IdentityStatus.UNKNOWN: "Unknown",
                IdentityStatus.UNCERTAIN: "Uncertain",
                IdentityStatus.NO_FACE: "No face",
                IdentityStatus.PENDING: "",
            }.get(self.status, "Unknown")
        return self.name or self.identity_id or "Unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "identity_id": self.identity_id,
            "name": self.name,
            "title": self.title,
            "track_id": self.track_id,
            "phase": self.phase.value,
            "visibility": self.visibility.value,
            # Deliberately separate keys; never merged into one "confidence".
            "detector_confidence": round(self.detector_confidence, 4),
            "face_detection_confidence": round(self.face_detection_confidence, 4),
            "face_similarity": round(self.face_similarity, 6),
            "face_quality": round(self.face_quality, 4),
            "identity_confidence": (
                round(self.identity_confidence, 4)
                if self.identity_confidence is not None
                else None
            ),
            "track_stability": round(self.track_stability, 4),
            "threshold_used": round(self.threshold_used, 4),
            "threshold_family": self.threshold_family,
            "failure": self.failure.value if self.failure else None,
            "failure_layer": self.failure.layer if self.failure else None,
            "reason": self.reason,
            "quality": self.quality_detail.to_dict() if self.quality_detail else None,
            "match": self.match.to_dict() if self.match else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }

    @classmethod
    def no_face(cls, reason: str = "", failure: FailureReason | None = None,
                **kwargs: Any) -> IdentityDecision:
        return cls(
            status=IdentityStatus.NO_FACE,
            failure=failure or FailureReason.FACE_NOT_FOUND,
            reason=reason or "no usable face in this person region",
            **kwargs,
        )

    @classmethod
    def unknown(cls, reason: str = "", failure: FailureReason | None = None,
                **kwargs: Any) -> IdentityDecision:
        return cls(
            status=IdentityStatus.UNKNOWN,
            failure=failure or FailureReason.NO_IDENTITY_MATCH,
            reason=reason or "face compared against the gallery and matched nobody",
            **kwargs,
        )
