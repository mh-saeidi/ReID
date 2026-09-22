"""Per-track recognition state.

This is bookkeeping for the *scheduler*, not a shortcut around recognition. The
distinction matters, and the rules are deliberately narrow:

* A cached embedding is never fed to the matcher in place of a face that is not
  currently visible. Doing that is exactly how a face-mode system starts
  asserting identities it cannot see.
* A cached *identity* survives only under the existing bounded temporal hold in
  :mod:`src.tracking.stabilizer`, which is frame-limited and auditable.
* When the scheduler does decide to re-evaluate, the full chain runs again:
  fresh face detection, fresh alignment, fresh embedding. Nothing is reused.

What the cache is for is deciding *when* to spend that work, and giving the
renderer and metadata something truthful to show in between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.schema import SchedulerState
from src.core.types import BBox, RecognitionStatus


@dataclass(slots=True)
class RecognitionCache:
    """Runtime recognition state for one track."""

    track_id: int = -1

    # --- scheduling ---------------------------------------------------------
    last_recognition_frame: int = -1_000_000
    last_recognition_time: float = 0.0
    scheduler_state: SchedulerState = SchedulerState.NEW_TRACK
    consecutive_unknown: int = 0
    """Drives the geometric backoff for genuinely unregistered people."""
    consecutive_no_face: int = 0
    force_next: bool = False
    force_reason: str = ""
    identity_changed: bool = False
    was_lost: bool = False

    # --- last observation ---------------------------------------------------
    last_status: RecognitionStatus = RecognitionStatus.PENDING
    last_identity_id: str | None = None
    last_similarity: float = 0.0
    last_face_bbox: BBox | None = None
    last_face_score: float = 0.0
    last_face_size: float = 0.0
    last_quality_reason: str = ""
    """Why the last face was rejected, when it was."""

    # --- embedding provenance ----------------------------------------------
    # The vector itself is deliberately NOT stored here: nothing downstream may
    # reuse it, so keeping it would only invite that mistake. Only the metadata
    # needed for diagnostics is retained.
    last_embedding_frame: int = -1_000_000
    last_embedding_dimension: int = 0
    embeddings_computed: int = 0

    # --- counters -----------------------------------------------------------
    recognitions_run: int = 0
    recognitions_skipped: int = 0
    quality_rejections: int = 0

    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- updates
    def note_skip(self) -> None:
        self.recognitions_skipped += 1

    def note_recognition(self, frame_index: int, timestamp: float) -> None:
        """A recognition pass is about to run for this track."""
        self.last_recognition_frame = frame_index
        self.last_recognition_time = timestamp
        self.recognitions_run += 1

    def note_embedding(self, frame_index: int, dimension: int) -> None:
        self.last_embedding_frame = frame_index
        self.last_embedding_dimension = dimension
        self.embeddings_computed += 1

    def note_quality_rejection(self, reason: str) -> None:
        self.quality_rejections += 1
        self.last_quality_reason = reason

    def note_result(self, status: RecognitionStatus, identity_id: str | None,
                    similarity: float, face: Any | None = None) -> None:
        """Record the outcome of a recognition pass."""
        previous_identity = self.last_identity_id
        self.last_status = status
        self.last_similarity = float(similarity)

        if status in (RecognitionStatus.UNKNOWN, RecognitionStatus.REJECTED):
            self.consecutive_unknown += 1
            self.consecutive_no_face = 0
            self.last_identity_id = None
        elif status is RecognitionStatus.NO_FACE:
            self.consecutive_no_face += 1
            # A hidden face is not evidence that the person is unregistered, so
            # it must not advance the unknown backoff.
        else:
            self.consecutive_unknown = 0
            self.consecutive_no_face = 0
            self.last_identity_id = identity_id

        if identity_id is not None and previous_identity is not None:
            if identity_id != previous_identity:
                self.identity_changed = True

        if face is not None:
            self.last_face_bbox = getattr(face, "bbox", None)
            self.last_face_score = float(getattr(face, "score", 0.0))
            self.last_face_size = float(getattr(face, "size", 0.0))

    def mark_lost(self) -> None:
        """The track was not observed this frame."""
        self.was_lost = True

    def force(self, reason: str) -> None:
        """Request a recognition pass on the next observation."""
        self.force_next = True
        self.force_reason = reason

    def reset_backoff(self, reason: str = "") -> None:
        """Clear the unknown backoff.

        Called when something changed that could plausibly change the answer --
        a gallery update, or a materially better face than last time.
        """
        self.consecutive_unknown = 0
        if reason:
            self.force(reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "scheduler_state": self.scheduler_state.value,
            "last_recognition_frame": self.last_recognition_frame,
            "last_status": self.last_status.value,
            "last_identity_id": self.last_identity_id,
            "last_similarity": round(self.last_similarity, 4),
            "last_face_bbox": (
                [round(v, 1) for v in self.last_face_bbox.to_list()]
                if self.last_face_bbox
                else None
            ),
            "last_face_score": round(self.last_face_score, 4),
            "last_face_size": round(self.last_face_size, 1),
            "consecutive_unknown": self.consecutive_unknown,
            "consecutive_no_face": self.consecutive_no_face,
            "recognitions_run": self.recognitions_run,
            "recognitions_skipped": self.recognitions_skipped,
            "embeddings_computed": self.embeddings_computed,
            "quality_rejections": self.quality_rejections,
            "last_quality_reason": self.last_quality_reason,
        }
