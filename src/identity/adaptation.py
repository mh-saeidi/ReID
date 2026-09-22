"""Conservative online gallery adaptation.

A passport photograph is one view of a person. Adding a handful of verified
live observations lets the gallery cover the lighting, camera and appearance
the person is actually seen under -- which is the single largest source of
improvement available without collecting more enrollment photos.

It is also the single largest way to destroy the system. The failure runs:

    unknown person -> wrongly matched to John -> saved into John's gallery
    -> future strangers now match John more easily -> more contamination

Once that starts it is self-reinforcing and invisible. So adaptation is gated
hard, and every gate exists to block a specific route into that failure:

``min_similarity``        the observation must be a strong match, not a marginal one
``min_quality``           a degraded face must never define an identity
``min_confirmation``      the track must have been confirmed repeatedly
``min_margin``            no competing identity may be scoring comparably
``require_full_face``     a masked or partial face is not a reference
``min_track_stability``   the track must not have been switching identities
``novelty band``          the observation must differ from what is already stored,
                          but not so much that it is probably someone else
``cooldown``              one track cannot flood a gallery in a few seconds

Every accepted and rejected decision is logged with its reason, so the gallery's
provenance can be audited after the fact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.face.quality import FaceQualityScore
from src.face.visibility import FaceVisibility
from src.identity.face_gallery import FaceGallery
from src.identity.state_machine import TrackIdentityState
from src.identity.types import FaceMatchResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AdaptationOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_DISABLED = "rejected_disabled"
    REJECTED_LOW_SIMILARITY = "rejected_low_similarity"
    REJECTED_LOW_QUALITY = "rejected_low_quality"
    REJECTED_UNCONFIRMED = "rejected_unconfirmed"
    REJECTED_AMBIGUOUS = "rejected_ambiguous"
    REJECTED_NOT_FULL_FACE = "rejected_not_full_face"
    REJECTED_UNSTABLE_TRACK = "rejected_unstable_track"
    REJECTED_REDUNDANT = "rejected_redundant"
    REJECTED_TOO_NOVEL = "rejected_too_novel"
    REJECTED_COOLDOWN = "rejected_cooldown"
    REJECTED_CAPACITY = "rejected_capacity"


@dataclass(slots=True)
class AdaptationConfig:
    """Gating for online gallery updates.

    The defaults are deliberately strict. Adaptation that never fires costs
    nothing; adaptation that fires wrongly is permanent.
    """

    enabled: bool = False
    """Off by default. It should be switched on deliberately, after the
    thresholds below have been set from this deployment's own calibration."""

    min_similarity: float | None = None
    """``None`` means "the calibrated threshold plus ``similarity_headroom``",
    which keeps this tied to measured data rather than a guessed constant."""
    similarity_headroom: float = 0.15
    min_quality: float = 0.70
    min_confirmation_frames: int = 5
    min_margin: float = 0.15
    require_full_face: bool = True
    min_track_stability: float = 0.60
    max_samples_per_identity: int = 20

    min_novelty: float = 0.02
    """Reject observations nearly identical to a stored one: they add nothing
    but consume a slot."""
    max_novelty: float = 0.45
    """Reject observations very unlike everything stored. A genuinely new view
    is valuable; something this different is more likely a different person."""

    cooldown_seconds: float = 20.0
    max_per_track: int = 3


@dataclass(slots=True)
class AdaptationRecord:
    """One adaptation decision, accepted or not."""

    outcome: AdaptationOutcome
    identity_id: str
    track_id: int
    reason: str
    similarity: float = 0.0
    quality: float = 0.0
    margin: float = 0.0
    novelty: float = 0.0
    embedding_key: str = ""
    timestamp: float = field(default_factory=time.time)
    frame_index: int = -1

    @property
    def accepted(self) -> bool:
        return self.outcome is AdaptationOutcome.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "accepted": self.accepted,
            "identity_id": self.identity_id,
            "track_id": self.track_id,
            "embedding_key": self.embedding_key,
            "similarity": round(self.similarity, 6),
            "quality": round(self.quality, 4),
            "margin": round(self.margin, 6),
            "novelty": round(self.novelty, 6),
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


class GalleryAdapter:
    """Decides whether a live observation may join the gallery."""

    def __init__(self, gallery: FaceGallery, config: AdaptationConfig) -> None:
        self._gallery = gallery
        self._config = config
        self._last_added: dict[str, float] = {}
        self._per_track: dict[int, int] = {}
        self.history: list[AdaptationRecord] = []

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def required_similarity(self, calibrated_threshold: float | None) -> float:
        """The similarity an observation must clear to be stored.

        Tied to the calibrated acceptance threshold plus headroom, so it moves
        with the deployment's measured operating point instead of sitting at a
        number chosen in advance.
        """
        if self._config.min_similarity is not None:
            return self._config.min_similarity
        base = calibrated_threshold if calibrated_threshold is not None else 0.45
        return min(0.95, base + self._config.similarity_headroom)

    def consider(
        self,
        *,
        embedding: np.ndarray,
        match: FaceMatchResult,
        quality: FaceQualityScore,
        visibility: FaceVisibility,
        track_state: TrackIdentityState,
        calibrated_threshold: float | None = None,
        frame_index: int = -1,
    ) -> AdaptationRecord:
        """Evaluate one observation against every gate."""
        identity_id = track_state.current_identity or match.best_identity_id or ""
        track_id = track_state.track_id

        def reject(outcome: AdaptationOutcome, reason: str, novelty: float = 0.0):
            record = AdaptationRecord(
                outcome=outcome, identity_id=identity_id, track_id=track_id,
                reason=reason, similarity=match.best_similarity,
                quality=quality.overall, margin=match.margin, novelty=novelty,
                frame_index=frame_index,
            )
            self.history.append(record)
            return record

        if not self._config.enabled:
            return reject(AdaptationOutcome.REJECTED_DISABLED, "adaptation is disabled")
        if not identity_id:
            return reject(
                AdaptationOutcome.REJECTED_UNCONFIRMED, "no confirmed identity"
            )

        identity = self._gallery.get(identity_id)
        if identity is None:
            return reject(
                AdaptationOutcome.REJECTED_UNCONFIRMED,
                f"'{identity_id}' is not in the gallery",
            )

        # --- the track must have earned this identity ----------------------
        if track_state.confirmation_count < self._config.min_confirmation_frames:
            return reject(
                AdaptationOutcome.REJECTED_UNCONFIRMED,
                f"only {track_state.confirmation_count} confirming frame(s); "
                f"{self._config.min_confirmation_frames} required",
            )
        if track_state.track_stability < self._config.min_track_stability:
            return reject(
                AdaptationOutcome.REJECTED_UNSTABLE_TRACK,
                f"track stability {track_state.track_stability:.2f} is below "
                f"{self._config.min_track_stability:.2f}",
            )

        # --- the observation itself must be strong and unambiguous ---------
        required = self.required_similarity(calibrated_threshold)
        if match.best_similarity < required:
            return reject(
                AdaptationOutcome.REJECTED_LOW_SIMILARITY,
                f"similarity {match.best_similarity:.3f} is below the "
                f"adaptation floor {required:.3f}",
            )
        if quality.overall < self._config.min_quality:
            return reject(
                AdaptationOutcome.REJECTED_LOW_QUALITY,
                f"face quality {quality.overall:.2f} is below "
                f"{self._config.min_quality:.2f} "
                f"(weakest: {quality.weakest_component})",
            )
        if match.runner_up_id is not None and match.margin < self._config.min_margin:
            return reject(
                AdaptationOutcome.REJECTED_AMBIGUOUS,
                f"'{match.runner_up_id}' is within {match.margin:.3f}; a "
                "contested observation must never be stored",
            )
        if self._config.require_full_face and visibility is not FaceVisibility.FULL_FACE:
            return reject(
                AdaptationOutcome.REJECTED_NOT_FULL_FACE,
                f"visibility is {visibility.value}; only unobstructed faces "
                "may extend an identity",
            )

        # --- rate limiting -------------------------------------------------
        now = time.time()
        last = self._last_added.get(identity_id)
        if last is not None and (now - last) < self._config.cooldown_seconds:
            return reject(
                AdaptationOutcome.REJECTED_COOLDOWN,
                f"only {now - last:.0f}s since the last addition for this person",
            )
        if self._per_track.get(track_id, 0) >= self._config.max_per_track:
            return reject(
                AdaptationOutcome.REJECTED_CAPACITY,
                f"track {track_id} has already contributed "
                f"{self._config.max_per_track} sample(s)",
            )

        # --- novelty band --------------------------------------------------
        novelty = self._novelty(identity_id, embedding)
        if novelty < self._config.min_novelty:
            return reject(
                AdaptationOutcome.REJECTED_REDUNDANT,
                f"novelty {novelty:.3f} is below {self._config.min_novelty:.3f}; "
                "this view is already represented",
                novelty,
            )
        if novelty > self._config.max_novelty:
            return reject(
                AdaptationOutcome.REJECTED_TOO_NOVEL,
                f"novelty {novelty:.3f} exceeds {self._config.max_novelty:.3f}; "
                "too unlike the stored views to trust as the same person",
                novelty,
            )

        record = self._gallery.add_live_embedding(
            identity_id,
            embedding,
            quality=quality.overall,
            similarity=match.best_similarity,
            frame_index=frame_index,
            track_id=track_id,
            notes=(
                f"adapted from track {track_id} after "
                f"{track_state.confirmation_count} confirmations"
            ),
        )
        if record is None:
            return reject(
                AdaptationOutcome.REJECTED_CAPACITY,
                "the identity is at capacity and this sample is not better "
                "than the weakest stored one",
                novelty,
            )

        self._last_added[identity_id] = now
        self._per_track[track_id] = self._per_track.get(track_id, 0) + 1
        accepted = AdaptationRecord(
            outcome=AdaptationOutcome.ACCEPTED,
            identity_id=identity_id,
            track_id=track_id,
            reason=(
                f"verified observation: similarity {match.best_similarity:.3f} "
                f">= {required:.3f}, quality {quality.overall:.2f}, margin "
                f"{match.margin:.3f}, novelty {novelty:.3f}, "
                f"{track_state.confirmation_count} confirmations"
            ),
            similarity=match.best_similarity,
            quality=quality.overall,
            margin=match.margin,
            novelty=novelty,
            embedding_key=record.key,
            frame_index=frame_index,
        )
        self.history.append(accepted)
        # Logged at INFO: a gallery change is an auditable event, not a detail.
        logger.info("Gallery updated", extra=accepted.to_dict())
        return accepted

    def _novelty(self, identity_id: str, embedding: np.ndarray) -> float:
        """1 - max similarity to what this person already has stored."""
        identity = self._gallery.get(identity_id)
        if identity is None or not identity.embeddings:
            return 1.0
        matrix, _ = identity.matrix()
        if matrix.size == 0:
            return 1.0
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return float(1.0 - np.max(matrix @ query))

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.history:
            counts[record.outcome.value] = counts.get(record.outcome.value, 0) + 1
        return {
            "considered": len(self.history),
            "accepted": sum(1 for r in self.history if r.accepted),
            "by_outcome": dict(sorted(counts.items())),
        }
