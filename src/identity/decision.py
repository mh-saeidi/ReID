"""The identity decision engine.

Everything upstream produces evidence; this is the only place that turns
evidence into a claim about who someone is. Concentrating that here means the
rules can be read in one place and tested directly.

The order of checks matters, and each one exists to prevent a specific failure:

1. **No usable face** -> NO_FACE, or TEMPORARILY_MAINTAINED for a track that
   already earned an identity. Never a match, never a rejection.
2. **Quality floor** -> a face too degraded to trust is reported as such rather
   than pushed through the matcher, where it would produce a plausible-looking
   score drawn from a distribution nobody calibrated.
3. **Empty gallery** -> UNKNOWN. There is no "closest" identity when there are
   no identities.
4. **Threshold, chosen by visibility** -> a masked face is compared against a
   masked-face threshold, because its scores come from a different distribution.
5. **Ambiguity margin** -> when two registered people score comparably, naming
   the higher one is a coin flip dressed as a decision.
6. **Calibrated confidence** -> applied only when calibration data exists.
7. **Temporal evidence** -> for tracked sources, the accumulated per-track
   evidence governs, not this single frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.face.quality import FaceQualityScore
from src.face.visibility import FaceVisibility
from src.identity.calibration import CalibrationModel
from src.identity.face_gallery import FaceGallery
from src.identity.state_machine import (
    IdentityStateMachine,
    StabilityConfig,
    TrackIdentityState,
)
from src.identity.types import (
    FaceMatchResult,
    FailureReason,
    IdentityDecision,
    IdentityObservation,
    IdentityStatus,
    TrackIdentityPhase,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MatchingThresholds:
    """Acceptance thresholds, one family per visibility class.

    A single threshold across conditions is wrong in both directions: set for
    full faces it rejects every masked one; set for masked faces it accepts
    strangers. These default to ``None``, meaning "use the calibrated value",
    and are only hard-coded when an operator overrides them deliberately.
    """

    full: float | None = None
    partial: float | None = None
    masked: float | None = None
    ambiguity_margin: float = 0.06
    min_quality: float = 0.25
    min_identity_confidence: float | None = None

    def for_visibility(self, visibility: FaceVisibility) -> tuple[float | None, str]:
        key = visibility.threshold_key
        return getattr(self, key, None), key


class IdentityDecisionEngine:
    """Fuses face match, quality, visibility and temporal evidence."""

    def __init__(
        self,
        gallery: FaceGallery,
        thresholds: MatchingThresholds,
        *,
        calibration: CalibrationModel | None = None,
        stability: StabilityConfig | None = None,
        fallback_threshold: float = 0.35,
    ) -> None:
        self._gallery = gallery
        self._thresholds = thresholds
        self._calibration = calibration
        self._stability = stability or StabilityConfig()
        self._machine = IdentityStateMachine(self._stability)
        self._fallback_threshold = fallback_threshold

    @property
    def calibration(self) -> CalibrationModel | None:
        return self._calibration

    def set_calibration(self, calibration: CalibrationModel | None) -> None:
        self._calibration = calibration

    @property
    def state_machine(self) -> IdentityStateMachine:
        return self._machine

    # ------------------------------------------------------------ threshold
    def resolve_threshold(self, visibility: FaceVisibility) -> tuple[float, str, str]:
        """(threshold, family, source) for this visibility class."""
        configured, family = self._thresholds.for_visibility(visibility)
        if configured is not None:
            return configured, family, "configured"
        if self._calibration is not None:
            calibrated = self._calibration.threshold(family)
            if calibrated is not None:
                return calibrated, family, "calibrated"
        return self._fallback_threshold, family, "fallback"

    def calibrated_confidence(
        self, similarity: float, visibility: FaceVisibility
    ) -> float | None:
        """Posterior probability, or ``None`` when nothing measured supports one."""
        if self._calibration is None:
            return None
        return self._calibration.probability(similarity, visibility.threshold_key)

    # ------------------------------------------------------------- decision
    def decide(
        self,
        *,
        detector_confidence: float,
        face_detection_confidence: float = 0.0,
        embedding: np.ndarray | None = None,
        quality: FaceQualityScore | None = None,
        visibility: FaceVisibility = FaceVisibility.NO_USABLE_FACE,
        track_state: TrackIdentityState | None = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
        face_failure: FailureReason | None = None,
        face_detail: str = "",
    ) -> IdentityDecision:
        """Produce the identity decision for one detection."""
        base: dict[str, Any] = {
            "detector_confidence": detector_confidence,
            "face_detection_confidence": face_detection_confidence,
            "track_id": track_state.track_id if track_state else None,
            "visibility": visibility,
            "quality_detail": quality,
            "face_quality": quality.overall if quality else 0.0,
        }

        # --- 1. nothing to recognise from ---------------------------------
        if embedding is None:
            return self._no_face_decision(
                track_state, frame_index, face_failure, face_detail, base
            )

        # --- 2. quality floor ---------------------------------------------
        quality_value = quality.overall if quality else 0.0
        if quality is not None and quality_value < self._thresholds.min_quality:
            if track_state is not None:
                self._machine.observe_no_face(track_state, frame_index)
            return self._maybe_maintain(
                track_state,
                IdentityDecision(
                    status=IdentityStatus.NO_FACE,
                    failure=_quality_failure(quality),
                    reason=(
                        f"face quality {quality_value:.2f} is below the floor "
                        f"{self._thresholds.min_quality:.2f} "
                        f"(weakest: {quality.weakest_component})"
                    ),
                    **base,
                ),
                frame_index,
            )

        # --- 3. empty gallery ---------------------------------------------
        if self._gallery.is_empty:
            return IdentityDecision(
                status=IdentityStatus.UNKNOWN,
                failure=FailureReason.EMPTY_GALLERY,
                reason="no identities are registered, so nobody can be matched",
                **base,
            )

        match = self._gallery.match(embedding)
        threshold, family, source = self.resolve_threshold(visibility)
        confidence = self.calibrated_confidence(match.best_similarity, visibility)

        base.update(
            {
                "face_similarity": match.best_similarity,
                "match": match,
                "threshold_used": threshold,
                "threshold_family": family,
                "identity_confidence": confidence,
            }
        )

        accepted, failure, reason = self._evaluate_match(
            match, threshold, source, confidence
        )

        # --- 7. temporal evidence, when tracking ---------------------------
        if track_state is not None:
            observation = IdentityObservation(
                frame_index=frame_index,
                timestamp=timestamp,
                identity_id=match.best_identity_id if accepted else None,
                similarity=match.best_similarity,
                quality=quality_value,
                visibility=visibility,
                margin=match.margin,
            )
            phase = self._machine.observe_face(track_state, observation, accepted=accepted)
            return self._from_track(track_state, phase, base, failure, reason, match)

        # Single images: this frame is all the evidence there is.
        if accepted and match.best_identity_id:
            identity = self._gallery.get(match.best_identity_id)
            return IdentityDecision(
                status=IdentityStatus.RECOGNIZED,
                identity_id=match.best_identity_id,
                name=identity.name if identity else match.best_identity_id,
                title=identity.title if identity else "",
                reason=reason,
                **base,
            )
        return IdentityDecision(
            status=IdentityStatus.UNKNOWN, failure=failure, reason=reason, **base
        )

    # ------------------------------------------------------------ internals
    def _evaluate_match(
        self,
        match: FaceMatchResult,
        threshold: float,
        source: str,
        confidence: float | None,
    ) -> tuple[bool, FailureReason | None, str]:
        """Apply the acceptance rules to a single comparison."""
        if match.best_similarity < threshold:
            return (
                False,
                FailureReason.LOW_SIMILARITY,
                (
                    f"best similarity {match.best_similarity:.3f} is below the "
                    f"{source} threshold {threshold:.3f}"
                ),
            )

        margin = self._thresholds.ambiguity_margin
        if margin > 0.0 and match.runner_up_id is not None and match.margin < margin:
            return (
                False,
                FailureReason.AMBIGUOUS_IDENTITY,
                (
                    f"'{match.best_identity_id}' ({match.best_similarity:.3f}) and "
                    f"'{match.runner_up_id}' ({match.runner_up_similarity:.3f}) are "
                    f"within {margin:.3f}; refusing to guess between them"
                ),
            )

        floor = self._thresholds.min_identity_confidence
        if floor is not None and confidence is not None and confidence < floor:
            return (
                False,
                FailureReason.LOW_CALIBRATED_CONFIDENCE,
                (
                    f"calibrated confidence {confidence:.3f} is below the "
                    f"required {floor:.3f}"
                ),
            )

        detail = f"similarity {match.best_similarity:.3f} >= {source} threshold {threshold:.3f}"
        if confidence is not None:
            detail += f"; calibrated confidence {confidence:.3f}"
        return True, None, detail

    def _no_face_decision(
        self,
        track_state: TrackIdentityState | None,
        frame_index: int,
        failure: FailureReason | None,
        detail: str,
        base: dict[str, Any],
    ) -> IdentityDecision:
        if track_state is not None:
            self._machine.observe_no_face(track_state, frame_index)
        return self._maybe_maintain(
            track_state,
            IdentityDecision(
                status=IdentityStatus.NO_FACE,
                failure=failure or FailureReason.FACE_NOT_FOUND,
                reason=detail or "no usable face in this person region",
                **base,
            ),
            frame_index,
        )

    def _maybe_maintain(
        self,
        track_state: TrackIdentityState | None,
        decision: IdentityDecision,
        frame_index: int,
    ) -> IdentityDecision:
        """Carry an already-earned identity through a short face outage.

        This is the only path by which a frame without face evidence can carry
        a name, and it is available only to a track that previously reached
        RECOGNIZED on its own facial evidence. A track that never earned an
        identity cannot acquire one here.
        """
        if track_state is None:
            return decision
        decision.phase = track_state.phase
        decision.track_stability = track_state.track_stability

        if (
            track_state.phase is TrackIdentityPhase.TEMPORARILY_OCCLUDED
            and track_state.current_identity
        ):
            identity = self._gallery.get(track_state.current_identity)
            elapsed = frame_index - (track_state.occluded_since_frame or frame_index)
            decision.status = IdentityStatus.TEMPORARILY_MAINTAINED
            decision.identity_id = track_state.current_identity
            decision.name = identity.name if identity else track_state.current_identity
            decision.title = identity.title if identity else ""
            decision.face_similarity = track_state.current_similarity
            decision.reason = (
                f"face unavailable for {elapsed} frame(s); identity held from "
                f"{track_state.confirmation_count} earlier confirmed observation(s)"
            )
            # No new evidence, so no new confidence is asserted.
            decision.identity_confidence = None
        return decision

    def _from_track(
        self,
        track_state: TrackIdentityState,
        phase: TrackIdentityPhase,
        base: dict[str, Any],
        failure: FailureReason | None,
        reason: str,
        match: FaceMatchResult,
    ) -> IdentityDecision:
        """Build the decision from the track's accumulated evidence."""
        base = dict(base)
        base["phase"] = phase
        base["track_stability"] = track_state.track_stability
        base["evidence"] = track_state.evidence

        if phase is TrackIdentityPhase.RECOGNIZED and track_state.current_identity:
            identity = self._gallery.get(track_state.current_identity)
            weighted = track_state.evidence.weighted_score(track_state.current_identity)
            # Confidence is reported from the accumulated evidence, not from
            # this single frame's score.
            base["identity_confidence"] = self.calibrated_confidence(
                weighted, base["visibility"]
            )
            return IdentityDecision(
                status=IdentityStatus.RECOGNIZED,
                identity_id=track_state.current_identity,
                name=identity.name if identity else track_state.current_identity,
                title=identity.title if identity else "",
                reason=(
                    f"{track_state.confirmation_count} confirming observation(s); "
                    f"quality-weighted similarity {weighted:.3f}"
                ),
                **base,
            )

        if phase is TrackIdentityPhase.UNCONFIRMED:
            leader = track_state.evidence.ranked_candidates()
            detail = ""
            if leader:
                detail = (
                    f" leading candidate '{leader[0][0]}' at {leader[0][1]:.3f}, "
                    f"{track_state.evidence.confirmations} confirmation(s) so far"
                )
            return IdentityDecision(
                status=IdentityStatus.UNCERTAIN,
                reason=(
                    "not enough evidence to name this person yet;" + detail
                ).strip(),
                failure=failure,
                **base,
            )

        return IdentityDecision(
            status=IdentityStatus.UNKNOWN,
            failure=failure or FailureReason.NO_IDENTITY_MATCH,
            reason=reason,
            **base,
        )


def _quality_failure(quality: FaceQualityScore) -> FailureReason:
    """Map the weakest quality component onto a specific failure reason."""
    return {
        "resolution": FailureReason.FACE_TOO_SMALL,
        "sharpness": FailureReason.FACE_TOO_BLURRY,
        "pose": FailureReason.FACE_POSE_TOO_EXTREME,
        "visibility": FailureReason.FACE_OCCLUDED,
        "landmarks": FailureReason.FACE_ALIGNMENT_FAILED,
        "exposure": FailureReason.FACE_LOW_QUALITY,
    }.get(quality.weakest_component, FailureReason.FACE_LOW_QUALITY)
