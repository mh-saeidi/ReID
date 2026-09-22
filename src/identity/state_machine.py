"""Per-track identity state machine.

    NEW_TRACK ──face evidence──▶ UNCONFIRMED ──repeated strong evidence──▶ RECOGNIZED
        │                             │                                    │
        │                             └──consistently no match──▶ UNKNOWN  │
        │                                                                  │
        └──────────────────────────────────────────────────────────────────┘
                                      ▲                                    │
                                      └───face returns──── TEMPORARILY_OCCLUDED
                                                                (face lost)

Two rules make this safe, and they are the reason the machine exists at all:

**Tracking never creates an identity.** A track can only enter RECOGNIZED
through accumulated *face* evidence. A masked person walking into frame has no
face evidence, so no amount of tracking will name them -- they stay UNKNOWN or
UNCERTAIN. This is what stops a new masked person inheriting the identity of
whoever was last seen.

**Tracking may preserve an identity it did not create.** A track that already
reached RECOGNIZED may pass through TEMPORARILY_OCCLUDED while the face is
unavailable, for a bounded number of frames. That is carrying forward an
already-earned conclusion, not inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.identity.types import (
    IdentityEvidence,
    IdentityObservation,
    TrackIdentityPhase,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class StabilityConfig:
    """Thresholds governing the transitions.

    Separate thresholds for acquiring, holding and switching an identity are
    what prevent the classic ``John / Unknown / John / Jane`` flicker: it should
    be harder to take an identity away than to confirm it, and harder still to
    replace it with a different one.
    """

    history_size: int = 15
    min_confirmation_frames: int = 4
    switch_margin: float = 0.10
    lost_track_timeout: int = 30
    occlusion_hold_frames: int = 45
    unknown_frames_to_release: int = 12
    min_evidence_weight: float = 1.0
    """Total quality-weighted evidence needed before naming anybody. Four
    pristine frames clear this; a dozen terrible ones do not."""


@dataclass(slots=True)
class TrackIdentityState:
    """Identity state for one tracked object."""

    track_id: int
    phase: TrackIdentityPhase = TrackIdentityPhase.NEW_TRACK
    evidence: IdentityEvidence = field(default=None)  # type: ignore[assignment]

    current_identity: str | None = None
    current_similarity: float = 0.0
    current_quality: float = 0.0
    confirmation_count: int = 0
    unknown_streak: int = 0
    occluded_since_frame: int | None = None

    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_face_frame: int = -1_000_000
    identity_switches: int = 0
    frames_named: int = 0

    def __post_init__(self) -> None:
        if self.evidence is None:
            self.evidence = IdentityEvidence(track_id=self.track_id)

    @property
    def is_confirmed(self) -> bool:
        return self.phase is TrackIdentityPhase.RECOGNIZED

    @property
    def track_stability(self) -> float:
        """How settled this track's identity is, in [0, 1].

        Reported separately from identity confidence: a rock-steady track of an
        unregistered person is stable *and* unknown.
        """
        if self.current_identity is None:
            return 0.0
        seen = max(1, self.last_seen_frame - self.first_seen_frame + 1)
        consistency = self.frames_named / seen
        confirmation = min(1.0, self.confirmation_count / 6.0)
        churn = 1.0 / (1.0 + self.identity_switches)
        return float(np.clip(0.45 * consistency + 0.35 * confirmation + 0.20 * churn, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "phase": self.phase.value,
            "current_identity": self.current_identity,
            "current_similarity": round(self.current_similarity, 6),
            "current_quality": round(self.current_quality, 4),
            "confirmation_count": self.confirmation_count,
            "unknown_streak": self.unknown_streak,
            "identity_switches": self.identity_switches,
            "track_stability": round(self.track_stability, 4),
            "frames_named": self.frames_named,
            "evidence": self.evidence.to_dict(),
        }


class IdentityStateMachine:
    """Drives one track's identity phase from accumulated face evidence."""

    def __init__(self, config: StabilityConfig) -> None:
        self._config = config

    # ---------------------------------------------------------- observation
    def observe_face(
        self,
        state: TrackIdentityState,
        observation: IdentityObservation,
        *,
        accepted: bool,
    ) -> TrackIdentityPhase:
        """Fold in one frame that produced a usable face.

        Args:
            accepted: Whether the match cleared its threshold. A sub-threshold
                observation is still recorded as evidence *against* naming, but
                never accumulates toward naming.
        """
        config = self._config
        state.last_face_frame = observation.frame_index
        state.occluded_since_frame = None

        evidence = state.evidence
        evidence.observations.append(observation)
        if len(evidence.observations) > config.history_size:
            removed = evidence.observations.pop(0)
            self._unaccumulate(evidence, removed)

        if accepted and observation.identity_id:
            self._accumulate(evidence, observation)
            evidence.confirmations += 1
            state.unknown_streak = 0
        else:
            evidence.unknown_count += 1
            state.unknown_streak += 1

        return self._resolve(state, observation.frame_index)

    def observe_no_face(
        self, state: TrackIdentityState, frame_index: int
    ) -> TrackIdentityPhase:
        """Fold in a frame with nothing to recognise from.

        Deliberately *not* evidence against the identity: a person turning
        their head is not evidence that they are someone else. It only starts
        the occlusion clock.
        """
        state.evidence.no_face_count += 1
        if state.occluded_since_frame is None:
            state.occluded_since_frame = frame_index

        if state.phase in (TrackIdentityPhase.RECOGNIZED,
                           TrackIdentityPhase.TEMPORARILY_OCCLUDED):
            elapsed = frame_index - state.occluded_since_frame
            if elapsed <= self._config.occlusion_hold_frames:
                state.phase = TrackIdentityPhase.TEMPORARILY_OCCLUDED
                return state.phase
            logger.debug(
                "Identity released: face unavailable beyond the hold budget",
                extra={
                    "track_id": state.track_id,
                    "identity": state.current_identity,
                    "frames": elapsed,
                },
            )
            state.current_identity = None
            state.current_similarity = 0.0
            state.confirmation_count = 0
            state.phase = TrackIdentityPhase.UNKNOWN
            return state.phase

        # An unconfirmed or brand-new track has nothing to carry. Critically,
        # it does NOT inherit an identity from anywhere: a new masked person
        # stays unnamed until their own face provides evidence.
        if state.phase is TrackIdentityPhase.NEW_TRACK:
            state.phase = TrackIdentityPhase.UNCONFIRMED
        return state.phase

    def mark_lost(self, state: TrackIdentityState) -> TrackIdentityPhase:
        state.phase = TrackIdentityPhase.LOST
        return state.phase

    # ------------------------------------------------------------ internals
    @staticmethod
    def _accumulate(evidence: IdentityEvidence, observation: IdentityObservation) -> None:
        identity_id = observation.identity_id
        if identity_id is None:
            return
        weight = observation.weight
        evidence.scores_by_identity[identity_id] = (
            evidence.scores_by_identity.get(identity_id, 0.0)
            + weight * observation.similarity
        )
        evidence.weights_by_identity[identity_id] = (
            evidence.weights_by_identity.get(identity_id, 0.0) + weight
        )

    @staticmethod
    def _unaccumulate(evidence: IdentityEvidence, observation: IdentityObservation) -> None:
        """Remove an observation that has aged out of the window."""
        identity_id = observation.identity_id
        if identity_id is None or identity_id not in evidence.weights_by_identity:
            return
        weight = observation.weight
        evidence.scores_by_identity[identity_id] -= weight * observation.similarity
        evidence.weights_by_identity[identity_id] -= weight
        if evidence.weights_by_identity[identity_id] <= 1e-6:
            evidence.scores_by_identity.pop(identity_id, None)
            evidence.weights_by_identity.pop(identity_id, None)

    def _resolve(self, state: TrackIdentityState, frame_index: int) -> TrackIdentityPhase:
        """Decide the phase from the evidence accumulated so far."""
        config = self._config
        ranked = state.evidence.ranked_candidates()

        if not ranked:
            if state.unknown_streak >= config.unknown_frames_to_release:
                state.current_identity = None
                state.phase = TrackIdentityPhase.UNKNOWN
            elif state.phase is TrackIdentityPhase.NEW_TRACK:
                state.phase = TrackIdentityPhase.UNCONFIRMED
            return state.phase

        leader, leader_score = ranked[0]
        leader_weight = state.evidence.weights_by_identity.get(leader, 0.0)
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0

        enough_evidence = (
            state.evidence.confirmations >= config.min_confirmation_frames
            and leader_weight >= config.min_evidence_weight
        )

        if state.current_identity is None:
            if enough_evidence:
                state.current_identity = leader
                state.current_similarity = leader_score
                state.confirmation_count = state.evidence.confirmations
                state.phase = TrackIdentityPhase.RECOGNIZED
                state.frames_named += 1
            else:
                state.phase = TrackIdentityPhase.UNCONFIRMED
            return state.phase

        if leader != state.current_identity:
            # Hysteresis: replacing an established identity needs a clear win,
            # otherwise two similar-looking people swap labels frame to frame.
            if (leader_score - state.current_similarity) >= config.switch_margin and enough_evidence:
                logger.debug(
                    "Track identity switched",
                    extra={
                        "track_id": state.track_id,
                        "from": state.current_identity,
                        "to": leader,
                        "margin": round(leader_score - state.current_similarity, 4),
                    },
                )
                state.current_identity = leader
                state.current_similarity = leader_score
                state.identity_switches += 1
                state.confirmation_count = state.evidence.confirmations
            state.phase = TrackIdentityPhase.RECOGNIZED
            state.frames_named += 1
            return state.phase

        state.current_similarity = leader_score
        state.confirmation_count = state.evidence.confirmations
        state.phase = TrackIdentityPhase.RECOGNIZED
        state.frames_named += 1
        _ = runner_score
        return state.phase
