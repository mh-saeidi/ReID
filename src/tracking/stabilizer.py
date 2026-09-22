"""Temporal identity stabilization.

A single frame is a weak piece of evidence: motion blur, a turned back or a
passing occluder can drop the similarity below the threshold for one or two
frames. Displaying the raw per-frame decision makes the label flicker
``John -> Unknown -> John``, which is both ugly and operationally useless.

Three strategies are provided; ``weighted_vote`` is the default.

**weighted_vote** (default)
    Each observation in the track's history contributes
    ``decay ** age * similarity`` to the candidate identity it voted for, where
    ``age`` is how many observations ago it happened. Recent, confident
    observations dominate; old ones fade out. A candidate is only *eligible*
    once it has at least ``minimum_recognized_frames`` recognised votes in the
    window, which is what stops a single lucky frame from naming someone. An
    incumbent identity is only replaced when the challenger's score beats it by
    ``switch_margin``, so two similar-looking people do not swap labels back and
    forth on marginal frames (hysteresis).

**ema**
    An exponential moving average of the similarity is kept per candidate,
    updated towards the observed similarity (or towards zero when that identity
    was not the best match). The winner is the highest average above the
    recognition threshold. Smoother than voting but slower to acquire.

**majority**
    Plain unweighted count over the window. Simple and predictable; ignores how
    confident each observation was.

In all strategies a recognised identity survives up to
``identity_persistence_frames`` consecutive unknown observations before the
track falls back to ``Unknown``. That is what carries an identity through a
short occlusion instead of resetting it on the first bad frame.

Face mode adds a third kind of observation: *no face visible*. Someone turning
around produces frames with nothing to compare, which is neither a match nor a
rejection. Those frames are excluded from the evidence window entirely, and the
track holds its established identity for ``face.identity_hold_frames``, after
which the label is dropped rather than carried indefinitely.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.config.schema import IdentityStabilityConfig, MatchingConfig, StabilizationStrategy
from src.core.types import RecognitionResult, RecognitionStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    """One per-frame recognition outcome for a track."""

    identity_id: str | None
    similarity: float
    recognized: bool
    frame_index: int


@dataclass(slots=True)
class IdentityHistory:
    """Rolling evidence window for one track."""

    history: deque[IdentityObservation] = field(default_factory=deque)
    current_identity: str | None = None
    current_similarity: float = 0.0
    unknown_streak: int = 0
    absent_since_frame: int | None = None
    """Frame index at which the face stopped being available, or None.

    Deliberately a frame index rather than an observation counter: the
    recognition scheduler decides *how often* to look, and the identity-hold
    budget is specified in frames. Counting observations would silently
    lengthen the hold whenever the scheduler backed off."""
    absent_streak: int = 0
    """Frames (not observations) since the face was last usable."""
    frames_recognized: int = 0
    frames_unknown: int = 0
    frames_absent: int = 0
    switches: int = 0

    def candidates(self) -> set[str]:
        return {o.identity_id for o in self.history if o.identity_id is not None}


class IdentityStabilizer:
    """Smooths per-frame recognition into a stable, per-track identity."""

    def __init__(
        self,
        config: IdentityStabilityConfig,
        matching: MatchingConfig,
        *,
        absent_hold_frames: int = 0,
    ) -> None:
        self._config = config
        self._matching = matching
        self._absent_hold_frames = absent_hold_frames

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def observe(
        self,
        state: IdentityHistory,
        result: RecognitionResult,
        frame_index: int,
    ) -> RecognitionResult:
        """Record one observation and return the stabilized decision."""
        if not self._config.enabled:
            return result

        if result.status is RecognitionStatus.NO_FACE:
            return self._observe_absent(state, result, frame_index)

        state.absent_streak = 0
        state.absent_since_frame = None
        recognized = result.is_recognized and result.identity_id is not None
        observation = IdentityObservation(
            identity_id=result.identity_id if recognized else None,
            similarity=float(result.similarity),
            recognized=recognized,
            frame_index=frame_index,
        )
        if len(state.history) >= self._config.history_size:
            state.history.popleft()
        state.history.append(observation)

        if recognized:
            state.frames_recognized += 1
            state.unknown_streak = 0
        else:
            state.frames_unknown += 1
            state.unknown_streak += 1

        winner, score = self._elect(state)
        decided = self._apply_hysteresis(state, winner, score, result)
        return decided

    def _observe_absent(
        self, state: IdentityHistory, result: RecognitionResult, frame_index: int
    ) -> RecognitionResult:
        """Handle a frame with nothing to compare (face mode, face not visible).

        Such a frame is *neutral*: it is not appended to the evidence window,
        because doing so would let someone turning around slowly erode their own
        identity. The track keeps the identity it already established, for up to
        ``absent_hold_frames``. After that the label is dropped rather than
        carried indefinitely, which bounds the damage if the tracker has
        meanwhile swapped this track onto a different person.
        """
        if state.absent_since_frame is None:
            state.absent_since_frame = frame_index
        # Measured in frames so the budget means what the configuration says,
        # regardless of how often the scheduler chose to look.
        state.absent_streak = max(1, frame_index - state.absent_since_frame + 1)
        state.frames_absent += 1

        if state.current_identity is not None:
            if state.absent_streak <= self._absent_hold_frames:
                held = self._as_result(
                    state, state.current_identity, state.current_similarity, result
                )
                held.detail = (
                    f"face not visible for {state.absent_streak} frame(s); "
                    "identity held from an earlier confirmed match"
                )
                return held
            logger.debug(
                "Track identity released: face hidden for too long",
                extra={
                    "identity": state.current_identity,
                    "absent_streak": state.absent_streak,
                    "hold_frames": self._absent_hold_frames,
                },
            )
            state.current_identity = None
            state.current_similarity = 0.0
        return result

    # ------------------------------------------------------------- strategies
    def _elect(self, state: IdentityHistory) -> tuple[str | None, float]:
        strategy = self._config.strategy
        if strategy is StabilizationStrategy.WEIGHTED_VOTE:
            return self._weighted_vote(state.history)
        if strategy is StabilizationStrategy.EMA:
            return self._ema(state.history)
        return self._majority(state.history)

    def _eligible(self, observations: Iterable[IdentityObservation], identity_id: str) -> bool:
        votes = sum(1 for o in observations if o.recognized and o.identity_id == identity_id)
        return votes >= self._config.minimum_recognized_frames

    def _weighted_vote(
        self, history: Iterable[IdentityObservation]
    ) -> tuple[str | None, float]:
        observations = list(history)
        if not observations:
            return None, 0.0
        decay = self._config.decay
        scores: dict[str, float] = {}
        weights: dict[str, float] = {}
        newest = len(observations) - 1
        for index, observation in enumerate(observations):
            if not observation.recognized or observation.identity_id is None:
                continue
            weight = decay ** (newest - index)
            scores[observation.identity_id] = (
                scores.get(observation.identity_id, 0.0) + weight * observation.similarity
            )
            weights[observation.identity_id] = weights.get(observation.identity_id, 0.0) + weight
        eligible = {
            identity: scores[identity] / weights[identity]
            for identity in scores
            if self._eligible(observations, identity)
        }
        if not eligible:
            return None, 0.0
        winner = max(eligible, key=lambda key: eligible[key])
        return winner, eligible[winner]

    def _ema(self, history: Iterable[IdentityObservation]) -> tuple[str | None, float]:
        observations = list(history)
        if not observations:
            return None, 0.0
        alpha = 1.0 - self._config.decay
        averages: dict[str, float] = {}
        for observation in observations:
            voted = observation.identity_id if observation.recognized else None
            if voted is not None and voted not in averages:
                # Seed from the first real observation. Starting at zero would
                # make the average lag far below the similarity scale the
                # recognition threshold is expressed in, so a correct identity
                # could stay ineligible forever.
                averages[voted] = observation.similarity
                continue
            for identity in averages:
                target = observation.similarity if identity == voted else 0.0
                averages[identity] = (1 - alpha) * averages[identity] + alpha * target
        eligible = {
            identity: value
            for identity, value in averages.items()
            if value >= self._matching.recognition_threshold
            and self._eligible(observations, identity)
        }
        if not eligible:
            return None, 0.0
        winner = max(eligible, key=lambda key: eligible[key])
        return winner, eligible[winner]

    def _majority(self, history: Iterable[IdentityObservation]) -> tuple[str | None, float]:
        observations = list(history)
        counts: dict[str, int] = {}
        sums: dict[str, float] = {}
        for observation in observations:
            if observation.recognized and observation.identity_id is not None:
                counts[observation.identity_id] = counts.get(observation.identity_id, 0) + 1
                sums[observation.identity_id] = (
                    sums.get(observation.identity_id, 0.0) + observation.similarity
                )
        eligible = {i: c for i, c in counts.items() if c >= self._config.minimum_recognized_frames}
        if not eligible:
            return None, 0.0
        winner = max(eligible, key=lambda key: (eligible[key], sums[key]))
        return winner, sums[winner] / counts[winner]

    # ------------------------------------------------------------- hysteresis
    def _apply_hysteresis(
        self,
        state: IdentityHistory,
        winner: str | None,
        score: float,
        instantaneous: RecognitionResult,
    ) -> RecognitionResult:
        incumbent = state.current_identity
        persistence = self._config.identity_persistence_frames
        # A run of unknown frames longer than the persistence budget releases the
        # identity even if the evidence window still holds old votes for it.
        # Without this the label would survive for as long as the history window,
        # which is not what identity_persistence_frames promises.
        exhausted = state.unknown_streak > persistence

        if winner is None or exhausted:
            if incumbent is not None and not exhausted:
                # A short unknown streak (occlusion, blur, back turned) holds.
                return self._as_result(state, incumbent, state.current_similarity, instantaneous)
            if incumbent is not None:
                logger.debug(
                    "Track identity released",
                    extra={"identity": incumbent, "unknown_streak": state.unknown_streak},
                )
            state.current_identity = None
            state.current_similarity = 0.0
            return RecognitionResult(
                status=RecognitionStatus.UNKNOWN,
                identity_name=self._matching.unknown_label,
                similarity=instantaneous.similarity,
                runner_up_id=instantaneous.runner_up_id,
                runner_up_similarity=instantaneous.runner_up_similarity,
                all_similarities=instantaneous.all_similarities,
            )

        if incumbent is not None and winner != incumbent:
            challenger_margin = score - state.current_similarity
            if challenger_margin < self._config.switch_margin:
                # Not enough evidence to take the label away from the incumbent.
                return self._as_result(state, incumbent, state.current_similarity, instantaneous)
            state.switches += 1
            logger.debug(
                "Track identity switched",
                extra={
                    "from": incumbent,
                    "to": winner,
                    "margin": round(challenger_margin, 4),
                    "required": self._config.switch_margin,
                },
            )

        state.current_identity = winner
        state.current_similarity = score
        return self._as_result(state, winner, score, instantaneous)

    def _as_result(
        self,
        state: IdentityHistory,
        identity_id: str,
        score: float,
        instantaneous: RecognitionResult,
    ) -> RecognitionResult:
        status = (
            RecognitionStatus.RECOGNIZED
            if score >= self._matching.high_confidence_threshold
            else RecognitionStatus.LOW_CONFIDENCE
        )
        name = (
            instantaneous.identity_name
            if instantaneous.identity_id == identity_id
            else identity_id
        )
        title = instantaneous.identity_title if instantaneous.identity_id == identity_id else None
        return RecognitionResult(
            status=status,
            identity_id=identity_id,
            identity_name=name,
            identity_title=title,
            similarity=score,
            runner_up_id=instantaneous.runner_up_id,
            runner_up_similarity=instantaneous.runner_up_similarity,
            all_similarities=instantaneous.all_similarities,
        )
