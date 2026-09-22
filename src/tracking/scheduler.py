"""Recognition scheduling: deciding which tracks get a face pass this frame.

The previous rule was "every N frames, unless the track is unidentified, in
which case every frame". That second clause is the expensive one: a genuinely
unregistered person -- a passer-by who will never match -- pinned the encoder
at full rate for as long as they stayed in view, which is precisely the case
that should cost the least.

This scheduler makes the decision depend on what the track's identity state
actually is, and bounds the unregistered case with geometric backoff. The
semantics are unchanged: it only ever decides *when* to look, never *what the
answer is*. A skipped frame produces no identity claim of its own -- the
existing bounded identity-hold in the stabilizer decides what is displayed.

States and their intent:

``NEW_TRACK``     just appeared -- recognise immediately, this is the frame
                  that matters most.
``RECOGNIZED``    confidently identified and stable -- re-check occasionally to
                  catch a tracker ID swap onto a different person.
``LOW_CONFIDENCE`` identified but near the threshold -- re-check often, this is
                  the state most likely to be wrong.
``UNKNOWN``       compared and matched nobody -- retry with backoff.
``NO_FACE``       nothing to compare. The *encoder* never runs, but the much
                  cheaper face detector is re-run periodically to notice the
                  face coming back.
``RECOVERING``    seen again after being lost -- the tracker may have re-used
                  the ID for someone else, so force a fresh pass.
``LOST``          not seen this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config.schema import RecognitionSchedulerConfig, SchedulerState
from src.core.types import RecognitionStatus
from src.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.tracking.tracker import TrackState

logger = get_logger(__name__)


@dataclass(slots=True)
class SchedulerDecision:
    """Whether to recognise this track now, and why."""

    should_recognize: bool
    state: SchedulerState
    reason: str
    interval: int = 0
    frames_since: int = 0
    forced: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "should_recognize": self.should_recognize,
            "state": self.state.value,
            "reason": self.reason,
            "interval": self.interval,
            "frames_since": self.frames_since,
            "forced": self.forced,
        }


@dataclass(slots=True)
class SchedulerStats:
    """Counters for benchmarking and the metrics overlay."""

    attempts: int = 0
    skips: int = 0
    forced: int = 0
    by_state: dict[str, int] = field(default_factory=dict)

    def record(self, decision: SchedulerDecision) -> None:
        key = decision.state.value
        self.by_state[key] = self.by_state.get(key, 0) + 1
        if decision.should_recognize:
            self.attempts += 1
            if decision.forced:
                self.forced += 1
        else:
            self.skips += 1

    @property
    def skip_rate(self) -> float:
        total = self.attempts + self.skips
        return self.skips / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "recognition_attempts": self.attempts,
            "recognition_skips": self.skips,
            "recognition_forced": self.forced,
            "skip_rate": round(self.skip_rate, 4),
            "by_state": dict(sorted(self.by_state.items())),
        }


class RecognitionScheduler:
    """Decides, per track per frame, whether to run face recognition."""

    def __init__(self, config: RecognitionSchedulerConfig, matching_config=None) -> None:
        self._config = config
        self._matching = matching_config
        self.stats = SchedulerStats()
        self._recognized_this_frame = 0
        self._frame_index = -1

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def begin_frame(self, frame_index: int) -> None:
        """Reset the per-frame recognition budget."""
        self._frame_index = frame_index
        self._recognized_this_frame = 0

    # ------------------------------------------------------------ classify
    def classify(self, state: TrackState, frame_index: int) -> SchedulerState:
        """Map a track's identity state onto a scheduler state."""
        cache = state.recognition_cache

        if state.frames_seen <= 1 or cache.last_recognition_frame < 0:
            return SchedulerState.NEW_TRACK

        # A gap in observations means the tracker may have re-used this ID for a
        # different person, so the identity has to be re-established.
        if frame_index - state.last_seen_frame > 1:
            return SchedulerState.LOST
        if cache.was_lost:
            return SchedulerState.RECOVERING

        status = cache.last_status
        if status is RecognitionStatus.NO_FACE:
            return SchedulerState.NO_FACE
        if status is RecognitionStatus.RECOGNIZED:
            return SchedulerState.RECOGNIZED
        if status is RecognitionStatus.LOW_CONFIDENCE:
            return SchedulerState.LOW_CONFIDENCE
        if status in (RecognitionStatus.UNKNOWN, RecognitionStatus.REJECTED):
            return SchedulerState.UNKNOWN
        return SchedulerState.NEW_TRACK

    # ------------------------------------------------------------- decision
    def decide(self, state: TrackState, frame_index: int) -> SchedulerDecision:
        """Should this track be recognised on this frame?"""
        scheduler_state = self.classify(state, frame_index)
        cache = state.recognition_cache
        cache.scheduler_state = scheduler_state

        if not self._config.enabled:
            decision = SchedulerDecision(
                True, scheduler_state, "scheduler disabled: recognising every frame"
            )
            return self._finish(cache, decision, frame_index)

        frames_since = frame_index - cache.last_recognition_frame

        forced_reason = self._forced_reason(state, scheduler_state, cache)
        if forced_reason is not None:
            decision = SchedulerDecision(
                True, scheduler_state, forced_reason, frames_since=frames_since, forced=True
            )
            return self._finish(cache, decision, frame_index)

        interval = self._interval_for(scheduler_state, cache)
        should = frames_since >= interval
        reason = (
            f"{scheduler_state.value}: {frames_since} frames since the last pass "
            f"(interval {interval})"
        )
        decision = SchedulerDecision(
            should, scheduler_state, reason, interval=interval, frames_since=frames_since
        )
        return self._finish(cache, decision, frame_index)

    def _forced_reason(self, state: TrackState, scheduler_state: SchedulerState,
                       cache: RecognitionCache) -> str | None:
        config = self._config
        if scheduler_state is SchedulerState.NEW_TRACK and config.force_on_new_track:
            return "new track: recognise immediately"
        if scheduler_state in (SchedulerState.RECOVERING, SchedulerState.LOST):
            if config.force_on_recovery:
                return "track recovered after a gap: re-establish identity"
        if cache.identity_changed and config.force_when_identity_changes:
            return "identity changed: confirm with a fresh pass"
        if cache.force_next:
            return cache.force_reason or "recognition explicitly forced"
        return None

    def _interval_for(self, scheduler_state: SchedulerState,
                      cache: RecognitionCache) -> int:
        config = self._config
        if scheduler_state is SchedulerState.RECOGNIZED:
            return config.stable_interval
        if scheduler_state is SchedulerState.LOW_CONFIDENCE:
            return config.low_confidence_interval
        if scheduler_state is SchedulerState.NO_FACE:
            return config.no_face_interval
        if scheduler_state is SchedulerState.UNKNOWN:
            if not config.unknown_backoff_enabled:
                return config.unknown_interval
            # Geometric backoff: an unregistered person who has failed to match
            # N times in a row is unlikely to match on frame N+1.
            interval = config.unknown_interval * (
                config.unknown_backoff_factor ** cache.consecutive_unknown
            )
            return int(min(interval, config.unknown_backoff_max_interval))
        return config.unknown_interval

    def _finish(self, cache: RecognitionCache, decision: SchedulerDecision,
                frame_index: int) -> SchedulerDecision:
        """Apply the per-frame budget and record the outcome."""
        budget = self._config.max_recognitions_per_frame
        if (
            decision.should_recognize
            and budget
            and self._frame_index == frame_index
            and self._recognized_this_frame >= budget
            and not decision.forced
        ):
            decision.should_recognize = False
            decision.reason = (
                f"deferred: frame budget of {budget} recognitions already spent"
            )

        if decision.should_recognize:
            self._recognized_this_frame += 1
            cache.force_next = False
            cache.force_reason = ""
            cache.identity_changed = False
            cache.was_lost = False
        self.stats.record(decision)
        return decision

    def reset(self) -> None:
        self.stats = SchedulerStats()
        self._recognized_this_frame = 0
        self._frame_index = -1


# Imported late to avoid a circular import at module load time.
from src.tracking.recognition_cache import RecognitionCache  # noqa: E402
