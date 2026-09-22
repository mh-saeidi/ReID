"""Per-track identity state, occlusion tolerance and lifecycle events.

Holds the :class:`TrackState` for every live track, decides when a track has
been gone long enough to forget, and applies the identity stabilizer. The
tracker itself keeps a track alive across short occlusions (``track_buffer``);
this manager keeps the *identity* alive across them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src.config.schema import MatchingConfig, TrackingConfig
from src.core.types import RecognitionResult, RecognitionStatus
from src.tracking.stabilizer import IdentityStabilizer
from src.tracking.tracker import TrackState
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrackTransition:
    """A lifecycle change the event layer turns into an event."""

    kind: str
    """``started``, ``ended``, ``identified`` or ``identity_changed``."""
    state: TrackState
    previous_identity: str | None = None


class TrackManager:
    """Owns the lifetime and identity state of every live track."""

    def __init__(
        self,
        tracking: TrackingConfig,
        matching: MatchingConfig,
        *,
        absent_hold_frames: int = 0,
    ) -> None:
        self._config = tracking
        self._stabilizer = IdentityStabilizer(
            tracking.identity_stability, matching, absent_hold_frames=absent_hold_frames
        )
        self._tracks: dict[int, TrackState] = {}
        self._next_synthetic_id = -1

    # ------------------------------------------------------------- accessors
    @property
    def tracks(self) -> dict[int, TrackState]:
        return self._tracks

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    def get(self, track_id: int) -> TrackState | None:
        return self._tracks.get(track_id)

    def __iter__(self) -> Iterator[TrackState]:
        return iter(self._tracks.values())

    def synthetic_id(self) -> int:
        """Negative id for a detection the tracker did not assign one to.

        Keeps per-detection bookkeeping uniform without ever colliding with a
        real tracker id.
        """
        value = self._next_synthetic_id
        self._next_synthetic_id -= 1
        return value

    # ---------------------------------------------------------------- updates
    def touch(self, track_id: int, frame_index: int, timestamp: float) -> tuple[TrackState, bool]:
        """Create or refresh a track. Returns the state and whether it is new."""
        state = self._tracks.get(track_id)
        if state is None:
            state = TrackState(
                track_id=track_id,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
                first_seen_time=timestamp,
                last_seen_time=timestamp,
            )
            self._tracks[track_id] = state
            state.frames_seen = 1
            return state, True
        state.last_seen_frame = frame_index
        state.last_seen_time = timestamp
        state.frames_seen += 1
        return state, False

    def apply_recognition(
        self,
        state: TrackState,
        result: RecognitionResult,
        frame_index: int,
    ) -> tuple[RecognitionResult, TrackTransition | None]:
        """Fold a per-frame recognition into the track and stabilize it."""
        previous = state.identity.current_identity
        stabilized = self._stabilizer.observe(state.identity, result, frame_index)
        state.last_result = stabilized

        if stabilized.is_recognized:
            state.frames_recognized += 1
        elif stabilized.status is RecognitionStatus.NO_FACE:
            state.frames_no_face += 1
        else:
            state.frames_unknown += 1

        current = state.identity.current_identity
        transition: TrackTransition | None = None
        if current != previous:
            if previous is None and current is not None:
                transition = TrackTransition("identified", state, previous)
            elif current is not None:
                transition = TrackTransition("identity_changed", state, previous)
        return stabilized, transition

    def expire(self, frame_index: int) -> list[TrackState]:
        """Forget tracks unseen for longer than ``lost_track_timeout`` frames."""
        timeout = self._config.lost_track_timeout
        expired = [
            state
            for state in self._tracks.values()
            if frame_index - state.last_seen_frame > timeout
        ]
        for state in expired:
            self._tracks.pop(state.track_id, None)
            logger.debug(
                "Track expired",
                extra={
                    "track_id": state.track_id,
                    "frames_seen": state.frames_seen,
                    "identity": state.identity_id,
                },
            )
        return expired

    def reset(self) -> list[TrackState]:
        """Drop every track (used when switching input source)."""
        ended = list(self._tracks.values())
        self._tracks.clear()
        self._next_synthetic_id = -1
        return ended

    # -------------------------------------------------------------- ReID gate
    def needs_reid(self, state: TrackState, frame_index: int, performance) -> bool:
        """Decide whether to spend a ReID forward pass on this track now.

        Running the encoder on every box of every frame is wasteful once a track
        is stable, so embeddings are refreshed every ``reid_interval`` frames.
        Newly created tracks and tracks that are still unknown are refreshed
        every frame, because those are exactly the cases where skipping would
        cost a recognition.
        """
        if not performance.embedding_cache or performance.reid_interval <= 1:
            return True
        if state.last_embedding is None:
            return True
        if performance.force_reid_on_new_track and state.frames_seen <= 1:
            return True
        if performance.force_reid_when_unknown and state.identity.current_identity is None:
            return True
        return (frame_index - state.last_reid_frame) >= performance.reid_interval
