"""Temporal identity stabilization, track lifecycle and the ReID scheduling gate."""

from __future__ import annotations

import pytest

from src.config.schema import (
    IdentityStabilityConfig,
    MatchingConfig,
    PerformanceConfig,
    StabilizationStrategy,
    TrackingConfig,
)
from src.core.types import RecognitionResult, RecognitionStatus
from src.tracking.stabilizer import IdentityHistory, IdentityStabilizer
from src.tracking.track_manager import TrackManager

MATCHING = MatchingConfig(recognition_threshold=0.55, high_confidence_threshold=0.70)


def recognized(identity_id: str, similarity: float) -> RecognitionResult:
    return RecognitionResult(
        status=(
            RecognitionStatus.RECOGNIZED
            if similarity >= MATCHING.high_confidence_threshold
            else RecognitionStatus.LOW_CONFIDENCE
        ),
        identity_id=identity_id,
        identity_name=identity_id.title(),
        identity_title="Tester",
        similarity=similarity,
    )


def unknown(similarity: float = 0.4) -> RecognitionResult:
    return RecognitionResult(
        status=RecognitionStatus.UNKNOWN, identity_name="Unknown", similarity=similarity
    )


def feed(stabilizer, state, observations):
    return [
        stabilizer.observe(state, result, index) for index, result in enumerate(observations)
    ]


class TestStabilizerWarmUp:
    def test_identity_is_withheld_until_enough_evidence(self) -> None:
        config = IdentityStabilityConfig(minimum_recognized_frames=3)
        stabilizer = IdentityStabilizer(config, MATCHING)
        state = IdentityHistory()

        results = feed(stabilizer, state, [recognized("john", 0.8)] * 4)
        assert [r.identity_id for r in results] == [None, None, "john", "john"]

    def test_a_single_good_frame_never_names_anybody(self) -> None:
        config = IdentityStabilityConfig(minimum_recognized_frames=3)
        stabilizer = IdentityStabilizer(config, MATCHING)
        state = IdentityHistory()
        results = feed(
            stabilizer, state, [recognized("john", 0.95), unknown(), unknown(), unknown()]
        )
        assert all(r.identity_id is None for r in results)


class TestFlickerSuppression:
    def test_one_bad_frame_does_not_drop_the_identity(self) -> None:
        """The scenario from the brief: John, John, Unknown, John must not flicker."""
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(minimum_recognized_frames=3, identity_persistence_frames=15),
            MATCHING,
        )
        state = IdentityHistory()
        sequence = [
            recognized("john", 0.72),
            recognized("john", 0.70),
            recognized("john", 0.71),
            unknown(0.43),                 # the bad frame
            recognized("john", 0.68),
        ]
        results = feed(stabilizer, state, sequence)
        assert results[2].identity_id == "john"
        assert results[3].identity_id == "john"      # held through the bad frame
        assert results[4].identity_id == "john"
        assert state.switches == 0

    def test_a_sustained_absence_eventually_releases_the_identity(self) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(minimum_recognized_frames=2, identity_persistence_frames=3,
                                    history_size=10),
            MATCHING,
        )
        state = IdentityHistory()
        feed(stabilizer, state, [recognized("john", 0.8)] * 3)
        assert state.current_identity == "john"

        results = feed(stabilizer, state, [unknown()] * 6)
        assert results[0].identity_id == "john"      # still held
        assert results[-1].identity_id is None       # finally released
        assert results[-1].status is RecognitionStatus.UNKNOWN


class TestIdentitySwitching:
    def test_a_marginal_challenger_cannot_steal_the_label(self) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(
                minimum_recognized_frames=2, switch_margin=0.20, history_size=10
            ),
            MATCHING,
        )
        state = IdentityHistory()
        feed(stabilizer, state, [recognized("john", 0.80)] * 4)
        assert state.current_identity == "john"

        # A slightly-better rival appears but does not clear the margin.
        results = feed(stabilizer, state, [recognized("jane", 0.84)] * 2)
        assert results[-1].identity_id == "john"
        assert state.switches == 0

    def test_decisive_evidence_does_switch_the_identity(self) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(
                minimum_recognized_frames=2, switch_margin=0.05, history_size=6, decay=0.6
            ),
            MATCHING,
        )
        state = IdentityHistory()
        feed(stabilizer, state, [recognized("john", 0.60)] * 3)
        assert state.current_identity == "john"

        results = feed(stabilizer, state, [recognized("jane", 0.95)] * 4)
        assert results[-1].identity_id == "jane"
        assert state.switches == 1

    def test_switch_margin_zero_follows_the_evidence_immediately(self) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(minimum_recognized_frames=1, switch_margin=0.0),
            MATCHING,
        )
        state = IdentityHistory()
        assert stabilizer.observe(state, recognized("john", 0.8), 0).identity_id == "john"
        assert stabilizer.observe(state, recognized("jane", 0.9), 1).identity_id == "jane"


class TestStrategies:
    @pytest.mark.parametrize(
        "strategy",
        [
            StabilizationStrategy.WEIGHTED_VOTE,
            StabilizationStrategy.EMA,
            StabilizationStrategy.MAJORITY,
        ],
    )
    def test_every_strategy_converges_on_a_consistent_person(self, strategy) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(
                strategy=strategy, minimum_recognized_frames=3, history_size=10
            ),
            MATCHING,
        )
        state = IdentityHistory()
        results = feed(stabilizer, state, [recognized("john", 0.85)] * 8)
        assert results[-1].identity_id == "john"

    def test_weighted_vote_favours_recent_observations(self) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(
                strategy=StabilizationStrategy.WEIGHTED_VOTE,
                minimum_recognized_frames=2,
                switch_margin=0.0,
                decay=0.5,
                history_size=10,
            ),
            MATCHING,
        )
        state = IdentityHistory()
        feed(stabilizer, state, [recognized("john", 0.60)] * 4 + [recognized("jane", 0.90)] * 3)
        assert state.current_identity == "jane"

    def test_disabled_stabilizer_passes_results_through(self) -> None:
        stabilizer = IdentityStabilizer(IdentityStabilityConfig(enabled=False), MATCHING)
        state = IdentityHistory()
        raw = recognized("john", 0.9)
        assert stabilizer.observe(state, raw, 0) is raw

    def test_history_window_is_bounded(self) -> None:
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(history_size=5, minimum_recognized_frames=2), MATCHING
        )
        state = IdentityHistory()
        feed(stabilizer, state, [recognized("john", 0.8)] * 20)
        assert len(state.history) == 5


class TestTrackManager:
    def test_track_creation_and_refresh(self) -> None:
        manager = TrackManager(TrackingConfig(), MATCHING)
        state, is_new = manager.touch(7, frame_index=0, timestamp=1.0)
        assert is_new and state.frames_seen == 1

        state, is_new = manager.touch(7, frame_index=1, timestamp=1.1)
        assert not is_new and state.frames_seen == 2
        assert manager.active_count == 1

    def test_synthetic_ids_are_negative_and_unique(self) -> None:
        manager = TrackManager(TrackingConfig(), MATCHING)
        ids = {manager.synthetic_id() for _ in range(5)}
        assert len(ids) == 5
        assert all(i < 0 for i in ids)

    def test_tracks_expire_after_the_timeout(self) -> None:
        manager = TrackManager(TrackingConfig(lost_track_timeout=5), MATCHING)
        manager.touch(1, 0, 0.0)
        assert manager.expire(5) == []          # still within the timeout
        expired = manager.expire(7)
        assert [s.track_id for s in expired] == [1]
        assert manager.active_count == 0

    def test_identification_and_change_produce_transitions(self) -> None:
        manager = TrackManager(
            TrackingConfig(
                identity_stability=IdentityStabilityConfig(
                    minimum_recognized_frames=1, switch_margin=0.0
                )
            ),
            MATCHING,
        )
        state, _ = manager.touch(1, 0, 0.0)
        _, first = manager.apply_recognition(state, recognized("john", 0.9), 0)
        assert first is not None and first.kind == "identified"

        _, second = manager.apply_recognition(state, recognized("jane", 0.95), 1)
        assert second is not None and second.kind == "identity_changed"
        assert second.previous_identity == "john"

    def test_reset_clears_everything(self) -> None:
        manager = TrackManager(TrackingConfig(), MATCHING)
        manager.touch(1, 0, 0.0)
        manager.touch(2, 0, 0.0)
        assert len(manager.reset()) == 2
        assert manager.active_count == 0


class TestReIDScheduling:
    """``reid_interval`` must save work without costing recognitions."""

    def test_every_frame_when_the_interval_is_one(self) -> None:
        manager = TrackManager(TrackingConfig(), MATCHING)
        state, _ = manager.touch(1, 0, 0.0)
        assert manager.needs_reid(state, 0, PerformanceConfig(reid_interval=1))

    def test_a_new_track_is_always_embedded(self) -> None:
        manager = TrackManager(TrackingConfig(), MATCHING)
        state, _ = manager.touch(1, 0, 0.0)
        assert manager.needs_reid(state, 0, PerformanceConfig(reid_interval=5))

    def test_an_unknown_track_is_re_checked_every_frame(self) -> None:
        import numpy as np

        manager = TrackManager(TrackingConfig(), MATCHING)
        state, _ = manager.touch(1, 0, 0.0)
        state.last_embedding = np.ones(4, dtype=np.float32)
        state.last_reid_frame = 0
        state.frames_seen = 10
        performance = PerformanceConfig(reid_interval=5, force_reid_when_unknown=True)
        assert manager.needs_reid(state, 1, performance)

    def test_a_stable_identified_track_is_skipped_between_intervals(self) -> None:
        import numpy as np

        manager = TrackManager(TrackingConfig(), MATCHING)
        state, _ = manager.touch(1, 0, 0.0)
        state.last_embedding = np.ones(4, dtype=np.float32)
        state.last_reid_frame = 10
        state.frames_seen = 20
        state.identity.current_identity = "john"
        performance = PerformanceConfig(reid_interval=5)

        assert not manager.needs_reid(state, 12, performance)   # 2 frames since
        assert manager.needs_reid(state, 15, performance)       # interval elapsed

    def test_disabling_the_cache_forces_every_frame(self) -> None:
        import numpy as np

        manager = TrackManager(TrackingConfig(), MATCHING)
        state, _ = manager.touch(1, 0, 0.0)
        state.last_embedding = np.ones(4, dtype=np.float32)
        state.last_reid_frame = 10
        state.identity.current_identity = "john"
        assert manager.needs_reid(
            state, 11, PerformanceConfig(reid_interval=5, embedding_cache=False)
        )
