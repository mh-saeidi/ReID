"""Recognition scheduling, the track-local cache and quality gating.

The scheduler decides *when* to look, never *what the answer is*, so these
tests assert both: that it saves work, and that it cannot change a recognition
outcome.
"""

from __future__ import annotations

import pytest

from src.config.schema import (
    FaceConfig,
    RecognitionSchedulerConfig,
    RuntimeQualityConfig,
    SchedulerState,
)
from src.core.types import RecognitionStatus
from src.face.quality_gate import FaceQualityGate
from src.face.types import FaceQuality
from src.reid.preprocess import CropStatus
from src.tracking.recognition_cache import RecognitionCache
from src.tracking.scheduler import RecognitionScheduler
from src.tracking.tracker import TrackState
from tests.conftest import make_face


def make_track(track_id: int = 1, frame: int = 0) -> TrackState:
    state = TrackState(
        track_id=track_id,
        first_seen_frame=frame,
        last_seen_frame=frame,
        first_seen_time=0.0,
        last_seen_time=0.0,
        frames_seen=1,
    )
    state.recognition_cache.track_id = track_id
    return state


def advance(state: TrackState, frames: int = 1) -> None:
    state.last_seen_frame += frames
    state.frames_seen += frames


def settle(state: TrackState, status: RecognitionStatus, frame: int,
           identity: str | None = "john", similarity: float = 0.9) -> None:
    """Record a completed recognition pass."""
    state.recognition_cache.note_recognition(frame, 0.0)
    state.recognition_cache.note_result(status, identity, similarity)


class TestSchedulerStates:
    def test_a_new_track_is_recognised_immediately(self) -> None:
        scheduler = RecognitionScheduler(RecognitionSchedulerConfig())
        state = make_track()
        decision = scheduler.decide(state, 0)
        assert decision.should_recognize
        assert decision.state is SchedulerState.NEW_TRACK
        assert decision.forced

    def test_a_stable_recognised_track_uses_the_long_interval(self) -> None:
        config = RecognitionSchedulerConfig(stable_interval=6)
        scheduler = RecognitionScheduler(config)
        state = make_track()
        settle(state, RecognitionStatus.RECOGNIZED, frame=0)

        for offset in range(1, 6):
            advance(state)
            decision = scheduler.decide(state, offset)
            assert not decision.should_recognize, f"frame {offset}"
            assert decision.state is SchedulerState.RECOGNIZED

        advance(state)
        assert scheduler.decide(state, 6).should_recognize

    def test_low_confidence_is_re_checked_more_often(self) -> None:
        config = RecognitionSchedulerConfig(stable_interval=10, low_confidence_interval=2)
        scheduler = RecognitionScheduler(config)
        state = make_track()
        settle(state, RecognitionStatus.LOW_CONFIDENCE, frame=0, similarity=0.5)

        advance(state)
        assert not scheduler.decide(state, 1).should_recognize
        advance(state)
        decision = scheduler.decide(state, 2)
        assert decision.should_recognize
        assert decision.state is SchedulerState.LOW_CONFIDENCE

    def test_no_face_never_runs_the_encoder_but_keeps_looking(self) -> None:
        """The face detector is cheap; the encoder is not and has nothing to do."""
        config = RecognitionSchedulerConfig(no_face_interval=2)
        scheduler = RecognitionScheduler(config)
        state = make_track()
        settle(state, RecognitionStatus.NO_FACE, frame=0, identity=None, similarity=0.0)

        advance(state)
        assert not scheduler.decide(state, 1).should_recognize
        advance(state)
        decision = scheduler.decide(state, 2)
        assert decision.should_recognize
        assert decision.state is SchedulerState.NO_FACE

    def test_a_recovered_track_is_forced(self) -> None:
        """A reused track ID may now be a different person."""
        scheduler = RecognitionScheduler(RecognitionSchedulerConfig(stable_interval=100))
        state = make_track()
        settle(state, RecognitionStatus.RECOGNIZED, frame=0)
        advance(state, 5)
        state.recognition_cache.mark_lost()

        decision = scheduler.decide(state, 5)
        assert decision.should_recognize
        assert decision.forced

    def test_an_identity_change_forces_confirmation(self) -> None:
        scheduler = RecognitionScheduler(RecognitionSchedulerConfig(stable_interval=100))
        state = make_track()
        settle(state, RecognitionStatus.RECOGNIZED, frame=0, identity="john")
        advance(state)
        state.recognition_cache.note_result(
            RecognitionStatus.RECOGNIZED, "jane", 0.9
        )
        assert state.recognition_cache.identity_changed
        assert scheduler.decide(state, 1).should_recognize

    def test_a_disabled_scheduler_recognises_every_frame(self) -> None:
        scheduler = RecognitionScheduler(RecognitionSchedulerConfig(enabled=False))
        state = make_track()
        settle(state, RecognitionStatus.RECOGNIZED, frame=0)
        for offset in range(1, 5):
            advance(state)
            assert scheduler.decide(state, offset).should_recognize


class TestUnknownBackoff:
    def test_the_interval_grows_while_nobody_matches(self) -> None:
        """A passer-by who will never match must not pin the encoder."""
        config = RecognitionSchedulerConfig(
            unknown_interval=2, unknown_backoff_factor=2.0,
            unknown_backoff_max_interval=32,
        )
        scheduler = RecognitionScheduler(config)
        state = make_track()
        cache = state.recognition_cache

        intervals = []
        for _ in range(5):
            cache.note_result(RecognitionStatus.UNKNOWN, None, 0.1)
            cache.last_recognition_frame = 0
            state.last_seen_frame = 0
            state.frames_seen = 10
            intervals.append(scheduler._interval_for(SchedulerState.UNKNOWN, cache))
        assert intervals == sorted(intervals)
        assert intervals[-1] > intervals[0]

    def test_the_backoff_is_capped(self) -> None:
        config = RecognitionSchedulerConfig(
            unknown_interval=2, unknown_backoff_factor=3.0,
            unknown_backoff_max_interval=20,
        )
        scheduler = RecognitionScheduler(config)
        cache = RecognitionCache(consecutive_unknown=50)
        assert scheduler._interval_for(SchedulerState.UNKNOWN, cache) == 20

    def test_the_backoff_can_be_disabled(self) -> None:
        config = RecognitionSchedulerConfig(
            unknown_interval=3, unknown_backoff_enabled=False
        )
        scheduler = RecognitionScheduler(config)
        cache = RecognitionCache(consecutive_unknown=10)
        assert scheduler._interval_for(SchedulerState.UNKNOWN, cache) == 3

    def test_a_match_resets_the_backoff(self) -> None:
        cache = RecognitionCache(consecutive_unknown=7)
        cache.note_result(RecognitionStatus.RECOGNIZED, "john", 0.95)
        assert cache.consecutive_unknown == 0

    def test_a_hidden_face_does_not_advance_the_unknown_backoff(self) -> None:
        """Not seeing a face is no evidence that the person is unregistered."""
        cache = RecognitionCache()
        for _ in range(5):
            cache.note_result(RecognitionStatus.NO_FACE, None, 0.0)
        assert cache.consecutive_unknown == 0
        assert cache.consecutive_no_face == 5


class TestFrameBudget:
    def test_the_budget_bounds_recognitions_per_frame(self) -> None:
        config = RecognitionSchedulerConfig(
            max_recognitions_per_frame=2, stable_interval=1
        )
        scheduler = RecognitionScheduler(config)
        scheduler.begin_frame(10)
        tracks = []
        for index in range(5):
            state = make_track(index)
            settle(state, RecognitionStatus.RECOGNIZED, frame=0)
            state.last_seen_frame = 10
            state.frames_seen = 20
            tracks.append(state)
        allowed = [scheduler.decide(s, 10).should_recognize for s in tracks]
        assert sum(allowed) == 2

    def test_a_forced_pass_ignores_the_budget(self) -> None:
        """A new track must never be starved by a crowd of stable ones."""
        config = RecognitionSchedulerConfig(max_recognitions_per_frame=1)
        scheduler = RecognitionScheduler(config)
        scheduler.begin_frame(5)
        stable = make_track(1)
        settle(stable, RecognitionStatus.RECOGNIZED, frame=0)
        stable.last_seen_frame = 5
        stable.frames_seen = 20
        scheduler.decide(stable, 5)

        fresh = make_track(2, frame=5)
        assert scheduler.decide(fresh, 5).should_recognize


class TestSchedulerStats:
    def test_attempts_and_skips_are_counted(self) -> None:
        scheduler = RecognitionScheduler(RecognitionSchedulerConfig(stable_interval=5))
        state = make_track()
        scheduler.decide(state, 0)
        settle(state, RecognitionStatus.RECOGNIZED, frame=0)
        for offset in range(1, 5):
            advance(state)
            scheduler.decide(state, offset)
        stats = scheduler.stats.to_dict()
        assert stats["recognition_attempts"] >= 1
        assert stats["recognition_skips"] >= 1
        assert 0.0 < stats["skip_rate"] < 1.0

    def test_reset_clears_the_counters(self) -> None:
        scheduler = RecognitionScheduler(RecognitionSchedulerConfig())
        scheduler.decide(make_track(), 0)
        scheduler.reset()
        assert scheduler.stats.attempts == 0


class TestRecognitionCache:
    def test_the_cache_holds_no_embedding(self) -> None:
        """Storing one would invite reusing it for a face that is not visible."""
        cache = RecognitionCache()
        assert not hasattr(cache, "embedding")
        assert not hasattr(cache, "last_embedding")
        assert "embedding" not in {
            k for k in cache.to_dict() if k.endswith("embedding")
        }

    def test_provenance_is_recorded(self) -> None:
        cache = RecognitionCache()
        cache.note_embedding(frame_index=12, dimension=512)
        assert cache.last_embedding_frame == 12
        assert cache.last_embedding_dimension == 512
        assert cache.embeddings_computed == 1

    def test_quality_rejections_are_recorded_with_a_reason(self) -> None:
        cache = RecognitionCache()
        cache.note_quality_rejection("face is 20px (minimum 36px)")
        assert cache.quality_rejections == 1
        assert "20px" in cache.last_quality_reason

    def test_face_geometry_is_remembered_for_display(self) -> None:
        cache = RecognitionCache()
        face = make_face((100, 100, 180, 200))
        cache.note_result(RecognitionStatus.RECOGNIZED, "john", 0.9, face=face)
        assert cache.last_face_bbox is not None
        assert cache.last_face_score == pytest.approx(0.95)

    def test_the_cache_serialises_for_debug_output(self) -> None:
        cache = RecognitionCache(track_id=7)
        cache.note_result(RecognitionStatus.UNKNOWN, None, 0.1)
        data = cache.to_dict()
        assert data["track_id"] == 7
        assert data["last_status"] == "unknown"
        assert data["consecutive_unknown"] == 1


class TestQualityGate:
    def test_a_good_face_passes(self) -> None:
        gate = FaceQualityGate(FaceConfig())
        assert gate.check_detection(make_face((100, 100, 200, 230))).accepted

    def test_a_small_face_is_rejected_not_guessed(self) -> None:
        gate = FaceQualityGate(FaceConfig(min_face_size=40))
        result = gate.check_detection(make_face((100, 100, 120, 125)))
        assert not result.accepted
        assert result.status is CropStatus.FACE_TOO_SMALL
        assert result.quality is FaceQuality.TOO_SMALL

    def test_low_detector_confidence_is_rejected(self) -> None:
        config = FaceConfig(
            runtime_quality=RuntimeQualityConfig(min_detector_confidence=0.9)
        )
        gate = FaceQualityGate(config)
        result = gate.check_detection(make_face((100, 100, 200, 230), score=0.5))
        assert not result.accepted
        assert result.status is CropStatus.LOW_QUALITY

    def test_thresholds_inherit_from_the_face_section(self) -> None:
        """Enabling the gate must not silently change existing behaviour."""
        config = FaceConfig(min_face_size=55)
        gate = FaceQualityGate(config)
        assert not gate.check_detection(make_face((100, 100, 150, 160))).accepted
        assert gate.check_detection(make_face((100, 100, 200, 230))).accepted

    def test_an_explicit_override_wins_over_the_inherited_value(self) -> None:
        config = FaceConfig(
            min_face_size=55, runtime_quality=RuntimeQualityConfig(min_face_size=20)
        )
        gate = FaceQualityGate(config)
        assert gate.check_detection(make_face((100, 100, 150, 160))).accepted

    def test_a_disabled_gate_accepts_everything(self) -> None:
        gate = FaceQualityGate(FaceConfig(runtime_quality=RuntimeQualityConfig(enabled=False)))
        assert not gate.enabled
        assert gate.check_detection(make_face((100, 100, 110, 112))).accepted

    def test_landmark_skew_catches_an_unreliable_fit(self) -> None:
        import numpy as np

        config = FaceConfig(runtime_quality=RuntimeQualityConfig(max_landmark_skew=0.1))
        gate = FaceQualityGate(config)
        face = make_face((100, 100, 200, 230))
        face.landmarks[1] = face.landmarks[0] + np.array([40.0, 40.0], dtype=np.float32)
        result = gate.check_detection(face)
        assert not result.accepted
        assert result.quality is FaceQuality.EXTREME_POSE

    def test_the_gate_never_produces_an_identity(self) -> None:
        """Every rejection path yields a semantic state, never a name."""
        gate = FaceQualityGate(FaceConfig(min_face_size=40))
        result = gate.check_detection(make_face((100, 100, 110, 112)))
        assert not hasattr(result, "identity_id")
        assert result.status in (
            CropStatus.FACE_TOO_SMALL, CropStatus.LOW_QUALITY, CropStatus.NO_FACE
        )

    def test_pixel_checks_are_skipped_when_no_threshold_needs_them(self) -> None:
        gate = FaceQualityGate(FaceConfig(min_blur_variance=0.0))
        assert gate.needs_pixels is False

    def test_a_blurry_chip_is_rejected(self) -> None:
        import numpy as np

        config = FaceConfig(runtime_quality=RuntimeQualityConfig(min_sharpness=50.0))
        gate = FaceQualityGate(config)
        flat = np.full((112, 112, 3), 128, dtype=np.uint8)
        result = gate.check_chip(flat, make_face())
        assert not result.accepted
        assert result.quality is FaceQuality.BLURRY


class TestBatchCapability:
    """The configuration expresses intent; the model decides what is possible."""

    def test_a_fixed_batch_model_disables_batching(self, base_config_dict, tmp_path) -> None:
        from src.config.loader import config_from_dict
        from src.config.paths import ProjectPaths
        from src.identity.gallery import IdentityGallery
        from src.identity.matcher import IdentityMatcher
        from src.pipeline.processor import ReIDPipeline
        from src.reid.preprocess import PersonCropPreprocessor
        from tests.conftest import FakeDetector, FakeEncoder

        base_config_dict["recognition"] = {"mode": "person_reid",
                                           "batch": {"enabled": True, "max_size": 8}}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)

        encoder = FakeEncoder()
        # Simulate a graph exported with a fixed batch dimension.
        object.__setattr__(encoder._info, "max_batch_size", 1)

        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.gallery_dir, paths.embeddings_dir, paths.metadata_dir)
        pipeline = ReIDPipeline(
            config, FakeDetector([]), encoder, PersonCropPreprocessor(),
            IdentityGallery(config, paths), IdentityMatcher(config.matching),
            use_tracking=False,
        )
        assert pipeline._batch_enabled is False
        assert pipeline._max_batch == 1

    def test_the_configured_size_is_clamped_to_the_model(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.config.loader import config_from_dict
        from src.config.paths import ProjectPaths
        from src.identity.gallery import IdentityGallery
        from src.identity.matcher import IdentityMatcher
        from src.pipeline.processor import ReIDPipeline
        from src.reid.preprocess import PersonCropPreprocessor
        from tests.conftest import FakeDetector, FakeEncoder

        base_config_dict["recognition"] = {"mode": "person_reid",
                                           "batch": {"enabled": True, "max_size": 32}}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        encoder = FakeEncoder()
        object.__setattr__(encoder._info, "max_batch_size", 4)

        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.gallery_dir, paths.embeddings_dir, paths.metadata_dir)
        pipeline = ReIDPipeline(
            config, FakeDetector([]), encoder, PersonCropPreprocessor(),
            IdentityGallery(config, paths), IdentityMatcher(config.matching),
            use_tracking=False,
        )
        assert pipeline._max_batch == 4
        assert pipeline._batch_enabled is True

    def test_an_unconstrained_model_uses_the_configured_size(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.config.loader import config_from_dict
        from src.config.paths import ProjectPaths
        from src.identity.gallery import IdentityGallery
        from src.identity.matcher import IdentityMatcher
        from src.pipeline.processor import ReIDPipeline
        from src.reid.preprocess import PersonCropPreprocessor
        from tests.conftest import FakeDetector, FakeEncoder

        base_config_dict["recognition"] = {"mode": "person_reid",
                                           "batch": {"enabled": True, "max_size": 8}}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        encoder = FakeEncoder()   # max_batch_size defaults to 0 = unconstrained

        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.gallery_dir, paths.embeddings_dir, paths.metadata_dir)
        pipeline = ReIDPipeline(
            config, FakeDetector([]), encoder, PersonCropPreprocessor(),
            IdentityGallery(config, paths), IdentityMatcher(config.matching),
            use_tracking=False,
        )
        assert pipeline._max_batch == 8
        assert pipeline._batch_enabled is True
