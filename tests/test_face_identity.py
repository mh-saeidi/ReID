"""Face identity engine: gallery, matching, decisions, state machine, adaptation.

The tests that matter most here are the ones asserting what the system must
*never* do: name a masked stranger, force an unknown onto the nearest identity,
treat a cosine as a probability, or let a false recognition contaminate the
gallery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.face.quality import FaceQualityScore, QualityWeights, assess_face_quality
from src.face.visibility import (
    FaceVisibility,
    VisibilityReport,
    estimate_pose,
)
from src.identity.adaptation import AdaptationConfig, AdaptationOutcome, GalleryAdapter
from src.identity.calibration import (
    CalibrationSample,
    fit_calibration,
    operating_points,
    separation_threshold,
)
from src.identity.decision import IdentityDecisionEngine, MatchingThresholds
from src.identity.face_gallery import FaceEmbeddingRecord, FaceGallery, FaceIdentity
from src.identity.state_machine import (
    IdentityStateMachine,
    StabilityConfig,
    TrackIdentityState,
)
from src.identity.types import (
    FailureReason,
    IdentityObservation,
    IdentityStatus,
    TrackIdentityPhase,
)
from src.reid.encoder import l2_normalize
from tests.conftest import make_face


def vector(seed: int, dimension: int = 64) -> np.ndarray:
    return l2_normalize(np.random.default_rng(seed).normal(size=dimension).astype(np.float32))


@pytest.fixture
def gallery(tmp_path: Path) -> FaceGallery:
    store = FaceGallery(tmp_path / "people")
    for index, pid in enumerate(("alice", "bob", "carol")):
        identity = FaceIdentity(id=pid, name=pid.title(), title="Tester")
        identity.embeddings["reference"] = FaceEmbeddingRecord(
            "reference", vector(index), quality=0.9
        )
        store.save(identity)
    return store


def quality_score(overall: float = 0.9) -> FaceQualityScore:
    return FaceQualityScore(
        overall=overall, resolution=overall, sharpness=overall, exposure=overall,
        pose=overall, landmarks=overall, visibility=overall, interocular_px=48.0,
    )


class TestFaceGallery:
    def test_one_reference_per_person_is_enough(self, gallery) -> None:
        assert len(gallery.active) == 3
        assert gallery.embedding_count == 3
        assert all(i.reference is not None for i in gallery.active)

    def test_a_person_matches_their_own_reference(self, gallery) -> None:
        result = gallery.match(vector(0))
        assert result.best_identity_id == "alice"
        assert result.best_similarity == pytest.approx(1.0, abs=1e-5)
        assert result.matched_embedding_key == "reference"

    def test_matching_reports_the_runner_up_and_margin(self, gallery) -> None:
        result = gallery.match(vector(0))
        assert result.runner_up_id in ("bob", "carol")
        assert result.margin > 0

    def test_an_empty_gallery_is_not_an_error(self, tmp_path: Path) -> None:
        empty = FaceGallery(tmp_path / "none")
        assert empty.is_empty
        assert empty.match(vector(9)).best_identity_id is None

    def test_a_person_can_hold_several_embeddings(self, gallery) -> None:
        gallery.add_live_embedding("alice", vector(50), quality=0.9, similarity=0.8)
        gallery.add_live_embedding("alice", vector(51), quality=0.9, similarity=0.8)
        assert gallery.get("alice").live_count == 2
        assert gallery.embedding_count == 5

    def test_the_best_embedding_wins_not_the_average(self, gallery) -> None:
        """Averaging across stored views would blur exactly the variation they
        were stored to capture."""
        live = vector(77)
        gallery.add_live_embedding("alice", live, quality=0.9, similarity=0.8)
        result = gallery.match(live)
        assert result.best_identity_id == "alice"
        assert result.best_similarity == pytest.approx(1.0, abs=1e-5)
        assert result.matched_embedding_key.startswith("live_")

    def test_the_gallery_round_trips_through_disk(self, gallery, tmp_path) -> None:
        gallery.add_live_embedding("bob", vector(60), quality=0.8, similarity=0.7)
        reloaded = FaceGallery(tmp_path / "people")
        assert reloaded.load() == 3
        assert reloaded.get("bob").live_count == 1
        assert reloaded.match(vector(1)).best_identity_id == "bob"

    def test_removal_deletes_the_person_directory(self, gallery) -> None:
        assert gallery.remove("carol")
        assert gallery.get("carol") is None
        assert not (gallery.root / "carol").exists()

    def test_capacity_evicts_the_weakest_live_sample_not_the_reference(
        self, tmp_path: Path
    ) -> None:
        store = FaceGallery(tmp_path / "people", max_live_per_identity=2)
        identity = FaceIdentity(id="alice", name="Alice")
        identity.embeddings["reference"] = FaceEmbeddingRecord(
            "reference", vector(0), quality=0.95
        )
        store.save(identity)
        store.add_live_embedding("alice", vector(10), quality=0.60, similarity=0.8)
        store.add_live_embedding("alice", vector(11), quality=0.80, similarity=0.8)
        store.add_live_embedding("alice", vector(12), quality=0.90, similarity=0.8)

        entry = store.get("alice")
        assert entry.reference is not None, "the passport reference must never be evicted"
        assert entry.live_count == 2
        assert min(r.quality for r in entry.embeddings.values() if not r.is_reference) >= 0.8

    def test_a_dimension_mismatch_is_actionable(self, gallery) -> None:
        from src.core.exceptions import GalleryError

        with pytest.raises(GalleryError, match="rebuild the gallery"):
            gallery.match(np.ones(128, dtype=np.float32))


class TestOpenSetRejection:
    def test_an_unknown_face_is_not_forced_onto_the_nearest_identity(
        self, gallery
    ) -> None:
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.5, min_quality=0.0)
        )
        decision = engine.decide(
            detector_confidence=0.95,
            embedding=vector(999),           # nobody in the gallery
            quality=quality_score(),
            visibility=FaceVisibility.FULL_FACE,
        )
        assert decision.status is IdentityStatus.UNKNOWN
        assert decision.identity_id is None
        assert decision.failure is FailureReason.LOW_SIMILARITY
        # The best candidate is still reported, for diagnosis only.
        assert decision.match.best_identity_id is not None

    def test_an_ambiguous_match_is_refused(self, tmp_path: Path) -> None:
        """Two registered people scoring comparably is a coin flip, not a match."""
        store = FaceGallery(tmp_path / "people")
        base = vector(0)
        twin = l2_normalize(base + 0.02 * vector(1))
        for pid, embedding in (("alice", base), ("alice_twin", twin)):
            identity = FaceIdentity(id=pid, name=pid)
            identity.embeddings["reference"] = FaceEmbeddingRecord("reference", embedding)
            store.save(identity)

        engine = IdentityDecisionEngine(
            store, MatchingThresholds(full=0.3, ambiguity_margin=0.20, min_quality=0.0)
        )
        decision = engine.decide(
            detector_confidence=0.9, embedding=base,
            quality=quality_score(), visibility=FaceVisibility.FULL_FACE,
        )
        assert decision.status is IdentityStatus.UNKNOWN
        assert decision.failure is FailureReason.AMBIGUOUS_IDENTITY

    def test_an_empty_gallery_reports_why(self, tmp_path: Path) -> None:
        engine = IdentityDecisionEngine(
            FaceGallery(tmp_path / "x"), MatchingThresholds(min_quality=0.0)
        )
        decision = engine.decide(
            detector_confidence=0.9, embedding=vector(3), quality=quality_score()
        )
        assert decision.failure is FailureReason.EMPTY_GALLERY


class TestQuantitiesStaySeparate:
    def test_similarity_is_never_reported_as_confidence(self, gallery) -> None:
        """A cosine is not a probability; without calibration none is reported."""
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.5, min_quality=0.0), calibration=None
        )
        decision = engine.decide(
            detector_confidence=0.91, face_detection_confidence=0.88,
            embedding=vector(0), quality=quality_score(0.77),
            visibility=FaceVisibility.FULL_FACE,
        )
        assert decision.identity_confidence is None
        assert decision.face_similarity == pytest.approx(1.0, abs=1e-5)

    def test_every_quantity_is_carried_separately(self, gallery) -> None:
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.5, min_quality=0.0)
        )
        decision = engine.decide(
            detector_confidence=0.91, face_detection_confidence=0.88,
            embedding=vector(0), quality=quality_score(0.77),
            visibility=FaceVisibility.FULL_FACE,
        )
        data = decision.to_dict()
        assert data["detector_confidence"] == pytest.approx(0.91)
        assert data["face_detection_confidence"] == pytest.approx(0.88)
        assert data["face_quality"] == pytest.approx(0.77)
        assert data["identity_confidence"] is None
        # They must be distinct keys, never merged into one "confidence".
        assert "confidence" not in data

    def test_a_calibrated_confidence_appears_only_with_calibration(self, gallery) -> None:
        samples = [
            CalibrationSample(float(s), True, condition="full")
            for s in np.random.default_rng(0).normal(0.7, 0.08, 40)
        ] + [
            CalibrationSample(float(s), False, condition="full")
            for s in np.random.default_rng(1).normal(0.05, 0.06, 80)
        ]
        model = fit_calibration(samples)
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.4, min_quality=0.0), calibration=model
        )
        decision = engine.decide(
            detector_confidence=0.9, embedding=vector(0),
            quality=quality_score(), visibility=FaceVisibility.FULL_FACE,
        )
        assert decision.identity_confidence is not None
        assert 0.0 <= decision.identity_confidence <= 1.0


class TestVisibilityAndThresholds:
    def test_a_masked_face_uses_its_own_threshold_family(self, gallery) -> None:
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.8, masked=0.3, min_quality=0.0)
        )
        full = engine.resolve_threshold(FaceVisibility.FULL_FACE)
        masked = engine.resolve_threshold(FaceVisibility.MASKED)
        assert full[0] == 0.8 and full[1] == "full"
        assert masked[0] == 0.3 and masked[1] == "masked"

    def test_visibility_classes_map_to_threshold_families(self) -> None:
        assert FaceVisibility.FULL_FACE.threshold_key == "full"
        assert FaceVisibility.MASKED.threshold_key == "masked"
        assert FaceVisibility.PARTIAL_FACE.threshold_key == "partial"

    def test_heavy_occlusion_is_not_usable(self) -> None:
        assert not FaceVisibility.HEAVILY_OCCLUDED.is_usable
        assert not FaceVisibility.NO_USABLE_FACE.is_usable
        assert FaceVisibility.MASKED.is_usable

    def test_pose_estimation_detects_a_turned_head(self) -> None:
        frontal = make_face((100, 100, 200, 230))
        yaw, _ = estimate_pose(frontal.landmarks)
        assert abs(yaw) < 15

        turned = make_face((100, 100, 200, 230))
        turned.landmarks[2][0] += 35        # nose displaced towards one eye
        yaw_turned, _ = estimate_pose(turned.landmarks)
        assert abs(yaw_turned) > abs(yaw)


class TestFaceQuality:
    def test_quality_is_continuous_not_binary(self) -> None:
        import cv2

        rng = np.random.default_rng(4)
        sharp = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
        face = make_face((100, 100, 200, 230))

        good = assess_face_quality(sharp, face)
        poor = assess_face_quality(blurred, face)
        assert 0.0 <= poor.overall < good.overall <= 1.0
        assert poor.sharpness < good.sharpness

    def test_the_weakest_component_is_identified(self) -> None:
        import cv2

        rng = np.random.default_rng(5)
        chip = cv2.GaussianBlur(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8),
                                (21, 21), 0)
        score = assess_face_quality(chip, make_face((100, 100, 200, 230)))
        assert score.weakest_component == "sharpness"

    def test_a_tiny_face_scores_low_on_resolution(self) -> None:
        rng = np.random.default_rng(6)
        chip = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        tiny = assess_face_quality(chip, make_face((100, 100, 118, 122)))
        large = assess_face_quality(chip, make_face((100, 100, 220, 250)))
        assert tiny.resolution < large.resolution

    def test_visibility_lowers_the_score(self) -> None:
        rng = np.random.default_rng(7)
        chip = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        face = make_face((100, 100, 200, 230))
        full = assess_face_quality(chip, face)
        masked = assess_face_quality(
            chip, face,
            VisibilityReport(FaceVisibility.MASKED, 0.9, 0.0, 0.9, 0.0, 0.0),
        )
        assert masked.overall < full.overall

    def test_weights_are_normalised(self) -> None:
        weights = QualityWeights(resolution=2.0, sharpness=2.0).normalised()
        total = sum(
            getattr(weights, f)
            for f in ("resolution", "sharpness", "exposure", "pose",
                      "landmarks", "visibility")
        )
        assert total == pytest.approx(1.0)


class TestStateMachine:
    @pytest.fixture
    def machine(self) -> IdentityStateMachine:
        return IdentityStateMachine(
            StabilityConfig(min_confirmation_frames=3, occlusion_hold_frames=5,
                            min_evidence_weight=0.5)
        )

    @staticmethod
    def observation(frame: int, identity: str | None, similarity: float,
                    quality: float = 0.9) -> IdentityObservation:
        return IdentityObservation(frame, float(frame), identity, similarity,
                                   quality, FaceVisibility.FULL_FACE)

    def test_an_identity_requires_repeated_evidence(self, machine) -> None:
        state = TrackIdentityState(track_id=1)
        phases = [
            machine.observe_face(state, self.observation(f, "john", 0.8), accepted=True)
            for f in range(4)
        ]
        assert phases[0] is TrackIdentityPhase.UNCONFIRMED
        assert phases[-1] is TrackIdentityPhase.RECOGNIZED

    def test_one_good_frame_never_names_anybody(self, machine) -> None:
        state = TrackIdentityState(track_id=1)
        machine.observe_face(state, self.observation(0, "john", 0.99), accepted=True)
        assert state.current_identity is None

    def test_a_confirmed_identity_survives_a_short_occlusion(self, machine) -> None:
        state = TrackIdentityState(track_id=1)
        for f in range(4):
            machine.observe_face(state, self.observation(f, "john", 0.8), accepted=True)
        assert state.current_identity == "john"

        for f in range(4, 8):
            phase = machine.observe_no_face(state, f)
        assert phase is TrackIdentityPhase.TEMPORARILY_OCCLUDED
        assert state.current_identity == "john"

    def test_the_occlusion_hold_is_bounded(self, machine) -> None:
        state = TrackIdentityState(track_id=1)
        for f in range(4):
            machine.observe_face(state, self.observation(f, "john", 0.8), accepted=True)
        for f in range(4, 30):
            phase = machine.observe_no_face(state, f)
        assert phase is TrackIdentityPhase.UNKNOWN
        assert state.current_identity is None

    def test_a_new_masked_track_is_never_named(self, machine) -> None:
        """The mandatory rule: tracking may preserve an identity, never create one."""
        state = TrackIdentityState(track_id=99, first_seen_frame=50, last_seen_frame=50)
        for f in range(50, 90):
            phase = machine.observe_no_face(state, f)
        assert state.current_identity is None
        assert phase is not TrackIdentityPhase.RECOGNIZED

    def test_switching_identity_needs_a_clear_margin(self) -> None:
        machine = IdentityStateMachine(
            StabilityConfig(min_confirmation_frames=2, switch_margin=0.25,
                            min_evidence_weight=0.5)
        )
        state = TrackIdentityState(track_id=1)
        for f in range(4):
            machine.observe_face(state, self.observation(f, "john", 0.80), accepted=True)
        assert state.current_identity == "john"

        for f in range(4, 8):
            machine.observe_face(state, self.observation(f, "jane", 0.85), accepted=True)
        assert state.current_identity == "john", "a marginal rival must not take over"
        assert state.identity_switches == 0

    def test_evidence_is_quality_weighted(self, machine) -> None:
        state = TrackIdentityState(track_id=1)
        for f in range(5):
            machine.observe_face(
                state, self.observation(f, "john", 0.9, quality=0.95), accepted=True
            )
        for f in range(5, 10):
            machine.observe_face(
                state, self.observation(f, "jane", 0.9, quality=0.05), accepted=True
            )
        candidates = dict(state.evidence.ranked_candidates())
        assert state.evidence.weights_by_identity["john"] > (
            state.evidence.weights_by_identity["jane"] * 5
        )
        assert candidates["john"] > 0

    def test_a_hidden_face_is_not_counted_against_the_identity(self, machine) -> None:
        state = TrackIdentityState(track_id=1)
        for f in range(4):
            machine.observe_face(state, self.observation(f, "john", 0.8), accepted=True)
        before = state.evidence.unknown_count
        for f in range(4, 8):
            machine.observe_no_face(state, f)
        assert state.evidence.unknown_count == before


class TestTemporarilyMaintained:
    def test_an_occluded_confirmed_track_keeps_its_name(self, gallery) -> None:
        engine = IdentityDecisionEngine(
            gallery,
            MatchingThresholds(full=0.5, min_quality=0.0),
            stability=StabilityConfig(min_confirmation_frames=2,
                                      occlusion_hold_frames=10,
                                      min_evidence_weight=0.5),
        )
        state = TrackIdentityState(track_id=1)
        for frame in range(4):
            engine.decide(
                detector_confidence=0.9, embedding=vector(0),
                quality=quality_score(), visibility=FaceVisibility.FULL_FACE,
                track_state=state, frame_index=frame,
            )
        assert state.current_identity == "alice"

        decision = engine.decide(
            detector_confidence=0.9, embedding=None,
            visibility=FaceVisibility.NO_USABLE_FACE,
            track_state=state, frame_index=5,
        )
        assert decision.status is IdentityStatus.TEMPORARILY_MAINTAINED
        assert decision.identity_id == "alice"
        # No new evidence, so no confidence is asserted.
        assert decision.identity_confidence is None

    def test_an_unconfirmed_track_gets_no_name_when_occluded(self, gallery) -> None:
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.5, min_quality=0.0)
        )
        state = TrackIdentityState(track_id=2)
        decision = engine.decide(
            detector_confidence=0.9, embedding=None,
            visibility=FaceVisibility.NO_USABLE_FACE,
            track_state=state, frame_index=0,
        )
        assert decision.status is IdentityStatus.NO_FACE
        assert decision.identity_id is None


class TestCalibration:
    def test_too_few_samples_yields_no_confidence(self) -> None:
        samples = [CalibrationSample(0.8, True) for _ in range(3)]
        samples += [CalibrationSample(0.1, False) for _ in range(3)]
        model = fit_calibration(samples)
        assert not model.conditions["full"].usable
        assert model.probability(0.8, "full") is None
        assert model.notes

    def test_a_fitted_mapping_is_monotone(self) -> None:
        rng = np.random.default_rng(2)
        samples = [
            CalibrationSample(float(s), True)
            for s in rng.normal(0.65, 0.1, 60)
        ] + [
            CalibrationSample(float(s), False)
            for s in rng.normal(0.08, 0.08, 150)
        ]
        model = fit_calibration(samples)
        probabilities = [model.probability(s, "full") for s in (0.0, 0.2, 0.4, 0.6, 0.8)]
        assert all(a <= b for a, b in zip(probabilities, probabilities[1:], strict=False))

    def test_a_clean_gap_puts_the_threshold_in_the_middle(self) -> None:
        """At the edge of the gap, one unlucky impostor is a false accept."""
        genuine = [0.70, 0.75, 0.80]
        impostor = [0.10, 0.12, 0.20]
        midpoint = separation_threshold(genuine, impostor)
        assert midpoint == pytest.approx(0.45)
        assert separation_threshold([0.3], [0.5]) is None

    def test_operating_points_report_the_real_rates(self) -> None:
        points = operating_points([0.8, 0.9], [0.1, 0.2], thresholds=[0.5])
        assert points[0].tar == 1.0
        assert points[0].far == 0.0

    def test_conditions_are_calibrated_separately(self) -> None:
        rng = np.random.default_rng(8)
        samples = []
        for condition, centre in (("full", 0.7), ("masked", 0.35)):
            samples += [
                CalibrationSample(float(s), True, condition=condition)
                for s in rng.normal(centre, 0.07, 40)
            ]
            samples += [
                CalibrationSample(float(s), False, condition=condition)
                for s in rng.normal(0.06, 0.06, 90)
            ]
        model = fit_calibration(samples)
        # The same cosine means different things under different conditions.
        assert model.probability(0.4, "full") != model.probability(0.4, "masked")


class TestAdaptationGating:
    @pytest.fixture
    def adapter(self, gallery) -> GalleryAdapter:
        return GalleryAdapter(
            gallery,
            AdaptationConfig(enabled=True, min_similarity=0.7, min_quality=0.7,
                             min_confirmation_frames=5, min_margin=0.15,
                             min_track_stability=0.0, cooldown_seconds=0.0),
        )

    @staticmethod
    def confirmed_track(confirmations: int = 8) -> TrackIdentityState:
        state = TrackIdentityState(track_id=1, last_seen_frame=30)
        state.current_identity = "alice"
        state.confirmation_count = confirmations
        state.frames_named = 25
        state.phase = TrackIdentityPhase.RECOGNIZED
        return state

    @staticmethod
    def match(similarity: float = 0.85, runner: float = 0.1):
        from src.identity.types import FaceMatchResult

        return FaceMatchResult(
            best_identity_id="alice", best_similarity=similarity,
            runner_up_id="bob", runner_up_similarity=runner, gallery_size=3,
        )

    def test_a_verified_observation_is_accepted(self, adapter, gallery) -> None:
        record = adapter.consider(
            # Close to the reference, but far enough to carry new information:
            # inside the novelty band the adapter insists on.
            embedding=l2_normalize(vector(0) * 0.78 + vector(30) * 0.50),
            match=self.match(), quality=quality_score(0.9),
            visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.accepted
        assert gallery.get("alice").live_count == 1

    def test_a_weak_match_is_refused(self, adapter) -> None:
        record = adapter.consider(
            embedding=vector(31), match=self.match(similarity=0.5),
            quality=quality_score(0.9), visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_LOW_SIMILARITY

    def test_a_poor_quality_face_is_refused(self, adapter) -> None:
        record = adapter.consider(
            embedding=vector(32), match=self.match(),
            quality=quality_score(0.3), visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_LOW_QUALITY

    def test_an_unconfirmed_track_cannot_contribute(self, adapter) -> None:
        record = adapter.consider(
            embedding=vector(33), match=self.match(), quality=quality_score(0.9),
            visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(confirmations=1),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_UNCONFIRMED

    def test_a_contested_observation_is_refused(self, adapter) -> None:
        """The contamination route: a marginal call must never be stored."""
        record = adapter.consider(
            embedding=vector(34), match=self.match(similarity=0.85, runner=0.80),
            quality=quality_score(0.9), visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_AMBIGUOUS

    def test_a_masked_face_never_extends_an_identity(self, adapter) -> None:
        record = adapter.consider(
            embedding=vector(35), match=self.match(), quality=quality_score(0.9),
            visibility=FaceVisibility.MASKED,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_NOT_FULL_FACE

    def test_a_near_duplicate_is_refused(self, adapter) -> None:
        record = adapter.consider(
            embedding=vector(0),                 # identical to the reference
            match=self.match(), quality=quality_score(0.9),
            visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_REDUNDANT

    def test_an_implausibly_different_observation_is_refused(self, adapter) -> None:
        record = adapter.consider(
            embedding=vector(500), match=self.match(), quality=quality_score(0.9),
            visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_TOO_NOVEL

    def test_adaptation_is_off_by_default(self, gallery) -> None:
        adapter = GalleryAdapter(gallery, AdaptationConfig())
        assert not adapter.enabled
        record = adapter.consider(
            embedding=vector(36), match=self.match(), quality=quality_score(0.9),
            visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert record.outcome is AdaptationOutcome.REJECTED_DISABLED

    def test_every_decision_is_logged_with_a_reason(self, adapter) -> None:
        adapter.consider(
            embedding=vector(37), match=self.match(similarity=0.4),
            quality=quality_score(0.9), visibility=FaceVisibility.FULL_FACE,
            track_state=self.confirmed_track(),
        )
        assert adapter.history
        record = adapter.history[-1]
        assert record.reason
        assert "identity_id" in record.to_dict()


class TestFailureTaxonomy:
    def test_failures_name_their_layer(self) -> None:
        assert FailureReason.PERSON_NOT_DETECTED.layer == "person_detection"
        assert FailureReason.FACE_TOO_SMALL.layer == "face"
        assert FailureReason.LOW_SIMILARITY.layer == "recognition"
        assert FailureReason.TRACK_LOST.layer == "tracking"

    def test_a_no_face_result_is_not_a_rejection(self, gallery) -> None:
        engine = IdentityDecisionEngine(gallery, MatchingThresholds(min_quality=0.0))
        decision = engine.decide(detector_confidence=0.9, embedding=None)
        assert decision.status is IdentityStatus.NO_FACE
        assert decision.status is not IdentityStatus.UNKNOWN
        assert decision.failure is FailureReason.FACE_NOT_FOUND

    def test_a_degraded_face_reports_which_component_failed(self, gallery) -> None:
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.5, min_quality=0.6)
        )
        poor = FaceQualityScore(
            overall=0.2, resolution=0.05, sharpness=0.9, exposure=0.9,
            pose=0.9, landmarks=0.9, visibility=0.9, interocular_px=9.0,
        )
        decision = engine.decide(
            detector_confidence=0.9, embedding=vector(0), quality=poor,
            visibility=FaceVisibility.FULL_FACE,
        )
        assert decision.status is IdentityStatus.NO_FACE
        assert decision.failure is FailureReason.FACE_TOO_SMALL
