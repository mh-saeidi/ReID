"""The live face identity pipeline, frame by frame.

Driven with fakes so the behaviour under test is the pipeline's own logic --
face-to-person assignment, scheduling, evidence accumulation, and what happens
when the face goes away -- rather than the models'.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.core.types import BBox, Detection, Frame, RecognitionStatus
from src.identity.adaptation import AdaptationConfig, GalleryAdapter
from src.identity.decision import IdentityDecisionEngine, MatchingThresholds
from src.identity.face_gallery import FaceEmbeddingRecord, FaceGallery, FaceIdentity
from src.identity.factory import FaceIdentitySystem
from src.identity.state_machine import StabilityConfig
from src.pipeline.face_identity_processor import (
    FaceIdentityPipeline,
    assign_faces_to_people,
)
from src.reid.encoder import l2_normalize
from tests.conftest import FakeDetector, FakeEncoder, FakeFaceDetector, make_face

EMBED_DIM = 32


def person(box: tuple[float, float, float, float], track_id: int | None = 1) -> Detection:
    return Detection(bbox=BBox(*box), confidence=0.9, track_id=track_id)


class ScriptedEncoder(FakeEncoder):
    """Returns one fixed vector per call, so identity is under test control."""

    def __init__(self, vector: np.ndarray) -> None:
        super().__init__(dimension=int(vector.shape[-1]))
        self.vector = vector

    def embed_batch(self, crops):
        self.calls += 1
        return np.stack([self.vector for _ in crops], axis=0)

    def embed(self, crop):
        return self.vector


@pytest.fixture
def known_vector() -> np.ndarray:
    return l2_normalize(
        np.random.default_rng(7).normal(size=EMBED_DIM).astype(np.float32)
    )


@pytest.fixture
def config(tmp_path):
    return config_from_dict(
        {
            "application": {"log_level": "ERROR"},
            "recognition": {"mode": "face"},
            "models": {"detector": "fake://detector", "reid": "fake://reid"},
            "face": {"detector_model": "fake://yunet", "recognition_model": "fake://sface"},
            "face_identity": {
                "gallery_dir": str(tmp_path / "people"),
                "threshold_full": 0.5,
                "threshold_partial": 0.5,
                "threshold_masked": 0.5,
                "min_face_quality": 0.0,
            },
            "temporal": {"min_confirmation_frames": 3, "occlusion_hold_frames": 6},
            "recognition_scheduler": {"enabled": False},
            "tracking": {"enabled": True},
            "gallery": {"directory": str(tmp_path / "gallery")},
            "output": {"directory": str(tmp_path / "output"), "save_snapshots": False},
            "display": {"show_window": False},
            "people": [],
        },
        base_dir=tmp_path,
    )


def build_system(config, tmp_path, known_vector, faces) -> FaceIdentitySystem:
    paths = ProjectPaths.from_config(config)
    gallery = FaceGallery(paths.resolve(config.face_identity.gallery_dir))
    identity = FaceIdentity(id="alice", name="Alice", title="Tester")
    identity.embeddings["reference"] = FaceEmbeddingRecord(
        key="reference", vector=known_vector, quality=0.95
    )
    gallery.save(identity)

    engine = IdentityDecisionEngine(
        gallery,
        MatchingThresholds(full=0.5, partial=0.5, masked=0.5, min_quality=0.0),
        stability=StabilityConfig(
            min_confirmation_frames=config.temporal.min_confirmation_frames,
            occlusion_hold_frames=config.temporal.occlusion_hold_frames,
            min_evidence_weight=0.5,
        ),
    )
    return FaceIdentitySystem(
        detector=FakeFaceDetector(faces),
        encoder=ScriptedEncoder(known_vector),
        gallery=gallery,
        enroller=None,
        engine=engine,
        adapter=GalleryAdapter(gallery, AdaptationConfig()),
        calibration=None,
        chip_size=112,
    )


def frame(index: int, size: tuple[int, int] = (640, 480)) -> Frame:
    width, height = size
    return Frame(
        image=np.full((height, width, 3), 120, dtype=np.uint8),
        index=index,
        timestamp=float(index) / 25.0,
        source_id="test",
    )


@pytest.fixture
def pipeline(config, tmp_path, known_vector):
    """One person, one face inside their box, matching the gallery."""
    face = make_face((250.0, 100.0, 330.0, 200.0))
    detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
    system = build_system(config, tmp_path, known_vector, [face])
    return FaceIdentityPipeline(config, detector, system, use_tracking=True)


class TestFaceToPersonAssignment:
    def test_a_face_goes_to_the_person_containing_it(self) -> None:
        people = [person((0.0, 0.0, 100.0, 300.0), 1), person((200.0, 0.0, 300.0, 300.0), 2)]
        faces = [make_face((210.0, 20.0, 260.0, 80.0))]
        assert assign_faces_to_people(faces, people) == {1: faces[0]}

    def test_a_face_belonging_to_nobody_is_dropped(self) -> None:
        """The system identifies people. A face with no person attached has no
        track to accumulate evidence on, so it is not promoted into one."""
        people = [person((0.0, 0.0, 100.0, 300.0), 1)]
        assert assign_faces_to_people([make_face((500.0, 20.0, 560.0, 90.0))], people) == {}

    def test_the_nearer_person_wins_an_overlap(self) -> None:
        behind = person((0.0, 0.0, 400.0, 400.0), 1)
        in_front = person((180.0, 40.0, 280.0, 300.0), 2)
        faces = [make_face((200.0, 60.0, 250.0, 120.0))]
        assert assign_faces_to_people(faces, [behind, in_front]) == {1: faces[0]}

    def test_each_person_keeps_their_largest_face(self) -> None:
        people = [person((0.0, 0.0, 400.0, 400.0), 1)]
        small = make_face((10.0, 10.0, 40.0, 50.0))
        large = make_face((100.0, 100.0, 200.0, 220.0))
        assert assign_faces_to_people([small, large], people) == {0: large}


class TestRecognitionOverFrames:
    def test_a_registered_person_is_recognized(self, pipeline) -> None:
        for index in range(5):
            outcome = pipeline.process(frame(index))
        result = outcome.result.detections[0]
        assert result.recognition_status is RecognitionStatus.RECOGNIZED
        assert result.identity_id == "alice"
        assert result.identity_name == "Alice"

    def test_a_name_is_not_asserted_on_the_first_sighting(self, pipeline) -> None:
        outcome = pipeline.process(frame(0))
        result = outcome.result.detections[0]
        assert result.identity_id is None
        assert pipeline.identity_state(1).current_identity is None

    def test_the_output_keeps_every_quantity_separate(self, pipeline) -> None:
        for index in range(5):
            outcome = pipeline.process(frame(index))
        payload = outcome.result.detections[0].to_dict()
        assert payload["detector_confidence"] == pytest.approx(0.9)
        assert payload["face_detection_confidence"] == pytest.approx(0.95)
        assert payload["reid_similarity"] == pytest.approx(1.0, abs=1e-4)
        assert payload["face_quality"] is not None
        # Uncalibrated: no probability is invented from the similarity.
        assert payload["identity_confidence"] is None
        assert payload["face_visibility"]

    def test_a_track_id_is_not_an_identity(self, config, tmp_path, known_vector) -> None:
        """A tracked person whose face never appears stays unnamed, however
        long the track lives."""
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [])   # no faces
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        for index in range(20):
            outcome = pipeline.process(frame(index))
        result = outcome.result.detections[0]
        assert result.identity_id is None
        assert result.recognition_status is RecognitionStatus.NO_FACE
        assert result.track_id == 1

    def test_an_unregistered_person_stays_unknown(
        self, config, tmp_path, known_vector
    ) -> None:
        stranger = l2_normalize(
            np.random.default_rng(99).normal(size=EMBED_DIM).astype(np.float32)
        )
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [make_face()])
        system.encoder = ScriptedEncoder(stranger)
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        for index in range(8):
            outcome = pipeline.process(frame(index))
        result = outcome.result.detections[0]
        assert result.identity_id is None
        assert result.recognition_status is RecognitionStatus.UNKNOWN
        assert result.to_dict()["failure_reason"] == "low_similarity"

    def test_an_unnamed_track_is_never_counted_as_recognized(
        self, config, tmp_path, known_vector
    ) -> None:
        """A leading candidate with too little evidence is still nobody. It
        must not inflate the recognized count, which is what LOW_CONFIDENCE
        would do."""
        stranger = l2_normalize(
            np.random.default_rng(101).normal(size=EMBED_DIM).astype(np.float32)
        )
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [make_face()])
        system.encoder = ScriptedEncoder(stranger)
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        outcome = pipeline.process(frame(0))
        assert outcome.result.recognized == []
        assert outcome.result.detections[0].identity_id is None


class TestFaceDisappearing:
    @staticmethod
    def run_until_recognized(pipeline, frames: int = 5) -> int:
        for index in range(frames):
            pipeline.process(frame(index))
        assert pipeline.identity_state(1).current_identity == "alice"
        return frames

    def test_a_confirmed_identity_survives_a_brief_occlusion(
        self, config, tmp_path, known_vector
    ) -> None:
        face_detector = FakeFaceDetector([make_face((250.0, 100.0, 330.0, 200.0))])
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [])
        system.detector = face_detector
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)

        index = self.run_until_recognized(pipeline)
        face_detector.faces = []                       # the person turns away
        outcome = pipeline.process(frame(index))
        result = outcome.result.detections[0]
        assert result.identity_id == "alice"
        assert result.recognition_status is RecognitionStatus.RECOGNIZED

    def test_the_hold_is_bounded(self, config, tmp_path, known_vector) -> None:
        """An identity held through an occlusion expires; it is not permanent."""
        face_detector = FakeFaceDetector([make_face((250.0, 100.0, 330.0, 200.0))])
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [])
        system.detector = face_detector
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)

        index = self.run_until_recognized(pipeline)
        face_detector.faces = []
        for offset in range(30):
            outcome = pipeline.process(frame(index + offset))
        result = outcome.result.detections[0]
        assert result.identity_id is None
        assert pipeline.identity_state(1).current_identity is None

    def test_a_hidden_face_is_reported_as_no_face_not_unknown(
        self, config, tmp_path, known_vector
    ) -> None:
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [])
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        outcome = pipeline.process(frame(0))
        result = outcome.result.detections[0]
        assert result.recognition_status is RecognitionStatus.NO_FACE
        assert result.to_dict()["failure_reason"] == "face_not_found"


class TestSeveralPeople:
    def test_each_person_is_decided_independently(
        self, config, tmp_path, known_vector
    ) -> None:
        left = person((50.0, 80.0, 230.0, 460.0), 1)
        right = person((300.0, 80.0, 480.0, 460.0), 2)
        faces = [
            make_face((100.0, 100.0, 180.0, 200.0)),
            make_face((350.0, 100.0, 430.0, 200.0)),
        ]
        detector = FakeDetector([[left, right]])
        system = build_system(config, tmp_path, known_vector, faces)
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)

        for index in range(5):
            outcome = pipeline.process(frame(index))
        assert len(outcome.result.detections) == 2
        assert {d.track_id for d in outcome.result.detections} == {1, 2}
        # Both see the same scripted embedding, so both are Alice here; what
        # matters is that each track reached its own decision.
        assert all(d.identity_id == "alice" for d in outcome.result.detections)
        assert pipeline.identity_state(1) is not pipeline.identity_state(2)

    def test_a_person_without_a_face_is_not_given_the_other_one(
        self, config, tmp_path, known_vector
    ) -> None:
        left = person((50.0, 80.0, 230.0, 460.0), 1)
        right = person((300.0, 80.0, 480.0, 460.0), 2)
        detector = FakeDetector([[left, right]])
        # Only the left person has a face in frame.
        system = build_system(
            config, tmp_path, known_vector, [make_face((100.0, 100.0, 180.0, 200.0))]
        )
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)

        for index in range(6):
            outcome = pipeline.process(frame(index))
        by_track = {d.track_id: d for d in outcome.result.detections}
        assert by_track[1].identity_id == "alice"
        assert by_track[2].identity_id is None
        assert by_track[2].recognition_status is RecognitionStatus.NO_FACE


class TestGalleryProtection:
    def test_recognition_alone_never_writes_to_the_gallery(self, pipeline) -> None:
        for index in range(10):
            pipeline.process(frame(index))
        assert pipeline.gallery.get("alice").live_count == 0

    def test_an_empty_gallery_names_nobody(self, config, tmp_path) -> None:
        paths = ProjectPaths.from_config(config)
        gallery = FaceGallery(paths.resolve(config.face_identity.gallery_dir))
        engine = IdentityDecisionEngine(
            gallery, MatchingThresholds(full=0.5, min_quality=0.0)
        )
        system = FaceIdentitySystem(
            detector=FakeFaceDetector([make_face((250.0, 100.0, 330.0, 200.0))]),
            encoder=ScriptedEncoder(
                l2_normalize(np.ones(EMBED_DIM, dtype=np.float32))
            ),
            gallery=gallery, enroller=None, engine=engine,
            adapter=GalleryAdapter(gallery, AdaptationConfig()),
            calibration=None,
        )
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        outcome = pipeline.process(frame(0))
        result = outcome.result.detections[0]
        assert result.identity_id is None
        assert result.to_dict()["failure_reason"] == "empty_gallery"


class TestQualityGate:
    def test_a_face_below_the_quality_floor_is_not_matched(
        self, config, tmp_path, known_vector
    ) -> None:
        system = build_system(
            config, tmp_path, known_vector, [make_face((250.0, 100.0, 262.0, 116.0))]
        )
        system.engine = IdentityDecisionEngine(
            system.gallery,
            MatchingThresholds(full=0.5, min_quality=0.95),   # unreachable floor
        )
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        for index in range(5):
            outcome = pipeline.process(frame(index))
        result = outcome.result.detections[0]
        assert result.identity_id is None
        assert result.recognition_status is RecognitionStatus.NO_FACE


class TestStillImages:
    def test_a_single_image_is_decided_on_its_own_merits(self, pipeline) -> None:
        """There is no temporal evidence in one photograph. Requiring several
        confirming observations would answer "unknown" to every still image."""
        result = pipeline.process_image(frame(0).image)
        assert result.detections[0].identity_id == "alice"
        assert result.detections[0].recognition_status is RecognitionStatus.RECOGNIZED

    def test_a_still_image_still_rejects_a_stranger(
        self, config, tmp_path, known_vector
    ) -> None:
        stranger = l2_normalize(
            np.random.default_rng(123).normal(size=EMBED_DIM).astype(np.float32)
        )
        detector = FakeDetector([[person((200.0, 80.0, 380.0, 460.0))]])
        system = build_system(config, tmp_path, known_vector, [make_face()])
        system.encoder = ScriptedEncoder(stranger)
        pipeline = FaceIdentityPipeline(config, detector, system, use_tracking=True)
        result = pipeline.process_image(frame(0).image)
        assert result.detections[0].identity_id is None
        assert result.detections[0].recognition_status is RecognitionStatus.UNKNOWN

    def test_tracking_is_restored_after_a_still_image(self, pipeline) -> None:
        pipeline.process_image(frame(0).image)
        assert pipeline.uses_tracking


class TestReset:
    def test_reset_clears_identity_state(self, pipeline) -> None:
        for index in range(5):
            pipeline.process(frame(index))
        assert pipeline.identity_state(1) is not None
        pipeline.reset()
        assert pipeline.identity_state(1) is None
