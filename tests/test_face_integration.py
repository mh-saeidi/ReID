"""End-to-end face identity tests against the real models.

These load SCRFD and ArcFace and run the evaluation dataset, so they are marked
``slow``. They assert behaviour -- a passport photo recognises its own subject
across conditions, a stranger is rejected, an identity survives an occlusion --
not specific accuracy figures, which belong in the evaluation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.face.visibility import FaceVisibility
from src.identity.face_gallery import FaceEmbeddingRecord, FaceIdentity
from src.identity.factory import build_face_identity_system
from src.identity.state_machine import TrackIdentityState
from src.identity.types import IdentityStatus
from src.tools.face_evaluation import FacePipeline
from src.utils.device import resolve_device
from tests.conftest import PROJECT_ROOT

pytestmark = [pytest.mark.slow, pytest.mark.integration]

DATASET = PROJECT_ROOT / "data" / "evaluation"
MODELS = PROJECT_ROOT / "models"

requires_stack = pytest.mark.skipif(
    not (MODELS / "scrfd_10g.onnx").exists()
    or not (MODELS / "w600k_r50.onnx").exists()
    or not (DATASET / "manifest.json").exists(),
    reason=(
        "face models or evaluation dataset missing; run "
        "scripts/fetch_face_models.py and scripts/build_evaluation_dataset.py"
    ),
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((DATASET / "manifest.json").read_text())


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    """A face identity system with a gallery built from passport photos only."""
    root = tmp_path_factory.mktemp("face-integration")
    config = config_from_dict(
        {
            "application": {"log_level": "ERROR"},
            "recognition": {"mode": "face"},
            "models": {"detector": "models/yolo26n.pt", "reid": "models/yolo26n-reid.onnx"},
            "face_identity": {
                "face_detector_model": str(MODELS / "scrfd_10g.onnx"),
                "face_encoder_model": str(MODELS / "w600k_r50.onnx"),
                "gallery_dir": str(root / "people"),
                "calibration_file": str(root / "calibration.json"),
            },
            "gallery": {"directory": str(root / "gallery")},
            "output": {"directory": str(root / "output"), "save_snapshots": False},
            "display": {"show_window": False},
            "people": [],
        },
        base_dir=PROJECT_ROOT,
    )
    paths = ProjectPaths.from_config(config)
    built = build_face_identity_system(config, paths, resolve_device(config.device))

    for directory in sorted((DATASET / "enrollment").iterdir()):
        if not directory.is_dir():
            continue
        reference = next(iter(sorted(directory.glob("*.jpg"))), None)
        if reference is None:
            continue
        result = built.enroller.enroll(directory.name, reference)
        identity = FaceIdentity(id=directory.name, name=directory.name)
        identity.embeddings["reference"] = FaceEmbeddingRecord(
            key="reference",
            vector=result.embedding,
            source="reference",
            quality=result.report.quality.overall if result.report.quality else 0.0,
        )
        built.gallery.save(identity)

    yield built
    built.close()


@pytest.fixture(scope="module")
def pipeline(system) -> FacePipeline:
    return FacePipeline(system.detector, system.encoder, system.gallery, system.engine,
                        chip_size=system.chip_size)


def query_images(condition: str, person: str) -> list[Path]:
    directory = DATASET / "query" / condition / person
    return sorted(directory.glob("*.jpg")) if directory.exists() else []


def identify(pipeline, system, path: Path):
    """Run one image all the way to a decision."""
    image = cv2.imread(str(path))
    assert image is not None, f"unreadable test image: {path}"
    embedding, quality, visibility, face = pipeline.embed_query(image)
    return system.engine.decide(
        detector_confidence=1.0,
        face_detection_confidence=face.score if face else None,
        embedding=embedding,
        quality=quality,
        visibility=visibility.visibility if visibility else FaceVisibility.NO_USABLE_FACE,
    )


@requires_stack
class TestEnrollmentFromPassportPhotos:
    def test_one_photo_per_person_produces_a_usable_gallery(self, system) -> None:
        assert len(system.gallery.active) >= 2
        for identity in system.gallery.active:
            assert identity.reference is not None
            assert identity.reference.vector.shape[-1] == 512

    def test_reference_embeddings_are_normalised(self, system) -> None:
        for identity in system.gallery.active:
            norm = float(np.linalg.norm(identity.reference.vector))
            assert norm == pytest.approx(1.0, abs=1e-4)

    def test_different_people_are_far_apart_in_embedding_space(self, system) -> None:
        identities = list(system.gallery.active)
        for i, a in enumerate(identities):
            for b in identities[i + 1 :]:
                similarity = float(a.reference.vector @ b.reference.vector)
                assert similarity < 0.5, (
                    f"{a.id} and {b.id} are not separable at all (cos={similarity:.3f})"
                )


@requires_stack
class TestRecognitionAcrossConditions:
    @pytest.mark.parametrize(
        "condition",
        ["normal", "glasses", "mask", "different_pose", "low_light",
         "different_distance", "blur"],
    )
    def test_the_passport_subject_is_found_under_this_condition(
        self, pipeline, system, condition
    ) -> None:
        """Rank-1: ignoring thresholds entirely, is the right person nearest?

        Kept separate from the accept/reject decision on purpose -- collapsing
        them hides whether a failure is the matcher or the threshold."""
        checked = 0
        correct = 0
        for identity in system.gallery.active:
            for path in query_images(condition, identity.id)[:3]:
                embedding, _, _, _ = pipeline.embed_query(cv2.imread(str(path)))
                if embedding is None:
                    continue
                checked += 1
                correct += system.gallery.match(embedding).best_identity_id == identity.id
        if checked == 0:
            pytest.skip(f"no usable query images for condition '{condition}'")
        assert correct / checked >= 0.8, (
            f"rank-1 under '{condition}' was {correct}/{checked}"
        )

    def test_a_normal_query_is_accepted_not_merely_ranked(
        self, pipeline, system
    ) -> None:
        accepted = 0
        total = 0
        for identity in system.gallery.active:
            for path in query_images("normal", identity.id)[:3]:
                decision = identify(pipeline, system, path)
                total += 1
                accepted += (
                    decision.status is IdentityStatus.RECOGNIZED
                    and decision.identity_id == identity.id
                )
        assert total and accepted / total >= 0.8


@requires_stack
class TestUnknownPeople:
    def test_an_unregistered_face_is_rejected(self, pipeline, system) -> None:
        """The open-set requirement: a stranger must come out UNKNOWN, not as
        whoever in the gallery happens to be nearest."""
        unknown_dir = DATASET / "unknown"
        images = sorted(unknown_dir.rglob("*.jpg"))
        if not images:
            pytest.skip("no unknown-person images in the dataset")

        rejected = 0
        considered = 0
        for path in images[:12]:
            decision = identify(pipeline, system, path)
            if decision.status is IdentityStatus.NO_FACE:
                continue                      # no face is a detection failure, not a match
            considered += 1
            rejected += decision.status is not IdentityStatus.RECOGNIZED
        assert considered, "every unknown image failed face detection"
        assert rejected / considered >= 0.9, (
            f"only {rejected}/{considered} strangers were rejected"
        )

    def test_the_gallery_is_not_contaminated_by_queries(self, system) -> None:
        """Nothing in a read-only recognition path may write to the gallery."""
        assert all(i.live_count == 0 for i in system.gallery.active)


@requires_stack
class TestSeparation:
    def test_genuine_and_impostor_scores_do_not_overlap(self, pipeline, system) -> None:
        """The property that decides whether *any* threshold can work.

        Accuracy at one operating point can look fine while the distributions
        overlap; this checks the thing that generalises.
        """
        genuine: list[float] = []
        impostor: list[float] = []
        for identity in system.gallery.active:
            for condition in ("normal", "glasses", "different_pose"):
                for path in query_images(condition, identity.id)[:2]:
                    embedding, _, _, _ = pipeline.embed_query(cv2.imread(str(path)))
                    if embedding is None:
                        continue
                    for other in system.gallery.active:
                        score = float(
                            embedding @ other.reference.vector
                        )
                        (genuine if other.id == identity.id else impostor).append(score)
        assert genuine and impostor
        gap = min(genuine) - max(impostor)
        assert gap > 0.0, (
            f"genuine minimum {min(genuine):.3f} is below impostor maximum "
            f"{max(impostor):.3f}; no threshold can separate these"
        )


@requires_stack
class TestTemporalBehaviour:
    def test_an_identity_survives_disappearing_and_reappearing(
        self, pipeline, system
    ) -> None:
        identity = next(iter(system.gallery.active))
        images = query_images("normal", identity.id)
        if len(images) < 2:
            pytest.skip("not enough query images to simulate a sequence")

        state = TrackIdentityState(track_id=1)
        frame = 0
        for _ in range(4):                     # seen, and confirmed
            identify_into(pipeline, system, images[0], state, frame)
            frame += 1
        assert state.current_identity == identity.id

        for _ in range(3):                     # turned away
            system.engine.decide(
                detector_confidence=0.9, embedding=None,
                visibility=FaceVisibility.NO_USABLE_FACE,
                track_state=state, frame_index=frame,
            )
            frame += 1
        assert state.current_identity == identity.id, "a brief occlusion lost the identity"

        decision = identify_into(pipeline, system, images[-1], state, frame)
        assert decision.identity_id == identity.id
        assert state.identity_switches == 0

    def test_several_people_keep_their_own_identities(self, pipeline, system) -> None:
        identities = list(system.gallery.active)[:2]
        if len(identities) < 2:
            pytest.skip("need two registered people")

        states = {i.id: TrackIdentityState(track_id=n) for n, i in enumerate(identities)}
        for frame in range(4):
            for identity in identities:
                images = query_images("normal", identity.id)
                if not images:
                    continue
                identify_into(
                    pipeline, system, images[frame % len(images)],
                    states[identity.id], frame,
                )
        for identity in identities:
            assert states[identity.id].current_identity in (None, identity.id), (
                "a track was assigned somebody else's identity"
            )


def identify_into(pipeline, system, path: Path, state: TrackIdentityState, frame: int):
    image = cv2.imread(str(path))
    embedding, quality, visibility, face = pipeline.embed_query(image)
    return system.engine.decide(
        detector_confidence=1.0,
        face_detection_confidence=face.score if face else None,
        embedding=embedding,
        quality=quality,
        visibility=visibility.visibility if visibility else FaceVisibility.NO_USABLE_FACE,
        track_state=state,
        frame_index=frame,
    )
