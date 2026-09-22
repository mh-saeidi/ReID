"""Face recognition: alignment, detection, the no-face state, and the property
that makes this mode worth having -- identity that does not depend on clothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.config.schema import FaceBackend, FaceSearchRegion, RecognitionMode
from src.core.exceptions import ModelLoadError, NoFaceFoundError
from src.core.types import BBox, RecognitionResult, RecognitionStatus
from src.face.align import (
    ARCFACE_TEMPLATE_112,
    align_face,
    crop_face_fallback,
    estimate_transform,
    template_for,
)
from src.face.factory import build_face_detector, build_face_encoder, select_backend
from src.face.preprocess import FaceCropPreprocessor
from src.face.types import FaceQuality
from src.reid.preprocess import CropStatus
from src.utils.device import resolve_device
from tests.conftest import FakeFaceDetector, make_face, person_scene


def textured(height: int, width: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class TestAlignment:
    def test_template_has_five_points(self) -> None:
        assert ARCFACE_TEMPLATE_112.shape == (5, 2)
        assert template_for(224).max() == pytest.approx(ARCFACE_TEMPLATE_112.max() * 2)

    def test_landmarks_map_onto_the_template(self) -> None:
        """Alignment must actually put the eyes where the encoder expects them."""
        face = make_face((100.0, 80.0, 200.0, 210.0))
        matrix = estimate_transform(face.landmarks, 112)
        points = np.hstack([face.landmarks, np.ones((5, 1), dtype=np.float32)])
        mapped = points @ matrix.T
        assert np.allclose(mapped, ARCFACE_TEMPLATE_112, atol=2.0)

    def test_alignment_output_geometry(self) -> None:
        image = textured(480, 640)
        chip = align_face(image, make_face().landmarks, 112)
        assert chip.shape == (112, 112, 3)

    def test_alignment_is_invariant_to_head_roll(self) -> None:
        """A rotated head must produce (nearly) the same chip, which is the
        whole reason alignment exists."""
        upright = make_face((200.0, 150.0, 300.0, 280.0))
        angle = np.deg2rad(20.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        centre = upright.landmarks.mean(axis=0)
        rolled = (upright.landmarks - centre) @ rotation.T + centre

        upright_matrix = estimate_transform(upright.landmarks, 112)
        rolled_matrix = estimate_transform(rolled, 112)
        # Both transforms land the landmarks on the same template.
        for landmarks, matrix in ((upright.landmarks, upright_matrix), (rolled, rolled_matrix)):
            points = np.hstack([landmarks, np.ones((5, 1), dtype=np.float32)])
            assert np.allclose(points @ matrix.T, ARCFACE_TEMPLATE_112, atol=2.0)

    def test_degenerate_landmarks_raise(self) -> None:
        collapsed = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises((ValueError, Exception)):
            estimate_transform(collapsed, 112)

    def test_fallback_crop_when_landmarks_are_missing(self) -> None:
        image = textured(480, 640)
        chip = crop_face_fallback(image, BBox(100, 100, 200, 220), 112)
        assert chip is not None and chip.shape == (112, 112, 3)

    def test_fallback_rejects_an_empty_region(self) -> None:
        image = textured(100, 100)
        assert crop_face_fallback(image, BBox(500, 500, 501, 501), 112) is None


class TestFaceDetectionType:
    def test_size_and_eye_distance(self) -> None:
        face = make_face((0.0, 0.0, 80.0, 120.0))
        assert face.size == 80
        # Template inter-ocular distance scaled to an 80px-wide face.
        expected = float(
            np.linalg.norm(ARCFACE_TEMPLATE_112[1] - ARCFACE_TEMPLATE_112[0]) * 80 / 112
        )
        assert face.eye_distance == pytest.approx(expected, abs=0.5)

    def test_offset_moves_box_and_landmarks_together(self) -> None:
        face = make_face((10.0, 20.0, 60.0, 90.0))
        moved = face.offset(100.0, 200.0)
        assert moved.bbox.x1 == 110.0
        assert moved.landmarks[0][0] == pytest.approx(face.landmarks[0][0] + 100.0)

    def test_a_face_without_landmarks_reports_so(self) -> None:
        face = make_face(landmarks=False)
        assert not face.has_landmarks
        assert face.eye_distance == 0.0


class TestFaceCropPreprocessor:
    @pytest.fixture
    def image(self) -> np.ndarray:
        return textured(480, 640)

    def test_a_good_face_produces_an_aligned_chip(self, image) -> None:
        pre = FaceCropPreprocessor(FakeFaceDetector([make_face((240, 100, 300, 175))]))
        result = pre.extract(image, BBox(200, 60, 340, 460))
        assert result.ok
        assert result.image.shape == (112, 112, 3)
        assert result.face is not None

    def test_no_face_is_reported_as_no_face_not_unknown(self, image) -> None:
        """The decisive distinction: nothing to compare is not a rejection."""
        pre = FaceCropPreprocessor(FakeFaceDetector([]))
        result = pre.extract(image, BBox(200, 60, 340, 460))
        assert not result.ok
        assert result.status is CropStatus.NO_FACE
        assert result.image is None
        assert "no face" in result.detail.lower()

    def test_a_tiny_face_is_refused_rather_than_guessed(self, image) -> None:
        pre = FaceCropPreprocessor(
            FakeFaceDetector([make_face((250, 110, 270, 135))]), min_face_size=40
        )
        result = pre.extract(image, BBox(200, 60, 340, 460))
        assert result.status is CropStatus.FACE_TOO_SMALL
        assert result.face.quality is FaceQuality.TOO_SMALL
        assert "too little detail" in result.detail

    def test_a_strongly_turned_face_is_refused(self, image) -> None:
        face = make_face((240, 100, 300, 175))
        # Collapse the eyes together: a near-profile view.
        face.landmarks[1] = face.landmarks[0] + np.array([2.0, 0.0], dtype=np.float32)
        pre = FaceCropPreprocessor(
            FakeFaceDetector([face]), min_face_size=10, min_eye_distance=15.0
        )
        result = pre.extract(image, BBox(200, 60, 340, 460))
        assert result.status is CropStatus.FACE_TOO_SMALL
        assert result.face.quality is FaceQuality.EXTREME_POSE

    def test_a_degenerate_person_box_is_reported(self, image) -> None:
        pre = FaceCropPreprocessor(FakeFaceDetector([make_face()]))
        assert pre.extract(image, BBox(10, 10, 14, 14)).status is CropStatus.DEGENERATE_BOX

    def test_face_mode_declares_it_requires_a_face(self) -> None:
        pre = FaceCropPreprocessor(FakeFaceDetector([]))
        assert pre.mode is RecognitionMode.FACE
        assert pre.requires_visible_face is True

    def test_only_this_person_s_face_is_used(self, image) -> None:
        """A bystander's face must not be borrowed for this person's identity."""
        mine = make_face((240, 100, 300, 175))
        bystander = make_face((520, 100, 610, 210))     # bigger, but elsewhere
        pre = FaceCropPreprocessor(
            FakeFaceDetector([bystander, mine]), search_region=FaceSearchRegion.FRAME
        )
        result = pre.extract(image, BBox(200, 60, 340, 460))
        assert result.ok
        assert result.face.bbox.x1 == 240

    def test_missing_landmarks_are_refused_when_required(self, image) -> None:
        pre = FaceCropPreprocessor(
            FakeFaceDetector([make_face((240, 100, 300, 175), landmarks=False)]),
            require_landmarks=True,
        )
        assert pre.extract(image, BBox(200, 60, 340, 460)).status is CropStatus.NO_FACE

    def test_missing_landmarks_fall_back_when_allowed(self, image) -> None:
        pre = FaceCropPreprocessor(
            FakeFaceDetector([make_face((240, 100, 300, 175), landmarks=False)]),
            require_landmarks=False,
        )
        result = pre.extract(image, BBox(200, 60, 340, 460))
        assert result.ok
        assert result.image.shape == (112, 112, 3)

    def test_upper_body_search_limits_the_region(self, image) -> None:
        detector = FakeFaceDetector([])
        pre = FaceCropPreprocessor(detector, search_region=FaceSearchRegion.UPPER_BODY)
        pre.extract(image, BBox(200, 60, 340, 460))
        assert detector.calls == 1   # one crop searched, not the whole frame


class TestNoFaceIsNeutralEvidence:
    """A hidden face must not erode, invent or silently extend an identity."""

    def _stabilizer(self, hold: int):
        from src.config.schema import IdentityStabilityConfig, MatchingConfig
        from src.tracking.stabilizer import IdentityHistory, IdentityStabilizer

        matching = MatchingConfig(
            recognition_threshold=0.45, high_confidence_threshold=0.60
        )
        stabilizer = IdentityStabilizer(
            IdentityStabilityConfig(minimum_recognized_frames=2, history_size=10),
            matching,
            absent_hold_frames=hold,
        )
        return stabilizer, IdentityHistory()

    @staticmethod
    def _seen(identity: str, similarity: float = 0.9) -> RecognitionResult:
        return RecognitionResult(
            status=RecognitionStatus.RECOGNIZED,
            identity_id=identity,
            identity_name=identity.title(),
            similarity=similarity,
        )

    def test_identity_is_held_while_the_face_is_hidden(self) -> None:
        stabilizer, state = self._stabilizer(hold=5)
        for i in range(3):
            stabilizer.observe(state, self._seen("john"), i)
        assert state.current_identity == "john"

        held = [
            stabilizer.observe(state, RecognitionResult.no_face(), 3 + i) for i in range(5)
        ]
        assert all(r.identity_id == "john" for r in held)
        assert state.frames_absent == 5

    def test_the_hold_is_bounded(self) -> None:
        """An identity must not be carried indefinitely on stale evidence."""
        stabilizer, state = self._stabilizer(hold=3)
        for i in range(3):
            stabilizer.observe(state, self._seen("john"), i)

        results = [
            stabilizer.observe(state, RecognitionResult.no_face(), 3 + i) for i in range(6)
        ]
        assert results[2].identity_id == "john"      # within the budget
        assert results[-1].identity_id is None       # released
        assert results[-1].status is RecognitionStatus.NO_FACE

    def test_hidden_frames_do_not_erode_the_evidence_window(self) -> None:
        stabilizer, state = self._stabilizer(hold=10)
        for i in range(3):
            stabilizer.observe(state, self._seen("john"), i)
        window_before = len(state.history)

        for i in range(4):
            stabilizer.observe(state, RecognitionResult.no_face(), 3 + i)
        assert len(state.history) == window_before   # nothing appended
        assert state.frames_unknown == 0             # and nothing counted against

    def test_a_hidden_face_never_invents_an_identity(self) -> None:
        stabilizer, state = self._stabilizer(hold=5)
        result = stabilizer.observe(state, RecognitionResult.no_face(), 0)
        assert result.identity_id is None
        assert result.status is RecognitionStatus.NO_FACE

    def test_the_face_returning_resets_the_hold(self) -> None:
        stabilizer, state = self._stabilizer(hold=5)
        for i in range(3):
            stabilizer.observe(state, self._seen("john"), i)
        for i in range(3):
            stabilizer.observe(state, RecognitionResult.no_face(), 3 + i)
        assert state.absent_streak == 3

        stabilizer.observe(state, self._seen("john"), 6)
        assert state.absent_streak == 0


class TestBackendSelection:
    def test_sface_filename_selects_opencv(self) -> None:
        assert (
            select_backend(FaceBackend.AUTO, Path("face_recognition_sface_2021dec.onnx"))
            is FaceBackend.OPENCV
        )

    def test_arcface_filenames_select_the_onnx_backend(self) -> None:
        for name in ("w600k_r50.onnx", "glintr100.onnx", "arcface_r100.onnx"):
            assert select_backend(FaceBackend.AUTO, Path(name)) is FaceBackend.ARCFACE_ONNX

    def test_an_explicit_backend_wins(self) -> None:
        assert (
            select_backend(FaceBackend.OPENCV, Path("w600k_r50.onnx")) is FaceBackend.OPENCV
        )

    def test_a_missing_detector_model_is_actionable(self, face_config_dict, tmp_path) -> None:
        config = config_from_dict(face_config_dict, base_dir=tmp_path)
        paths = ProjectPaths.from_config(config)
        with pytest.raises(ModelLoadError, match="fetch_face_models"):
            build_face_detector(config, paths)

    def test_a_missing_recognition_model_is_actionable(
        self, face_config_dict, tmp_path
    ) -> None:
        config = config_from_dict(face_config_dict, base_dir=tmp_path)
        paths = ProjectPaths.from_config(config)
        device = resolve_device(config.device)
        with pytest.raises(ModelLoadError, match="fetch_face_models"):
            build_face_encoder(config, paths, device)


class TestFaceEnrollment:
    def test_a_reference_without_a_face_is_refused_not_downgraded(
        self, face_config_dict, tmp_path
    ) -> None:
        """Falling back to body appearance here would silently defeat the mode."""
        from src.config.schema import PersonConfig
        from src.core.types import Detection
        from src.identity.enrollment import Enroller
        from src.utils.image import imwrite
        from tests.conftest import FakeDetector, FakeEncoder

        config = config_from_dict(face_config_dict, base_dir=tmp_path)
        path = imwrite(tmp_path / "ref.jpg", person_scene((60, 120, 200)))
        enroller = Enroller(
            config,
            FakeDetector([[Detection(bbox=BBox(200, 80, 320, 440), confidence=0.9)]]),
            FakeEncoder(),
            FaceCropPreprocessor(FakeFaceDetector([])),   # no face found
        )
        with pytest.raises(NoFaceFoundError, match="no face detected"):
            enroller.enroll(
                PersonConfig(id="p", name="P", image_path=str(path)), [path]
            )

    def test_a_tiny_reference_face_is_refused_with_advice(
        self, face_config_dict, tmp_path
    ) -> None:
        from src.config.schema import PersonConfig
        from src.core.types import Detection
        from src.identity.enrollment import Enroller
        from src.utils.image import imwrite
        from tests.conftest import FakeDetector, FakeEncoder

        config = config_from_dict(face_config_dict, base_dir=tmp_path)
        path = imwrite(tmp_path / "ref.jpg", person_scene((60, 120, 200)))
        enroller = Enroller(
            config,
            FakeDetector([[Detection(bbox=BBox(200, 80, 320, 440), confidence=0.9)]]),
            FakeEncoder(),
            FaceCropPreprocessor(
                FakeFaceDetector([make_face((250, 110, 272, 138))]), min_face_size=40
            ),
        )
        with pytest.raises(NoFaceFoundError, match="min_face_size"):
            enroller.enroll(
                PersonConfig(id="p", name="P", image_path=str(path)), [path]
            )
