"""Face detection, visibility and enrollment: the layers below identity.

SCRFD's decode is tested against a synthetic session whose outputs are
constructed by hand, so the assertions cover the decode arithmetic itself
rather than the weights.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.exceptions import EnrollmentError
from src.core.types import BBox
from src.face.align import (
    ARCFACE_TEMPLATE_112,
    align_face,
    crop_face_fallback,
    estimate_transform,
)
from src.face.scrfd import (
    _ANCHORS_PER_LOCATION,
    _STRIDES,
    SCRFDFaceDetector,
    _distance_to_boxes,
    _distance_to_landmarks,
    _nms,
)
from src.face.visibility import FaceVisibility, classify_visibility, estimate_pose
from src.identity.face_enrollment import (
    FaceEnroller,
    FaceSelection,
    ReferenceQualityConfig,
    ReferenceWarning,
    select_face,
)
from tests.conftest import FakeEncoder, FakeFaceDetector, make_face


class TestSCRFDDecode:
    def test_distances_decode_back_to_the_original_box(self) -> None:
        centres = np.array([[100.0, 200.0]], dtype=np.float32)
        distance = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        boxes = _distance_to_boxes(centres, distance)
        assert boxes[0] == pytest.approx([90.0, 180.0, 130.0, 240.0])

    def test_landmark_distances_interleave_x_and_y(self) -> None:
        """A transposed read here puts every eye on a cheekbone, and alignment
        then fails silently rather than loudly."""
        centres = np.array([[50.0, 60.0]], dtype=np.float32)
        distance = np.arange(10, dtype=np.float32).reshape(1, 10)
        points = _distance_to_landmarks(centres, distance)
        assert points.shape == (1, 5, 2)
        assert points[0][0] == pytest.approx([50.0, 61.0])   # +0, +1
        assert points[0][1] == pytest.approx([52.0, 63.0])   # +2, +3
        assert points[0][4] == pytest.approx([58.0, 69.0])   # +8, +9

    def test_nms_keeps_the_strongest_of_overlapping_boxes(self) -> None:
        boxes = np.array(
            [[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]],
            dtype=np.float32,
        )
        keep = _nms(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), 0.4)
        assert keep == [0, 2]

    def test_nms_on_nothing_is_not_an_error(self) -> None:
        assert _nms(np.empty((0, 4)), np.empty((0,)), 0.4) == []

    def test_anchor_centres_match_the_network_flattening(self) -> None:
        detector = SCRFDFaceDetector("unused.onnx")
        centres = detector._anchor_centres(2, 3, 8)
        assert centres.shape == (2 * 3 * _ANCHORS_PER_LOCATION, 2)
        # The two anchors at one location are adjacent, and columns vary fastest.
        assert centres[0] == pytest.approx([0.0, 0.0])
        assert centres[1] == pytest.approx([0.0, 0.0])
        assert centres[2] == pytest.approx([8.0, 0.0])
        assert centres[_ANCHORS_PER_LOCATION * 3] == pytest.approx([0.0, 8.0])

    def test_the_geometry_cache_is_keyed_by_shape(self) -> None:
        detector = SCRFDFaceDetector("unused.onnx")
        first = detector._anchor_centres(4, 4, 8)
        assert detector._anchor_centres(4, 4, 8) is first
        assert detector._anchor_centres(4, 4, 16) is not first

    @staticmethod
    def _synthetic_outputs(
        size: int, box: tuple[float, float, float, float], score: float
    ) -> list[np.ndarray]:
        """Build the nine maps SCRFD emits, with one positive anchor."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        scores, boxes, landmarks = [], [], []
        for level, stride in enumerate(_STRIDES):
            cells = (size // stride) ** 2 * _ANCHORS_PER_LOCATION
            score_map = np.zeros((cells, 1), dtype=np.float32)
            box_map = np.zeros((cells, 4), dtype=np.float32)
            landmark_map = np.zeros((cells, 10), dtype=np.float32)
            if level == 0:
                # Plant one face on the anchor nearest the box centre.
                grid = size // stride
                index = (
                    (int(cy) // stride) * grid + (int(cx) // stride)
                ) * _ANCHORS_PER_LOCATION
                anchor_x = (index // _ANCHORS_PER_LOCATION % grid) * stride
                anchor_y = (index // _ANCHORS_PER_LOCATION // grid) * stride
                score_map[index, 0] = score
                box_map[index] = np.array(
                    [anchor_x - x1, anchor_y - y1, x2 - anchor_x, y2 - anchor_y]
                ) / stride
                template = ARCFACE_TEMPLATE_112 * ((x2 - x1) / 112.0) + [x1, y1]
                landmark_map[index] = (
                    (template - [anchor_x, anchor_y]).reshape(-1) / stride
                )
            scores.append(score_map)
            boxes.append(box_map)
            landmarks.append(landmark_map)
        return scores + boxes + landmarks

    def test_a_planted_face_decodes_back_to_its_box(self) -> None:
        size = 640
        detector = SCRFDFaceDetector("unused.onnx", confidence=0.4)
        detector._input_size = (size, size)
        boxes, scores, landmarks = detector._decode(
            self._synthetic_outputs(size, (200.0, 160.0, 280.0, 260.0), 0.92),
            size, size,
        )
        assert scores.shape == (1,) and scores[0] == pytest.approx(0.92)
        assert boxes[0] == pytest.approx([200.0, 160.0, 280.0, 260.0], abs=1e-3)
        assert landmarks.shape == (1, 5, 2)
        # Left eye above and left of the mouth, as the template dictates.
        assert landmarks[0][0][0] < landmarks[0][1][0]
        assert landmarks[0][0][1] < landmarks[0][3][1]

    def test_sub_threshold_anchors_are_dropped(self) -> None:
        size = 640
        detector = SCRFDFaceDetector("unused.onnx", confidence=0.5)
        detector._input_size = (size, size)
        boxes, _, _ = detector._decode(
            self._synthetic_outputs(size, (200.0, 160.0, 280.0, 260.0), 0.2),
            size, size,
        )
        assert boxes.size == 0

    def test_letterboxing_maps_coordinates_back_to_the_frame(self) -> None:
        """Faces are reported in the caller's pixels, not the network's."""
        size = 640
        detector = SCRFDFaceDetector("unused.onnx", confidence=0.4)
        detector._input_size = (size, size)
        outputs = self._synthetic_outputs(size, (200.0, 160.0, 280.0, 260.0), 0.9)

        class _Session:
            def run(self, _names, _inputs):
                return outputs

        detector._session = _Session()
        detector._output_names = [str(i) for i in range(9)]
        detector._input_name = "input"

        # A 1280x960 frame letterboxes by 0.5, so the box should double.
        faces = detector.detect(np.zeros((960, 1280, 3), dtype=np.uint8))
        assert len(faces) == 1
        assert faces[0].bbox.x1 == pytest.approx(400.0, abs=1.0)
        assert faces[0].bbox.y2 == pytest.approx(520.0, abs=1.0)

    def test_a_degenerate_image_returns_nothing(self) -> None:
        detector = SCRFDFaceDetector("unused.onnx")
        detector._session = object()
        assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []


class TestVisibility:
    @staticmethod
    def face_image(
        skin: tuple[int, int, int], *, lower_cover: tuple[int, int, int] | None = None,
        size: int = 240,
    ) -> tuple[np.ndarray, BBox, np.ndarray]:
        """A textured synthetic face, optionally with the mouth region covered."""
        rng = np.random.default_rng(3)
        image = np.full((size, size, 3), 40, dtype=np.uint8)
        x1, y1, x2, y2 = 60, 50, 180, 200
        patch = np.full((y2 - y1, x2 - x1, 3), skin, dtype=np.int16)
        patch += rng.integers(-18, 18, patch.shape)          # skin texture
        image[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)
        if lower_cover is not None:
            # A flat covering, with none of the structure skin has.
            image[y1 + 90 : y2, x1:x2] = lower_cover
        landmarks = (
            ARCFACE_TEMPLATE_112 * ((x2 - x1) / 112.0) + [x1, y1]
        ).astype(np.float32)
        return image, BBox(x1, y1, x2, y2), landmarks

    def test_a_clear_face_is_full(self) -> None:
        image, bbox, landmarks = self.face_image((170, 150, 140))
        report = classify_visibility(image, bbox, landmarks)
        assert report.visibility is FaceVisibility.FULL_FACE
        assert report.lower_face_visible > 0.5

    def test_a_covered_mouth_is_detected_as_masked(self) -> None:
        image, bbox, landmarks = self.face_image(
            (170, 150, 140), lower_cover=(210, 210, 215)
        )
        report = classify_visibility(image, bbox, landmarks)
        assert report.visibility is FaceVisibility.MASKED
        assert report.eye_region_visible > report.lower_face_visible

    @pytest.mark.parametrize(
        "skin",
        [(245, 230, 215), (190, 165, 140), (130, 100, 80), (75, 55, 45), (45, 32, 26)],
        ids=["very-light", "light", "medium", "dark", "very-dark"],
    )
    def test_the_verdict_does_not_depend_on_skin_tone(self, skin) -> None:
        """A skin-colour test would report darker faces as occluded. Structure
        -- texture and chroma spread -- is what actually distinguishes skin
        from a mask, and it is tone-independent."""
        bare, bbox, landmarks = self.face_image(skin)
        assert classify_visibility(bare, bbox, landmarks).visibility is (
            FaceVisibility.FULL_FACE
        ), f"a bare face with skin {skin} was not read as fully visible"

        masked, bbox, landmarks = self.face_image(skin, lower_cover=(30, 30, 35))
        assert classify_visibility(masked, bbox, landmarks).visibility is (
            FaceVisibility.MASKED
        ), f"a covered face with skin {skin} was not read as masked"

    def test_a_turned_head_is_partial(self) -> None:
        image, bbox, landmarks = self.face_image((170, 150, 140))
        turned = landmarks.copy()
        turned[2][0] += 40.0                 # nose far towards one eye
        turned[3][0] += 30.0
        turned[4][0] += 30.0
        report = classify_visibility(image, bbox, turned)
        assert report.visibility is FaceVisibility.PARTIAL_FACE
        assert abs(report.yaw_estimate) > 45.0

    def test_missing_landmarks_are_reported_not_assumed(self) -> None:
        image, bbox, _ = self.face_image((170, 150, 140))
        report = classify_visibility(image, bbox, None)
        assert report.visibility is FaceVisibility.PARTIAL_FACE
        assert report.reasons

    def test_a_box_too_short_for_its_eyes_is_treated_as_occluded(self) -> None:
        """When a scarf makes the detector box only the visible upper face, the
        box stops spanning the occlusion -- but its aspect gives it away."""
        image, _, landmarks = self.face_image((170, 150, 140))
        interocular = float(np.linalg.norm(landmarks[0] - landmarks[1]))
        cropped = BBox(60, 50, 180, 50 + interocular * 1.4)
        report = classify_visibility(image, cropped, landmarks)
        assert report.visibility is not FaceVisibility.FULL_FACE

    def test_pose_is_zero_for_a_frontal_template(self) -> None:
        yaw, pitch = estimate_pose(ARCFACE_TEMPLATE_112)
        assert abs(yaw) < 10.0
        assert abs(pitch) < 20.0


class TestAlignment:
    def test_alignment_maps_landmarks_onto_the_template(self) -> None:
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        face = make_face((100.0, 100.0, 212.0, 212.0))
        chip = align_face(image, face.landmarks, size=112)
        assert chip.shape == (112, 112, 3)

        # The transform must actually land the landmarks on the template.
        matrix = estimate_transform(face.landmarks, 112)
        points = np.hstack([face.landmarks, np.ones((5, 1), np.float32)]) @ matrix.T
        assert np.abs(points - ARCFACE_TEMPLATE_112).max() < 1.0

    def test_alignment_is_scale_invariant(self) -> None:
        """A face at 10 m and the same face at 2 m must produce the same chip,
        which is the entire reason alignment exists."""
        # A low-frequency pattern: resampling between scales preserves it, so
        # any residual difference is misalignment rather than lost detail.
        grid = np.mgrid[:112, :112].astype(np.float32)
        source = np.stack(
            [
                127 + 100 * np.sin(grid[0] / 18.0),
                127 + 100 * np.sin(grid[1] / 18.0),
                127 + 100 * np.sin((grid[0] + grid[1]) / 26.0),
            ],
            axis=-1,
        ).astype(np.uint8)

        big = np.zeros((600, 600, 3), dtype=np.uint8)
        big[100:324, 100:324] = cv2.resize(source, (224, 224))
        small = np.zeros((600, 600, 3), dtype=np.uint8)
        small[300:356, 300:356] = cv2.resize(source, (56, 56))

        big_chip = align_face(
            big, make_face((100.0, 100.0, 324.0, 324.0)).landmarks, size=112
        )
        small_chip = align_face(
            small, make_face((300.0, 300.0, 356.0, 356.0)).landmarks, size=112
        )
        difference = np.abs(
            big_chip.astype(np.int16) - small_chip.astype(np.int16)
        ).mean()
        assert difference < 25, "the same face at two scales aligned differently"

    def test_collapsed_landmarks_fail_loudly(self) -> None:
        """Degenerate points have no transform; guessing one would silently
        feed the encoder a meaningless chip."""
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="alignment transform"):
            align_face(image, np.zeros((5, 2), np.float32), size=112)

    def test_a_detection_without_landmarks_falls_back_to_a_crop(self) -> None:
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        face = make_face((100.0, 100.0, 212.0, 212.0), landmarks=False)
        assert face.landmarks is None
        chip = crop_face_fallback(image, face.bbox, size=112)
        assert chip.shape == (112, 112, 3)


class TestReferenceEnrollment:
    @staticmethod
    def passport(path: Path, *, size: int = 400) -> Path:
        rng = np.random.default_rng(12)
        image = np.full((size, size, 3), 220, dtype=np.uint8)
        image[80:320, 100:300] = rng.integers(60, 200, (240, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(path), image)
        return path

    def enroller(self, faces, **overrides) -> FaceEnroller:
        return FaceEnroller(
            FakeFaceDetector(faces), FakeEncoder(),
            ReferenceQualityConfig(**overrides),
        )

    def test_enrollment_starts_from_the_face_not_the_person(self, tmp_path) -> None:
        """A passport photo is a head and shoulders; requiring a person box
        first would reject the only image we are given."""
        path = self.passport(tmp_path / "ref.jpg")
        result = self.enroller(
            [make_face((100.0, 80.0, 300.0, 320.0))], min_overall_quality=0.0,
            min_sharpness=0.0, min_exposure=0.0,
        ).enroll("alice", path)
        assert result.person_id == "alice"
        assert result.dimension > 0
        assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)

    def test_a_photo_with_no_face_is_refused_with_guidance(self, tmp_path) -> None:
        path = self.passport(tmp_path / "ref.jpg")
        with pytest.raises(EnrollmentError, match="no face detected"):
            self.enroller([]).enroll("alice", path)

    def test_two_faces_are_refused_rather_than_guessed(self, tmp_path) -> None:
        path = self.passport(tmp_path / "ref.jpg")
        faces = [make_face((100.0, 80.0, 220.0, 240.0)), make_face((240.0, 90.0, 330.0, 220.0))]
        with pytest.raises(EnrollmentError, match="refusing to guess"):
            self.enroller(faces).enroll("alice", path)

    def test_several_faces_can_be_resolved_by_an_explicit_rule(self, tmp_path) -> None:
        path = self.passport(tmp_path / "ref.jpg")
        faces = [
            make_face((100.0, 80.0, 300.0, 320.0)),
            make_face((320.0, 90.0, 360.0, 140.0)),
            make_face((10.0, 300.0, 60.0, 360.0)),
        ]
        result = self.enroller(
            faces, max_faces=2, selection=FaceSelection.LARGEST_FACE,
            min_overall_quality=0.0, min_sharpness=0.0, min_exposure=0.0,
        ).enroll("alice", path)
        assert result.face.bbox.area == max(f.bbox.area for f in faces)
        assert ReferenceWarning.MULTIPLE_FACES_DETECTED in result.report.warnings

    def test_a_small_reference_face_is_flagged(self, tmp_path) -> None:
        path = self.passport(tmp_path / "ref.jpg")
        result = self.enroller(
            [make_face((150.0, 150.0, 195.0, 195.0))],
            min_overall_quality=0.0, min_sharpness=0.0, min_exposure=0.0,
        ).enroll("alice", path)
        assert ReferenceWarning.REFERENCE_FACE_TOO_SMALL in result.report.warnings
        assert result.report.messages, "a warning must explain itself"

    def test_a_bad_reference_can_be_made_fatal(self, tmp_path) -> None:
        path = self.passport(tmp_path / "ref.jpg")
        with pytest.raises(EnrollmentError):
            self.enroller(
                [make_face((150.0, 150.0, 190.0, 190.0))], fail_on_warnings=True
            ).enroll("alice", path)

    def test_warnings_do_not_block_enrollment_by_default(self, tmp_path) -> None:
        """The operator is told the reference is weak; they are not prevented
        from registering the only photo they have."""
        path = self.passport(tmp_path / "ref.jpg")
        result = self.enroller(
            [make_face((150.0, 150.0, 195.0, 195.0))],
            min_overall_quality=0.0, min_sharpness=0.0, min_exposure=0.0,
        ).enroll("alice", path)
        assert result.report.warnings and not result.report.ok
        assert result.embedding is not None

    def test_a_missing_file_names_the_person_and_the_path(self, tmp_path) -> None:
        with pytest.raises(EnrollmentError, match="alice"):
            self.enroller([make_face()]).enroll("alice", tmp_path / "nope.jpg")

    def test_the_reference_image_is_never_modified(self, tmp_path) -> None:
        path = self.passport(tmp_path / "ref.jpg")
        before = path.read_bytes()
        self.enroller(
            [make_face((100.0, 80.0, 300.0, 320.0))], min_overall_quality=0.0,
            min_sharpness=0.0, min_exposure=0.0,
        ).enroll("alice", path)
        assert path.read_bytes() == before

    def test_selection_strategies_pick_different_faces(self) -> None:
        faces = [
            make_face((10.0, 10.0, 150.0, 150.0), score=0.60),
            make_face((300.0, 300.0, 380.0, 380.0), score=0.99),
        ]
        shape = (640, 640)
        assert select_face(faces, FaceSelection.LARGEST_FACE, shape) is faces[0]
        assert select_face(faces, FaceSelection.HIGHEST_CONFIDENCE, shape) is faces[1]
        assert select_face(faces, FaceSelection.CENTER_MOST, shape) is faces[1]
