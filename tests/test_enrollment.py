"""One-shot enrollment: person selection, quality gates and failure modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.schema import SelectionStrategy
from src.core.exceptions import AmbiguousEnrollmentError, EnrollmentError, NoPersonFoundError
from src.core.types import BBox, Detection
from src.identity.enrollment import Enroller, select_detection
from src.identity.quality import assess_reference
from src.reid.preprocess import PersonCropPreprocessor
from src.utils.image import imwrite
from tests.conftest import FakeDetector, FakeEncoder, person_scene


def detection(box: tuple[float, float, float, float], confidence: float = 0.9) -> Detection:
    return Detection(bbox=BBox(*box), confidence=confidence)


@pytest.fixture
def reference_image(tmp_path: Path) -> Path:
    return imwrite(tmp_path / "person.jpg", person_scene((60, 120, 200)))


def make_enroller(config, detections, encoder=None):
    return Enroller(
        config,
        FakeDetector([detections]),
        encoder or FakeEncoder(),
        PersonCropPreprocessor(),
        crop_dir=None,
    )


def person_config(config, image_path: Path, **overrides):
    from src.config.schema import PersonConfig

    return PersonConfig(
        id=overrides.pop("id", "p1"),
        name=overrides.pop("name", "Test Person"),
        title=overrides.pop("title", "Tester"),
        image_path=str(image_path),
        **overrides,
    )


class TestSelectionStrategies:
    def test_largest_person_wins(self) -> None:
        small = detection((0, 0, 10, 20), 0.99)
        large = detection((100, 100, 200, 400), 0.5)
        chosen = select_detection([small, large], SelectionStrategy.LARGEST_PERSON, (480, 640))
        assert chosen is large

    def test_highest_confidence_wins(self) -> None:
        small = detection((0, 0, 10, 20), 0.99)
        large = detection((100, 100, 200, 400), 0.5)
        chosen = select_detection(
            [small, large], SelectionStrategy.HIGHEST_CONFIDENCE, (480, 640)
        )
        assert chosen is small

    def test_center_most_wins(self) -> None:
        edge = detection((0, 0, 60, 200))
        middle = detection((300, 220, 340, 260))
        chosen = select_detection([edge, middle], SelectionStrategy.CENTER_MOST, (480, 640))
        assert chosen is middle

    def test_empty_input_raises(self) -> None:
        with pytest.raises(NoPersonFoundError):
            select_detection([], SelectionStrategy.LARGEST_PERSON, (480, 640))


class TestOneShotEnrollment:
    def test_single_person_reference_succeeds(self, config, reference_image) -> None:
        enroller = make_enroller(config, [detection((200, 80, 320, 440))])
        result = enroller.enroll(person_config(config, reference_image), [reference_image])

        assert result.person_id == "p1"
        assert result.dimension == 32
        assert result.embeddings == [result.embedding]        # exactly one reference
        assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)
        assert result.detections_found == 1
        assert result.image_path == str(reference_image)

    def test_original_reference_image_is_never_modified(self, config, reference_image) -> None:
        before = reference_image.read_bytes()
        enroller = make_enroller(config, [detection((200, 80, 320, 440))])
        enroller.enroll(person_config(config, reference_image), [reference_image])
        assert reference_image.read_bytes() == before

    def test_no_person_detected_is_actionable(self, config, reference_image) -> None:
        enroller = make_enroller(config, [])
        with pytest.raises(NoPersonFoundError, match="no person detected"):
            enroller.enroll(person_config(config, reference_image), [reference_image])

    def test_multiple_people_are_refused_by_default(self, config, reference_image) -> None:
        assert config.enrollment.require_single_person is True
        enroller = make_enroller(
            config, [detection((200, 80, 320, 440)), detection((400, 90, 520, 430))]
        )
        with pytest.raises(AmbiguousEnrollmentError, match="refusing to guess"):
            enroller.enroll(person_config(config, reference_image), [reference_image])

    def test_multiple_people_are_resolved_when_allowed(
        self, base_config_dict: dict, tmp_path: Path, reference_image
    ) -> None:
        base_config_dict["enrollment"] = {
            "require_single_person": False,
            "selection_strategy": "largest_person",
        }
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        enroller = make_enroller(
            config, [detection((0, 0, 40, 60)), detection((200, 80, 320, 440))]
        )
        result = enroller.enroll(person_config(config, reference_image), [reference_image])
        assert result.detections_found == 2
        assert result.bbox.area == pytest.approx(BBox(200, 80, 320, 440).area)

    def test_missing_reference_file_is_actionable(self, config, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.jpg"
        enroller = make_enroller(config, [detection((200, 80, 320, 440))])
        with pytest.raises(EnrollmentError, match="not found"):
            enroller.enroll(person_config(config, missing), [missing])

    def test_degenerate_box_is_rejected(self, config, reference_image) -> None:
        enroller = make_enroller(config, [detection((10.0, 10.0, 11.0, 11.0))])
        with pytest.raises(EnrollmentError, match="too small to crop"):
            enroller.enroll(person_config(config, reference_image), [reference_image])

    def test_zero_embedding_is_rejected(self, config, reference_image) -> None:
        class ZeroEncoder(FakeEncoder):
            def embed_batch(self, crops):
                return np.zeros((len(crops), 32), dtype=np.float32)

        enroller = make_enroller(config, [detection((200, 80, 320, 440))], ZeroEncoder())
        with pytest.raises(EnrollmentError, match="empty embedding"):
            enroller.enroll(person_config(config, reference_image), [reference_image])

    def test_crop_is_saved_for_audit_when_configured(
        self, base_config_dict: dict, tmp_path: Path, reference_image
    ) -> None:
        base_config_dict["enrollment"] = {"save_normalized_crop": True}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        crop_dir = tmp_path / "crops"
        enroller = Enroller(
            config,
            FakeDetector([[detection((200, 80, 320, 440))]]),
            FakeEncoder(),
            PersonCropPreprocessor(),
            crop_dir=crop_dir,
        )
        result = enroller.enroll(person_config(config, reference_image), [reference_image])
        assert result.crop_path is not None
        assert Path(result.crop_path).exists()


class TestQualityGates:
    def test_a_good_reference_produces_no_warnings(self, config) -> None:
        # A flat colour has zero Laplacian variance and would (correctly) be
        # reported as blurry, so the stand-in person is given some texture.
        image = person_scene((60, 120, 200), size=(640, 900), box=(220, 60, 420, 860))
        rng = np.random.default_rng(5)
        image[60:860, 220:420] = rng.integers(0, 255, (800, 200, 3), dtype=np.uint8)
        crop = image[60:860, 220:420]
        report = assess_reference(
            image, crop, BBox(220, 60, 420, 860), config.enrollment.quality
        )
        assert report.ok, report.warnings

    def test_a_tiny_person_is_flagged(self, config) -> None:
        image = person_scene((60, 120, 200), size=(640, 480), box=(10, 10, 30, 50))
        crop = image[10:50, 10:30]
        report = assess_reference(image, crop, BBox(10, 10, 30, 50), config.enrollment.quality)
        assert any("tall" in w or "covers" in w for w in report.warnings)

    def test_a_dark_reference_is_flagged(self, config) -> None:
        image = person_scene((3, 3, 3), size=(640, 900), box=(220, 60, 420, 860))
        crop = image[60:860, 220:420]
        report = assess_reference(
            image, crop, BBox(220, 60, 420, 860), config.enrollment.quality
        )
        assert any("dark" in w for w in report.warnings)

    def test_a_truncated_person_is_flagged(self, config) -> None:
        image = person_scene((60, 120, 200), size=(640, 900), box=(0, 0, 200, 899))
        crop = image[0:899, 0:200]
        report = assess_reference(image, crop, BBox(0, 0, 200, 899), config.enrollment.quality)
        assert any("truncated" in w for w in report.warnings)

    def test_multiple_people_are_reported_in_the_quality_metrics(self, config) -> None:
        image = person_scene((60, 120, 200), size=(640, 900), box=(220, 60, 420, 860))
        crop = image[60:860, 220:420]
        report = assess_reference(
            image, crop, BBox(220, 60, 420, 860), config.enrollment.quality, person_count=3
        )
        assert report.metrics["person_count"] == 3
        assert any("3 people detected" in w for w in report.warnings)

    def test_fail_on_warnings_turns_a_warning_into_an_error(
        self, base_config_dict: dict, tmp_path: Path
    ) -> None:
        base_config_dict["enrollment"] = {"quality": {"fail_on_warnings": True}}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        # A tiny person triggers several warnings.
        dark = person_scene((2, 2, 2), size=(200, 200), box=(10, 10, 40, 60))
        path = imwrite(tmp_path / "poor.jpg", dark)
        enroller = make_enroller(config, [detection((10, 10, 40, 60))])
        with pytest.raises(EnrollmentError, match="rejected by quality gates"):
            enroller.enroll(person_config(config, path), [path])

    def test_quality_checks_can_be_disabled(self, base_config_dict: dict, tmp_path: Path) -> None:
        base_config_dict["enrollment"] = {"quality": {"enabled": False}}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        image = person_scene((2, 2, 2), size=(200, 200), box=(10, 10, 40, 60))
        report = assess_reference(
            image, image[10:60, 10:40], BBox(10, 10, 40, 60), config.enrollment.quality
        )
        assert report.ok


class TestFutureMultiImageEnrollment:
    """One image is enough today; the aggregation path is already in place."""

    def test_several_references_are_averaged_and_renormalised(
        self, config, tmp_path: Path
    ) -> None:
        first = imwrite(tmp_path / "a.jpg", person_scene((60, 120, 200)))
        second = imwrite(tmp_path / "b.jpg", person_scene((70, 130, 190)))
        enroller = Enroller(
            config,
            FakeDetector([[detection((200, 80, 320, 440))]]),
            FakeEncoder(),
            PersonCropPreprocessor(),
        )
        result = enroller.enroll(person_config(config, first), [first, second])
        assert len(result.embeddings) == 2
        assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)
