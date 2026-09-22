"""End-to-end pipeline behaviour with mocked models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.core.types import BBox, Detection, Frame, PersonIdentity, RecognitionStatus
from src.identity.gallery import IdentityGallery
from src.identity.matcher import IdentityMatcher
from src.pipeline.processor import ReIDPipeline
from src.reid.preprocess import PersonCropPreprocessor
from tests.conftest import EMBED_DIM, FakeDetector, FakeEncoder

ALICE_COLOR = (200, 40, 40)
BOB_COLOR = (40, 200, 40)
STRANGER_COLOR = (40, 40, 200)

ALICE_BOX = (60, 80, 180, 440)
BOB_BOX = (260, 80, 380, 440)
STRANGER_BOX = (440, 80, 560, 440)


def multi_person_scene(colors_and_boxes) -> np.ndarray:
    image = np.full((480, 640, 3), 90, dtype=np.uint8)
    for color, (x1, y1, x2, y2) in colors_and_boxes:
        image[y1:y2, x1:x2] = color
    return image


def detection(box, confidence=0.9, track_id=None) -> Detection:
    return Detection(bbox=BBox(*box), confidence=confidence, track_id=track_id)


class Harness:
    """A fully wired pipeline backed by fakes, with a pre-populated gallery."""

    def __init__(self, config, script, gallery_colors: dict[str, tuple[int, int, int]]):
        self.encoder = FakeEncoder()
        self.detector = FakeDetector(script)
        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.gallery_dir, paths.embeddings_dir, paths.metadata_dir)

        self.gallery = IdentityGallery(config, paths)
        for identity_id, color in gallery_colors.items():
            self.gallery._identities[identity_id] = PersonIdentity(  # noqa: SLF001
                id=identity_id,
                name=identity_id.title(),
                title="Tester",
                embedding=self.encoder.vector_for_color(color),
                embedding_dimension=EMBED_DIM,
            )
        self.gallery._invalidate()  # noqa: SLF001

        self.pipeline = ReIDPipeline(
            config,
            self.detector,
            self.encoder,
            PersonCropPreprocessor(),
            self.gallery,
            IdentityMatcher(config.matching),
            use_tracking=config.tracking.enabled,
        )

    def run(self, image, index: int = 0):
        return self.pipeline.process(
            Frame(image=image, index=index, timestamp=float(index), source_id="test")
        )


@pytest.fixture
def strict_config(base_config_dict: dict, tmp_path: Path):
    base_config_dict["matching"] = {
        "recognition_threshold": 0.95,
        "high_confidence_threshold": 0.99,
    }
    base_config_dict["tracking"] = {"enabled": False}
    return config_from_dict(base_config_dict, base_dir=tmp_path)


class TestOpenSetOnStillImages:
    def test_registered_people_are_recognized_independently(self, strict_config) -> None:
        """Several registered people in one image, each matched on its own."""
        image = multi_person_scene(
            [(ALICE_COLOR, ALICE_BOX), (BOB_COLOR, BOB_BOX), (STRANGER_COLOR, STRANGER_BOX)]
        )
        harness = Harness(
            strict_config,
            [[detection(ALICE_BOX), detection(BOB_BOX), detection(STRANGER_BOX)]],
            {"alice": ALICE_COLOR, "bob": BOB_COLOR},
        )
        result = harness.run(image).result

        identities = [d.identity_id for d in result.detections]
        assert identities == ["alice", "bob", None]
        assert len(result.recognized) == 2
        assert len(result.unknown) == 1

    def test_the_same_person_twice_is_recognized_twice(self, strict_config) -> None:
        """'John, Jane, a stranger and John again' -- all four resolve correctly."""
        boxes = [ALICE_BOX, BOB_BOX, STRANGER_BOX, (10, 80, 50, 440)]
        image = multi_person_scene(
            [
                (ALICE_COLOR, ALICE_BOX),
                (BOB_COLOR, BOB_BOX),
                (STRANGER_COLOR, STRANGER_BOX),
                (ALICE_COLOR, boxes[3]),
            ]
        )
        harness = Harness(
            strict_config,
            [[detection(b) for b in boxes]],
            {"alice": ALICE_COLOR, "bob": BOB_COLOR},
        )
        result = harness.run(image).result
        assert [d.identity_id for d in result.detections] == ["alice", "bob", None, "alice"]

    def test_an_unregistered_person_is_never_forced_onto_an_identity(
        self, strict_config
    ) -> None:
        image = multi_person_scene([(STRANGER_COLOR, ALICE_BOX)])
        harness = Harness(
            strict_config, [[detection(ALICE_BOX)]], {"alice": ALICE_COLOR, "bob": BOB_COLOR}
        )
        result = harness.run(image).result
        assert result.detections[0].recognition_status is RecognitionStatus.UNKNOWN
        assert result.detections[0].identity_id is None

    def test_an_empty_gallery_reports_everyone_as_unknown(self, strict_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        harness = Harness(strict_config, [[detection(ALICE_BOX)]], {})
        result = harness.run(image).result
        assert result.detections[0].identity_id is None
        assert result.detections[0].identity_name == "Unknown"

    def test_a_frame_with_no_people_produces_no_detections(self, strict_config) -> None:
        harness = Harness(strict_config, [[]], {"alice": ALICE_COLOR})
        result = harness.run(np.full((480, 640, 3), 90, dtype=np.uint8)).result
        assert result.detections == []
        assert result.timings["total"] > 0


class TestTrackingIntegration:
    @pytest.fixture
    def tracking_config(self, base_config_dict: dict, tmp_path: Path):
        base_config_dict["matching"] = {
            "recognition_threshold": 0.95,
            "high_confidence_threshold": 0.99,
        }
        base_config_dict["tracking"] = {
            "enabled": True,
            "identity_stability": {"minimum_recognized_frames": 2, "history_size": 8},
        }
        return config_from_dict(base_config_dict, base_dir=tmp_path)

    def test_identity_is_stabilized_across_frames(self, tracking_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        script = [[detection(ALICE_BOX, track_id=1)] for _ in range(5)]
        harness = Harness(tracking_config, script, {"alice": ALICE_COLOR})

        identities = [
            harness.run(image, index=i).result.detections[0].identity_id for i in range(5)
        ]
        # Withheld while the evidence window fills, then stable.
        assert identities[:1] == [None]
        assert identities[2:] == ["alice"] * 3

    def test_a_new_track_id_does_not_reset_the_identity(self, tracking_config) -> None:
        """Track ids are temporary; identity comes from the gallery, not the id."""
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        script = [[detection(ALICE_BOX, track_id=1)] for _ in range(4)]
        script += [[detection(ALICE_BOX, track_id=99)] for _ in range(4)]
        harness = Harness(tracking_config, script, {"alice": ALICE_COLOR})

        identities = [
            harness.run(image, index=i).result.detections[0].identity_id for i in range(8)
        ]
        assert identities[3] == "alice"
        # The re-numbered track re-acquires the same identity within the window.
        assert identities[-1] == "alice"

    def test_track_lifecycle_transitions_are_reported(self, tracking_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        script = [[detection(ALICE_BOX, track_id=1)] for _ in range(4)]
        harness = Harness(tracking_config, script, {"alice": ALICE_COLOR})

        first = harness.run(image, index=0)
        assert first.new_tracks == [1]
        transitions = []
        for index in range(1, 4):
            transitions += harness.run(image, index=index).transitions
        assert any(t.kind == "identified" for t in transitions)

    def test_detections_without_a_tracker_id_get_synthetic_ids(self, tracking_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX), (BOB_COLOR, BOB_BOX)])
        harness = Harness(
            tracking_config,
            [[detection(ALICE_BOX), detection(BOB_BOX)]],
            {"alice": ALICE_COLOR, "bob": BOB_COLOR},
        )
        result = harness.run(image).result
        track_ids = [d.track_id for d in result.detections]
        assert all(t is not None and t < 0 for t in track_ids)
        assert len(set(track_ids)) == 2


class TestPerformanceBehaviour:
    @pytest.fixture
    def interval_config(self, base_config_dict: dict, tmp_path: Path):
        base_config_dict["matching"] = {
            "recognition_threshold": 0.95,
            "high_confidence_threshold": 0.99,
        }
        base_config_dict["tracking"] = {
            "enabled": True,
            "identity_stability": {"minimum_recognized_frames": 1},
        }
        base_config_dict["performance"] = {
            "reid_interval": 4,
            "embedding_cache": True,
            "force_reid_when_unknown": False,
        }
        return config_from_dict(base_config_dict, base_dir=tmp_path)

    def test_reid_interval_reuses_cached_embeddings(self, interval_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        script = [[detection(ALICE_BOX, track_id=1)] for _ in range(8)]
        harness = Harness(interval_config, script, {"alice": ALICE_COLOR})

        for index in range(8):
            result = harness.run(image, index=index).result
            assert result.detections[0].identity_id == "alice"   # correctness preserved

        # 8 frames, interval 4: far fewer encoder invocations than frames.
        assert harness.encoder.crops_seen < 8
        assert harness.pipeline.track_manager.get(1).reid_runs < 8

    def test_max_detections_is_enforced(self, base_config_dict, tmp_path) -> None:
        base_config_dict["performance"] = {"max_detections": 2}
        base_config_dict["tracking"] = {"enabled": False}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        boxes = [(x, 80, x + 40, 440) for x in range(0, 300, 60)]
        image = multi_person_scene([(ALICE_COLOR, b) for b in boxes])
        harness = Harness(config, [[detection(b) for b in boxes]], {"alice": ALICE_COLOR})
        assert len(harness.run(image).result.detections) == 2

    def test_metrics_are_collected(self, strict_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        harness = Harness(strict_config, [[detection(ALICE_BOX)]], {"alice": ALICE_COLOR})
        harness.run(image)
        snapshot = harness.pipeline.metrics.snapshot()
        assert snapshot["frames"] == 1
        assert snapshot["detections"] == 1
        assert "detector" in snapshot["stages"]
        assert "reid" in snapshot["stages"]

    def test_a_degenerate_box_is_skipped_without_crashing(self, strict_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        harness = Harness(
            strict_config, [[detection((10.0, 10.0, 10.5, 10.5))]], {"alice": ALICE_COLOR}
        )
        result = harness.run(image).result
        assert result.detections[0].recognition_status is RecognitionStatus.PENDING


class TestSerialisation:
    def test_frame_result_serialises_the_documented_fields(self, strict_config) -> None:
        image = multi_person_scene([(ALICE_COLOR, ALICE_BOX)])
        harness = Harness(strict_config, [[detection(ALICE_BOX)]], {"alice": ALICE_COLOR})
        payload = harness.run(image).result.to_dict()

        assert set(payload) >= {
            "frame_index", "timestamp", "source_id", "width", "height",
            "num_detections", "num_recognized", "num_unknown", "detections",
        }
        detection_payload = payload["detections"][0]
        assert set(detection_payload) >= {
            "detection_id", "track_id", "bbox", "detector_confidence",
            "identity_id", "identity_name", "identity_title", "reid_similarity",
            "recognition_status", "timestamp", "source_id",
        }
        assert len(detection_payload["bbox"]) == 4
