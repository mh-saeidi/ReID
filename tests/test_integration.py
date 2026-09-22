"""End-to-end acceptance tests against the real models and demo data.

Identity comes from the face, so the decisive cases here are the two that
separate "identifies the person" from "identifies the outfit": a change of
clothing must not change the answer, and a hidden face must not produce one.

These are the scenarios from the acceptance checklist. They are marked
``slow``/``integration`` and skipped automatically when the model weights or the
demo dataset are absent, so the unit suite stays fast and hardware-independent.

    python scripts/build_demo.py
    pytest -m integration
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config
from src.core.types import RecognitionStatus
from src.pipeline.engine import build_engine
from src.pipeline.image_runner import ImageRunner
from src.pipeline.runner import StreamRunner
from src.sources.directory import ImageDirectorySource
from src.sources.image import ImageSource
from src.sources.video import VideoFileSource
from tests.conftest import (
    DEMO_DIR,
    PROJECT_ROOT,
    requires_demo,
    requires_face_models,
    requires_models,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    requires_models(),
    requires_face_models(),
    requires_demo(),
]


@pytest.fixture(scope="module")
def engine_config(tmp_path_factory):
    """The shipped config, redirected at a temporary output/gallery directory."""
    output_root = tmp_path_factory.mktemp("integration")
    base = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())

    # The temporary config lives outside the repository, so every relative path
    # it inherits has to be re-anchored on the project root.
    base["models"] = {
        key: str(PROJECT_ROOT / value) for key, value in base["models"].items()
    }
    base["people"] = [
        {**person, "image_path": str(PROJECT_ROOT / person["image_path"])}
        for person in base["people"]
    ]
    base["gallery"] = {**base.get("gallery", {}), "directory": str(output_root / "gallery")}
    # The face identity gallery is redirected too: a test must never read the
    # developer's enrolled people, and an empty one keeps these cases on the
    # person-gallery path they were written for.
    base["face_identity"] = {
        **base.get("face_identity", {}),
        "gallery_dir": str(output_root / "people"),
        "calibration_file": str(output_root / "people" / "calibration.json"),
    }
    base["output"] = {
        **base.get("output", {}),
        "directory": str(output_root / "output"),
        "save_snapshots": False,
        "save_video": False,
    }
    base["display"] = {"show_window": False, "show_metrics": False}
    base["events"] = {"enabled": True, "log_to_file": False}
    path = output_root / "config.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return load_config(path)


@pytest.fixture(scope="module")
def engine(engine_config):
    built = build_engine(engine_config)
    report = built.build_gallery()
    assert report.ok, report.failed
    assert len(built.gallery.active_identities) == 2
    yield built
    built.close()
    # Native model handles (OpenCV DNN, ONNX Runtime, torch) must be released
    # before the interpreter starts tearing those libraries down, or their
    # static destructors can race. Dropping the last references here keeps
    # teardown deterministic.
    del built
    gc.collect()


def names_in(result) -> list[str | None]:
    return [d.identity_id for d in result.detections]


class TestEnrollment:
    def test_one_reference_image_per_person_is_enough(self, engine) -> None:
        identities = {i.id: i for i in engine.gallery.active_identities}
        assert set(identities) == {"person_a", "person_b"}
        for identity in identities.values():
            assert identity.embedding is not None
            assert identity.embedding_dimension == engine.encoder.info.embedding_dimension
            assert len(identity.image_paths) == 1

    def test_a_reference_image_with_no_person_is_rejected(self, engine, tmp_path) -> None:
        import numpy as np

        from src.config.schema import PersonConfig
        from src.core.exceptions import NoPersonFoundError
        from src.utils.image import imwrite

        blank = imwrite(
            tmp_path / "blank.jpg", np.full((480, 640, 3), 200, dtype=np.uint8)
        )
        with pytest.raises(NoPersonFoundError, match="no person detected"):
            engine.enroller().enroll(
                PersonConfig(id="ghost", name="Ghost", image_path=str(blank)), [blank]
            )

    def test_a_crowded_reference_image_is_refused(self, engine) -> None:
        from src.config.schema import PersonConfig
        from src.core.exceptions import AmbiguousEnrollmentError

        crowded = DEMO_DIR / "test_images" / "03_a_and_b.jpg"
        assert engine.config.enrollment.require_single_person
        with pytest.raises(AmbiguousEnrollmentError, match="refusing to guess"):
            engine.enroller().enroll(
                PersonConfig(id="crowd", name="Crowd", image_path=str(crowded)), [crowded]
            )


class TestImageAcceptanceCases:
    """Cases 1-4 of the acceptance checklist."""

    def _run(self, engine, filename: str):
        pipeline = engine.pipeline(use_tracking=False)
        runner = ImageRunner(engine, pipeline, save=False)
        summary = runner.run(ImageSource(DEMO_DIR / "test_images" / filename))
        assert summary.images == 1
        return summary.outcomes[0].result

    def test_case_1_person_a_is_recognized(self, engine) -> None:
        result = self._run(engine, "01_person_a.jpg")
        assert "person_a" in names_in(result)

    def test_case_2_person_b_is_recognized(self, engine) -> None:
        result = self._run(engine, "02_person_b.jpg")
        assert "person_b" in names_in(result)

    def test_case_3_both_are_recognized_independently(self, engine) -> None:
        result = self._run(engine, "03_a_and_b.jpg")
        identities = names_in(result)
        assert "person_a" in identities
        assert "person_b" in identities

    def test_case_4_unregistered_people_stay_unknown(self, engine) -> None:
        result = self._run(engine, "04_unknown_only.jpg")
        assert result.detections, "the detector should still find people"
        assert all(d.identity_id is None for d in result.detections)
        # Strangers whose face is visible are rejected; those facing away report
        # no_face. Neither may ever become a match.
        assert all(
            d.recognition_status
            in (
                RecognitionStatus.UNKNOWN,
                RecognitionStatus.REJECTED,
                RecognitionStatus.NO_FACE,
            )
            for d in result.detections
        )

    def test_case_5_a_change_of_clothing_does_not_change_the_identity(
        self, engine
    ) -> None:
        """The reason this system identifies by face.

        05 recolours everything below the chin and leaves the face untouched.
        """
        baseline = self._run(engine, "01_person_a.jpg")
        changed = self._run(engine, "05_person_a_different_clothes.jpg")

        assert "person_a" in names_in(baseline)
        assert "person_a" in names_in(changed), (
            "the identity was lost when the clothing changed, which means the "
            "decision is not coming from the face"
        )
        before = next(d for d in baseline.detections if d.identity_id == "person_a")
        after = next(d for d in changed.detections if d.identity_id == "person_a")
        # Similarity must stay high, not merely scrape past the threshold.
        assert after.reid_similarity > 0.7
        assert abs(after.reid_similarity - before.reid_similarity) < 0.25

    def test_case_6_a_hidden_face_is_reported_not_guessed(self, engine) -> None:
        """06 obscures the face and leaves the clothing intact.

        Naming the person here would prove the system is reading the clothes.
        """
        result = self._run(engine, "06_person_a_face_hidden.jpg")
        assert result.detections, "the person should still be detected"
        assert all(d.identity_id is None for d in result.detections)
        assert any(
            d.recognition_status is RecognitionStatus.NO_FACE for d in result.detections
        )

    def test_directory_processing_covers_every_image(self, engine, tmp_path) -> None:
        pipeline = engine.pipeline(use_tracking=False)
        summary = ImageRunner(engine, pipeline, save=False).run(
            ImageDirectorySource(DEMO_DIR / "test_images")
        )
        assert summary.images == 6
        assert summary.failures == {}
        assert set(summary.identities) == {"person_a", "person_b"}


class TestVideoAcceptanceCase:
    """Case 5: a person is occluded, leaves the frame and comes back."""

    @pytest.fixture(scope="class")
    def video_run(self, engine):
        pipeline = engine.pipeline(use_tracking=True)
        frames = []
        runner = StreamRunner(
            engine, pipeline, show_window=False, save_video=False, on_frame=frames.append
        )
        summary = runner.run(VideoFileSource(DEMO_DIR / "test_video" / "walkthrough.mp4"))
        return summary, frames

    def test_the_whole_clip_is_processed(self, video_run) -> None:
        summary, frames = video_run
        assert summary.frames == 300
        assert len(frames) == 300

    def test_both_people_are_recognized_for_most_of_the_clip(self, video_run) -> None:
        summary, _ = video_run
        assert summary.identities.get("person_a", 0) > 150
        assert summary.identities.get("person_b", 0) > 250

    def test_identity_survives_a_brief_occlusion(self, video_run) -> None:
        """The label must not flicker when one frame's evidence collapses."""
        _, frames = video_run
        held = 0
        for frame in frames:
            for detection in frame.detections:
                instantaneous_failed = not detection.recognition.is_recognized
                stabilized_held = detection.effective.identity_id is not None
                if instantaneous_failed and stabilized_held:
                    held += 1
        assert held > 0, "expected at least one frame where stabilization held the identity"

    def test_a_hidden_face_produces_no_face_not_a_wrong_match(self, video_run) -> None:
        """Between 3.5s and 6.0s the clip obscures A's face and nothing else."""
        _, frames = video_run
        hidden = [
            d
            for frame in frames
            for d in frame.detections
            if d.recognition.status is RecognitionStatus.NO_FACE
        ]
        assert hidden, "the clip obscures a face; the pipeline should report no_face"
        # No detection may be matched to the wrong person while unobservable.
        assert all(
            d.effective.identity_id in (None, "person_a") for d in hidden
        )

    def test_the_identity_hold_is_bounded(self, video_run) -> None:
        """A held identity must expire; it must not ride on stale evidence."""
        _, frames = video_run
        hidden = [
            d
            for frame in frames
            for d in frame.detections
            if d.recognition.status is RecognitionStatus.NO_FACE
        ]
        held = [d for d in hidden if d.effective.identity_id is not None]
        released = [d for d in hidden if d.effective.identity_id is None]
        assert held, "the identity should be held across a short absence"
        assert released, "the identity should eventually be released"

    def test_a_person_who_leaves_and_returns_is_re_identified(self, video_run) -> None:
        _, frames = video_run
        # Person A is absent between roughly 5.5s and 8.0s at 25 fps.
        before = {
            d.track_id
            for f in frames[:130]
            for d in f.detections
            if d.identity_id == "person_a"
        }
        after = {
            d.track_id
            for f in frames[210:]
            for d in f.detections
            if d.identity_id == "person_a"
        }
        assert before, "person A should be identified before leaving"
        assert after, "person A should be re-identified after returning"

    def test_identity_is_not_merely_the_track_id(self, video_run) -> None:
        """Different track ids over the clip map back to the same identity."""
        _, frames = video_run
        track_ids = {
            d.track_id for f in frames for d in f.detections if d.identity_id == "person_a"
        }
        assert len(track_ids) >= 2, (
            "the demo clip re-numbers the track after occlusion; ReID should "
            "still resolve the same identity"
        )


class TestBenchmarkAndCalibration:
    def test_benchmark_reports_measured_values(self, engine) -> None:
        from src.tools.benchmark import run_benchmark

        pipeline = engine.pipeline(use_tracking=True)
        report = run_benchmark(
            engine,
            pipeline,
            VideoFileSource(DEMO_DIR / "test_video" / "walkthrough.mp4"),
            max_frames=25,
            warmup=3,
        )
        assert report.frames == 25
        assert report.end_to_end_fps > 0
        assert report.detector_fps > 0
        assert report.reid_fps > 0
        assert any(stage.name == "detector" for stage in report.stages)

    def test_calibration_separates_genuine_from_impostor(self, engine) -> None:
        from src.tools.calibration import evaluate_dataset

        dataset = DEMO_DIR / "evaluation"
        if not dataset.exists():
            pytest.skip("evaluation dataset missing; run scripts/build_demo.py")

        pipeline = engine.pipeline(use_tracking=False)
        report = evaluate_dataset(engine, pipeline, dataset)
        assert report.genuine_scores
        assert report.impostor_scores
        # The point of the tool: genuine matches score higher on average.
        assert sum(report.genuine_scores) / len(report.genuine_scores) > sum(
            report.impostor_scores
        ) / len(report.impostor_scores)
        assert report.best_point is not None


class TestOutputArtefacts:
    def test_an_image_run_writes_annotated_output_and_metadata(
        self, engine_config, tmp_path
    ) -> None:
        config = engine_config.model_copy(deep=True)
        config.base_dir = engine_config.base_dir
        config.output.directory = str(tmp_path / "out")
        config.output.save_metadata = True
        config.output.save_crops = True

        engine = build_engine(config)
        try:
            engine.ensure_gallery()
            runner = ImageRunner(engine, engine.pipeline(use_tracking=False))
            summary = runner.run(
                ImageSource(DEMO_DIR / "test_images" / "03_a_and_b.jpg")
            )
            outcome = summary.outcomes[0]
            assert outcome.annotated_path and Path(outcome.annotated_path).exists()
            assert outcome.metadata_path and Path(outcome.metadata_path).exists()
            assert outcome.crop_paths and all(Path(p).exists() for p in outcome.crop_paths)

            payload = json.loads(Path(outcome.metadata_path).read_text())
            assert payload["num_detections"] == len(outcome.result.detections)
        finally:
            engine.close()
