"""Output layer: snapshots, metadata, recording, rendering, events, retention."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.config.paths import ProjectPaths
from src.config.schema import (
    DisplayConfig,
    OutputConfig,
    RecordingConfig,
    RecordingMode,
    RetentionConfig,
    SnapshotMode,
)
from src.core.types import (
    BBox,
    DetectionResult,
    FrameResult,
    RecognitionResult,
    RecognitionStatus,
)
from src.events.manager import EventManager
from src.events.types import Event, EventType
from src.output.metadata import MetadataWriter, summarize, write_json
from src.output.recorder import AnnotatedVideoWriter, RecorderState, VideoRecorder
from src.output.renderer import Renderer, RenderMetrics
from src.output.retention import apply_retention
from src.output.snapshot import SnapshotWriter
from src.utils.image import imwrite, sanitize_filename, unique_path
from tests.conftest import person_scene


def make_detection(
    *,
    identity_id: str | None = "john_doe",
    name: str | None = "John Doe",
    title: str = "Manager",
    similarity: float = 0.73,
    status: RecognitionStatus = RecognitionStatus.RECOGNIZED,
    track_id: int | None = 17,
) -> DetectionResult:
    return DetectionResult(
        detection_id="abc123",
        bbox=BBox(100, 50, 220, 400),
        detector_confidence=0.91,
        source_id="webcam_0",
        timestamp=time.time(),
        frame_index=5,
        track_id=track_id,
        recognition=RecognitionResult(
            status=status,
            identity_id=identity_id,
            identity_name=name,
            identity_title=title,
            similarity=similarity,
        ),
    )


def make_frame_result(detections) -> FrameResult:
    return FrameResult(
        frame_index=5,
        timestamp=time.time(),
        source_id="webcam_0",
        width=640,
        height=480,
        detections=list(detections),
    )


class TestFilenameCollisions:
    def test_existing_files_are_never_silently_overwritten(self, tmp_path: Path) -> None:
        target = tmp_path / "shot.jpg"
        target.write_bytes(b"first")
        assert unique_path(target) == tmp_path / "shot_1.jpg"

        (tmp_path / "shot_1.jpg").write_bytes(b"second")
        assert unique_path(target) == tmp_path / "shot_2.jpg"

    def test_opting_in_allows_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "shot.jpg"
        target.write_bytes(b"first")
        assert unique_path(target, overwrite=True) == target

    def test_a_free_name_is_returned_unchanged(self, tmp_path: Path) -> None:
        assert unique_path(tmp_path / "fresh.jpg") == tmp_path / "fresh.jpg"

    def test_filename_sanitisation(self) -> None:
        assert sanitize_filename("John Doe / Manager") == "John_Doe___Manager"
        assert sanitize_filename("   ") == "unnamed"
        assert "/" not in sanitize_filename("a/b\\c")


class TestSnapshotWriter:
    @pytest.fixture
    def writer_factory(self, tmp_path: Path):
        def build(**overrides):
            config = OutputConfig(
                save_snapshots=True, snapshot_cooldown_seconds=0.0, **overrides
            )
            return SnapshotWriter(config, tmp_path / "snapshots", tmp_path / "crops"), config

        return build

    def test_recognized_mode_saves_only_recognized_people(self, writer_factory) -> None:
        writer, _ = writer_factory(snapshot_mode=SnapshotMode.RECOGNIZED)
        assert writer.should_save(make_detection())
        assert not writer.should_save(
            make_detection(identity_id=None, status=RecognitionStatus.UNKNOWN)
        )

    def test_unknown_mode_inverts_the_selection(self, writer_factory) -> None:
        writer, _ = writer_factory(snapshot_mode=SnapshotMode.UNKNOWN)
        assert not writer.should_save(make_detection())
        assert writer.should_save(
            make_detection(identity_id=None, status=RecognitionStatus.UNKNOWN)
        )

    def test_all_and_disabled_modes(self, writer_factory) -> None:
        all_writer, _ = writer_factory(snapshot_mode=SnapshotMode.ALL)
        assert all_writer.should_save(make_detection())
        off, _ = writer_factory(snapshot_mode=SnapshotMode.DISABLED)
        assert not off.enabled
        assert not off.should_save(make_detection())

    def test_events_only_mode_requires_an_event(self, writer_factory) -> None:
        writer, _ = writer_factory(snapshot_mode=SnapshotMode.EVENTS_ONLY)
        assert not writer.should_save(make_detection(), is_event=False)
        assert writer.should_save(make_detection(), is_event=True)

    def test_snapshot_and_sidecar_are_written(self, writer_factory, tmp_path: Path) -> None:
        writer, _ = writer_factory(snapshot_mode=SnapshotMode.ALL, save_metadata=True)
        detection = make_detection()
        record = writer.save(person_scene((10, 20, 30)), detection, make_frame_result([detection]))

        assert record is not None
        assert record.image_path.exists()
        assert record.metadata_path is not None and record.metadata_path.exists()

        payload = json.loads(record.metadata_path.read_text())
        for key in (
            "source", "timestamp", "identity_id", "identity_name", "identity_title",
            "reid_similarity", "bbox", "track_id", "detector_confidence",
        ):
            assert key in payload, key
        assert payload["identity_name"] == "John Doe"
        assert len(payload["bbox"]) == 4

    def test_snapshots_are_date_partitioned_and_labelled(self, writer_factory) -> None:
        writer, _ = writer_factory(snapshot_mode=SnapshotMode.ALL)
        detection = make_detection()
        record = writer.save(person_scene(), detection, make_frame_result([detection]))
        assert record.image_path.parent.name.count("-") == 2      # YYYY-MM-DD
        assert "john_doe" in record.image_path.name
        assert "0.73" in record.image_path.name

    def test_cooldown_rate_limits_a_stream(self, tmp_path: Path) -> None:
        config = OutputConfig(
            save_snapshots=True, snapshot_mode=SnapshotMode.ALL, snapshot_cooldown_seconds=60.0
        )
        writer = SnapshotWriter(config, tmp_path / "snaps", tmp_path / "crops")
        detection = make_detection()
        frame = make_frame_result([detection])
        assert writer.save(person_scene(), detection, frame) is not None
        assert writer.save(person_scene(), detection, frame) is None      # suppressed

    def test_crops_are_written_when_enabled(self, writer_factory) -> None:
        writer, _ = writer_factory(snapshot_mode=SnapshotMode.ALL, save_crops=True)
        detection = make_detection()
        record = writer.save(person_scene(), detection, make_frame_result([detection]))
        assert record.crop_path is not None and record.crop_path.exists()


class TestMetadataWriter:
    def test_one_json_object_per_frame(self, tmp_path: Path) -> None:
        writer = MetadataWriter(tmp_path / "meta", "webcam_0")
        for index in range(3):
            result = make_frame_result([make_detection()])
            result.frame_index = index
            writer.write(result)
        path = writer.close()

        assert path is not None
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert [row["frame_index"] for row in lines] == [0, 1, 2]
        assert lines[0]["num_recognized"] == 1

    def test_write_json_avoids_collisions(self, tmp_path: Path) -> None:
        first = write_json(tmp_path / "a.json", {"x": 1})
        second = write_json(tmp_path / "a.json", {"x": 2})
        assert first != second
        assert json.loads(second.read_text())["x"] == 2

    def test_summarize_counts_identities(self) -> None:
        frames = [
            make_frame_result([make_detection(), make_detection(identity_id="jane")]),
            make_frame_result(
                [make_detection(identity_id=None, status=RecognitionStatus.UNKNOWN)]
            ),
        ]
        stats = summarize(frames)
        assert stats["frames"] == 2
        assert stats["detections"] == 3
        assert stats["recognized"] == 2
        assert stats["unknown"] == 1
        assert stats["identities"]["john_doe"] == 1


class TestRecorder:
    @pytest.fixture
    def frames(self):
        return [person_scene((i * 5 % 250, 100, 200), size=(320, 240)) for i in range(40)]

    def test_continuous_mode_writes_every_frame(self, tmp_path: Path, frames) -> None:
        recorder = VideoRecorder(
            RecordingConfig(mode=RecordingMode.CONTINUOUS),
            OutputConfig(),
            tmp_path / "videos",
            fps=10.0,
            size=(320, 240),
            source_id="test",
        )
        for frame in frames[:10]:
            recorder.process(frame, trigger=False)
        clip = recorder.close()
        assert clip is not None and clip.frames == 10
        assert clip.path.exists() and clip.path.stat().st_size > 0

    def test_event_mode_includes_pre_event_frames_from_the_ring_buffer(
        self, tmp_path: Path, frames
    ) -> None:
        """A clip must start BEFORE the person appeared, not at detection time."""
        recorder = VideoRecorder(
            RecordingConfig(
                mode=RecordingMode.EVENT, pre_event_seconds=1.0, post_event_seconds=0.0
            ),
            OutputConfig(),
            tmp_path / "videos",
            fps=10.0,
            size=(320, 240),
            source_id="test",
        )
        for frame in frames[:15]:            # 15 quiet frames -> buffer holds 10
            recorder.process(frame, trigger=False)
        assert not recorder.is_recording

        recorder.process(frames[15], trigger=True, reason="john")
        assert recorder.is_recording
        clip_path = recorder.current_path
        recorder.close()

        # 10 buffered pre-roll frames + the triggering frame.
        assert clip_path is not None and clip_path.exists()
        assert "john" in clip_path.name

    def test_event_mode_writes_a_post_event_tail_then_stops(self, tmp_path: Path, frames) -> None:
        recorder = VideoRecorder(
            RecordingConfig(
                mode=RecordingMode.EVENT, pre_event_seconds=0.0, post_event_seconds=0.5
            ),
            OutputConfig(),
            tmp_path / "videos",
            fps=10.0,
            size=(320, 240),
            source_id="test",
        )
        recorder.process(frames[0], trigger=True, reason="john")
        assert recorder.state is RecorderState.RECORDING

        for frame in frames[1:4]:            # 0.5s @ 10fps == 5 frames of tail
            recorder.process(frame, trigger=False)
        assert recorder.state is RecorderState.TRAILING

        for frame in frames[4:8]:
            recorder.process(frame, trigger=False)
        assert recorder.state is RecorderState.IDLE
        assert len(recorder.clips) == 1

    def test_disabled_mode_writes_nothing(self, tmp_path: Path, frames) -> None:
        recorder = VideoRecorder(
            RecordingConfig(mode=RecordingMode.DISABLED),
            OutputConfig(),
            tmp_path / "videos",
            fps=10.0,
            size=(320, 240),
        )
        assert not recorder.enabled
        for frame in frames[:5]:
            recorder.process(frame, trigger=True)
        assert recorder.close() is None
        assert not (tmp_path / "videos").exists() or not list((tmp_path / "videos").iterdir())

    def test_annotated_writer_preserves_dimensions(self, tmp_path: Path, frames) -> None:
        writer = AnnotatedVideoWriter(
            OutputConfig(), tmp_path / "videos", fps=10.0, size=(320, 240), source_id="clip"
        )
        for frame in frames[:6]:
            writer.write(frame)
        path = writer.close()
        assert path is not None and path.exists()
        assert writer.frames == 6

    def test_output_fps_override_is_honoured(self, tmp_path: Path) -> None:
        writer = AnnotatedVideoWriter(
            OutputConfig(video_fps=5.0),
            tmp_path / "videos",
            fps=30.0,
            size=(320, 240),
            source_id="clip",
        )
        writer.write(person_scene(size=(320, 240)))
        assert writer.close() is not None


class TestRenderer:
    def test_recognized_label_has_name_title_and_score(self) -> None:
        renderer = Renderer(DisplayConfig())
        lines = renderer.label_lines(make_detection())
        assert lines[0] == "[17] John Doe"
        assert lines[1] == "Manager"
        assert lines[2] == "ReID: 0.73"

    def test_unknown_label_omits_the_title(self) -> None:
        renderer = Renderer(DisplayConfig())
        lines = renderer.label_lines(
            make_detection(
                identity_id=None, name="Unknown", title="", similarity=0.42,
                status=RecognitionStatus.UNKNOWN,
            )
        )
        assert lines == ["[17] Unknown", "ReID: 0.42"]

    def test_statuses_are_visually_distinguishable(self) -> None:
        renderer = Renderer(DisplayConfig())
        colors = {
            renderer.color_for(status)
            for status in (
                RecognitionStatus.RECOGNIZED,
                RecognitionStatus.LOW_CONFIDENCE,
                RecognitionStatus.UNKNOWN,
            )
        }
        assert len(colors) == 3

    def test_track_id_can_be_hidden(self) -> None:
        renderer = Renderer(DisplayConfig(show_track_id=False))
        assert renderer.label_lines(make_detection())[0] == "John Doe"

    def test_render_does_not_mutate_the_source_frame(self) -> None:
        renderer = Renderer(DisplayConfig())
        image = person_scene((30, 30, 30))
        original = image.copy()
        annotated = renderer.render(image, make_frame_result([make_detection()]))
        assert np.array_equal(image, original)
        assert not np.array_equal(annotated, original)

    def test_metrics_overlay_is_drawn_when_enabled(self) -> None:
        renderer = Renderer(DisplayConfig(show_metrics=True))
        image = person_scene((30, 30, 30))
        metrics = RenderMetrics(fps=27.4, persons=4, recognized=2, unknown=2, tracks=4)
        annotated = renderer.render(image, make_frame_result([]), metrics)
        assert not np.array_equal(annotated, image)
        assert "FPS: 27.4" in metrics.lines()[0]

    def test_a_box_at_the_frame_edge_does_not_crash(self) -> None:
        renderer = Renderer(DisplayConfig())
        detection = make_detection()
        detection.bbox = BBox(-50, -50, 700, 600)
        renderer.render(person_scene((30, 30, 30), size=(640, 480)),
                        make_frame_result([detection]))


class TestEvents:
    def test_events_are_dispatched_and_persisted(self, tmp_path: Path) -> None:
        log = tmp_path / "events.jsonl"
        received: list[Event] = []
        with EventManager(log_path=log) as manager:
            manager.subscribe(received.append)
            manager.emit(
                Event(
                    type=EventType.PERSON_RECOGNIZED,
                    source_id="webcam_0",
                    identity_id="john_doe",
                    identity_name="John Doe",
                    similarity=0.73,
                )
            )
        assert len(received) == 1
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        assert rows[0]["type"] == "person_recognized"
        assert rows[0]["similarity"] == 0.73

    def test_handlers_can_filter_by_type(self, tmp_path: Path) -> None:
        recognized: list[Event] = []
        with EventManager(log_path=None) as manager:
            manager.subscribe(recognized.append, EventType.PERSON_RECOGNIZED)
            manager.emit(Event(type=EventType.PERSON_RECOGNIZED, source_id="s"))
            manager.emit(Event(type=EventType.TRACK_ENDED, source_id="s"))
        assert len(recognized) == 1

    def test_a_failing_handler_does_not_break_processing(self) -> None:
        def explode(event: Event) -> None:
            raise RuntimeError("integration is down")

        seen: list[Event] = []
        with EventManager(log_path=None) as manager:
            manager.subscribe(explode)
            manager.subscribe(seen.append)
            manager.emit(Event(type=EventType.PERSON_DETECTED, source_id="s"))
        assert len(seen) == 1

    def test_history_is_queryable(self) -> None:
        with EventManager(log_path=None) as manager:
            for _ in range(3):
                manager.emit(Event(type=EventType.PERSON_DETECTED, source_id="s"))
            manager.emit(Event(type=EventType.TRACK_ENDED, source_id="s"))
            assert len(manager.history()) == 4
            assert len(manager.history(event_type=EventType.PERSON_DETECTED)) == 3
            assert len(manager.history(limit=2)) == 2

    def test_disabled_manager_records_nothing(self) -> None:
        manager = EventManager(enabled=False)
        manager.emit(Event(type=EventType.PERSON_DETECTED, source_id="s"))
        assert manager.history() == []


class TestRetention:
    def test_old_artefacts_are_removed_and_recent_ones_kept(
        self, config, tmp_path: Path
    ) -> None:
        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.snapshots_dir, paths.videos_dir)

        old = imwrite(paths.snapshots_dir / "old.jpg", person_scene(size=(32, 32)))
        new = imwrite(paths.snapshots_dir / "new.jpg", person_scene(size=(32, 32)))
        ancient = time.time() - 40 * 86400
        import os

        os.utime(old, (ancient, ancient))

        report = apply_retention(RetentionConfig(enabled=True, snapshots_days=30), paths)
        assert report.removed_files == 1
        assert not old.exists()
        assert new.exists()

    def test_dry_run_deletes_nothing(self, config, tmp_path: Path) -> None:
        import os

        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.snapshots_dir)
        old = imwrite(paths.snapshots_dir / "old.jpg", person_scene(size=(32, 32)))
        ancient = time.time() - 40 * 86400
        os.utime(old, (ancient, ancient))

        report = apply_retention(
            RetentionConfig(enabled=True, snapshots_days=30), paths, dry_run=True
        )
        assert report.removed_files == 1 and report.dry_run
        assert old.exists()

    def test_zero_days_means_keep_forever(self, config) -> None:
        import os

        paths = ProjectPaths.from_config(config)
        paths.ensure(paths.snapshots_dir)
        old = imwrite(paths.snapshots_dir / "old.jpg", person_scene(size=(32, 32)))
        ancient = time.time() - 4000 * 86400
        os.utime(old, (ancient, ancient))

        report = apply_retention(RetentionConfig(enabled=True, snapshots_days=0), paths)
        assert report.removed_files == 0
        assert old.exists()

    def test_disabled_retention_is_a_no_op(self, config) -> None:
        paths = ProjectPaths.from_config(config)
        assert apply_retention(RetentionConfig(enabled=False), paths).removed_files == 0
