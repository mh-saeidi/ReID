"""All frame-derived output in one place.

Snapshots, metadata, annotated video, event clips and debug artefacts were
previously interleaved with inference in the frame loop. Collecting them here
means the synchronous and asynchronous runners execute exactly the same output
logic -- the only difference is which thread calls :meth:`OutputSink.handle`.

Rendering is lazy on purpose. The old loop copied every full frame and drew on
it even when running headless with no video output, which is pure cost on an
embedded board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config.schema import AppConfig, RecordingMode, SnapshotMode
from src.core.types import DetectionResult, FrameResult, RecognitionStatus
from src.events.types import Event, EventType
from src.output.metadata import MetadataWriter
from src.output.recorder import AnnotatedVideoWriter, VideoRecorder
from src.output.renderer import Renderer, RenderMetrics
from src.output.snapshot import SnapshotWriter
from src.pipeline.debug import DebugRecorder
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class OutputTask:
    """One frame's worth of output work."""

    frame_image: np.ndarray
    result: FrameResult
    annotated: np.ndarray | None = None
    metrics: RenderMetrics | None = None
    tracks: dict[int, Any] | None = None
    trigger: bool = False
    trigger_reason: str = ""
    snapshot_events: dict[str, bool] = field(default_factory=dict)
    """detection_id -> is_event, decided on the inference thread where the
    track state lives."""
    allow_visualization: bool = True
    """Cleared when the pipeline is shedding load."""


@dataclass(slots=True)
class OutputStats:
    frames_written: int = 0
    snapshots: int = 0
    video_frames: int = 0
    metadata_records: int = 0
    dropped_visualization: int = 0
    errors: int = 0


class OutputSink:
    """Owns every writer and performs the actual output work.

    Not thread-safe: exactly one thread must call :meth:`handle`. The
    asynchronous dispatcher guarantees that by using a single worker.
    """

    def __init__(
        self,
        config: AppConfig,
        paths,
        events,
        *,
        renderer: Renderer,
        save_video: bool,
        source_info=None,
    ) -> None:
        self._config = config
        self._paths = paths
        self._events = events
        self._renderer = renderer
        self._save_video = save_video
        self._info = source_info

        self._snapshots = SnapshotWriter(
            config.output, paths.snapshots_dir, paths.crops_dir
        )
        self._debug = DebugRecorder(config.debug, paths.debug_dir)
        self._video: AnnotatedVideoWriter | None = None
        self._recorder: VideoRecorder | None = None
        self._metadata: MetadataWriter | None = None
        self.stats = OutputStats()

    # ------------------------------------------------------------- lifecycle
    def open(self, info) -> None:
        """Create the writers this source and configuration require."""
        self._info = info
        config = self._config
        is_still = info.kind in ("image", "directory")

        if self._save_video and not is_still:
            self._video = AnnotatedVideoWriter(
                config.output,
                self._paths.videos_dir,
                fps=info.fps or 25.0,
                size=(info.width, info.height),
                source_id=info.source_id,
            )
        if config.recording.mode is not RecordingMode.DISABLED and not is_still:
            self._recorder = build_recorder(
                config, self._paths.videos_dir,
                fps=info.fps or 25.0,
                size=(info.width, info.height),
                source_id=info.source_id,
            )
        if config.output.save_metadata:
            self._metadata = MetadataWriter(
                self._paths.output_metadata_dir,
                info.source_id,
                overwrite=config.output.overwrite,
            )

    @property
    def needs_annotation(self) -> bool:
        """Does anything actually consume an annotated frame?

        Rendering costs a full-frame copy plus drawing, so it is skipped
        entirely when nothing will look at the result.
        """
        config = self._config
        return bool(
            self._video is not None
            or self._recorder is not None
            or (self._snapshots.enabled and config.output.save_snapshots)
            or config.debug.enabled
        )

    @property
    def snapshot_writer(self) -> SnapshotWriter:
        return self._snapshots

    @property
    def recorder(self) -> VideoRecorder | None:
        return self._recorder

    def render(self, image: np.ndarray, result: FrameResult,
               metrics: RenderMetrics | None) -> np.ndarray:
        return self._renderer.render(image, result, metrics)

    # ---------------------------------------------------------------- output
    def handle(self, task: OutputTask) -> None:
        """Write everything this frame produced."""
        annotated = task.annotated
        if annotated is None and self.needs_annotation and task.allow_visualization:
            annotated = self.render(task.frame_image, task.result, task.metrics)

        try:
            if self._snapshots.enabled:
                self._write_snapshots(task, annotated)
            if self._video is not None and task.allow_visualization and annotated is not None:
                self._video.write(annotated)
                self.stats.video_frames += 1
            if self._recorder is not None:
                source = annotated if annotated is not None else task.frame_image
                self._recorder.process(
                    source, trigger=task.trigger, reason=task.trigger_reason
                )
            if self._metadata is not None:
                self._metadata.write(task.result)
                self.stats.metadata_records += 1
            if self._debug.enabled and annotated is not None:
                self._debug.record(task.frame_image, annotated, task.result, task.tracks)
        except Exception as exc:  # noqa: BLE001 - output must never kill the run
            self.stats.errors += 1
            logger.error(
                "Output stage failed for a frame",
                extra={"frame": task.result.frame_index, "error": str(exc)},
            )
        self.stats.frames_written += 1

    def _write_snapshots(self, task: OutputTask, annotated: np.ndarray | None) -> None:
        for detection in task.result.detections:
            is_event = task.snapshot_events.get(detection.detection_id, False)
            record = self._snapshots.save(
                task.frame_image, detection, task.result,
                annotated=annotated, is_event=is_event,
            )
            if record is None:
                continue
            self.stats.snapshots += 1
            self._events.emit(
                Event(
                    type=EventType.SNAPSHOT_SAVED,
                    source_id=task.result.source_id,
                    frame_index=task.result.frame_index,
                    track_id=detection.track_id,
                    identity_id=detection.identity_id,
                    identity_name=detection.identity_name,
                    similarity=detection.reid_similarity,
                    bbox=detection.bbox.to_list(),
                    payload={"path": str(record.image_path)},
                )
            )

    # ------------------------------------------------------------- shutdown
    def close(self) -> dict[str, Any]:
        """Flush and release every writer, reporting what was produced."""
        video_path = self._video.close() if self._video is not None else None
        clips: list[str] = []
        if self._recorder is not None:
            self._recorder.close()
            clips = [str(c.path) for c in self._recorder.clips]
        metadata_path = self._metadata.close() if self._metadata is not None else None
        return {
            "video": str(video_path) if video_path else None,
            "metadata": str(metadata_path) if metadata_path else None,
            "clips": clips,
            "snapshots": self.stats.snapshots,
            "errors": self.stats.errors,
        }


def build_recorder(config: AppConfig, directory: Path, *, fps: float,
                   size: tuple[int, int], source_id: str) -> VideoRecorder:
    """Create the recorder, preferring a hardware encoder where available."""
    from src.output.gst_recorder import make_video_writer_factory  # noqa: PLC0415

    return VideoRecorder(
        config.recording,
        config.output,
        directory,
        fps=fps,
        size=size,
        source_id=source_id,
        writer_factory=make_video_writer_factory(config),
    )


def snapshot_event_map(
    result: FrameResult,
    track_manager,
    config: AppConfig,
) -> dict[str, bool]:
    """Decide per detection whether this frame is its "event" frame.

    Evaluated where the track state lives (the inference thread) rather than in
    the output worker, so the decision cannot race with track updates.
    """
    if config.output.snapshot_mode is not SnapshotMode.EVENTS_ONLY:
        return {}

    stability = config.tracking.identity_stability
    settled_required = (
        stability.minimum_recognized_frames
        if config.tracking.enabled and stability.enabled
        else 0
    )
    events: dict[str, bool] = {}
    for detection in result.detections:
        if detection.recognition_status is RecognitionStatus.PENDING:
            events[detection.detection_id] = False
            continue
        state = track_manager.get(detection.track_id or -1)
        recognized = detection.effective.is_recognized
        settled = state is None or state.frames_seen >= settled_required
        if not recognized and not settled:
            events[detection.detection_id] = False
            continue
        if state is None:
            events[detection.detection_id] = True
            continue
        key = f"snapshot:{detection.identity_id or 'unknown'}"
        if key in state.events_emitted:
            events[detection.detection_id] = False
        else:
            state.events_emitted.add(key)
            events[detection.detection_id] = True
    return events


def recording_trigger(config: AppConfig, result: FrameResult) -> tuple[bool, str]:
    """Does this frame satisfy the configured recording trigger?"""
    recording = config.recording
    if recording.mode is RecordingMode.CONTINUOUS:
        return True, ""
    if recording.save_when_identity_detected:
        for detection in result.recognized:
            return True, detection.identity_id or "recognized"
    if recording.save_unknown and result.unknown:
        return True, "unknown"
    return False, ""


def detection_is_event(detection: DetectionResult) -> bool:
    return detection.effective.is_recognized
