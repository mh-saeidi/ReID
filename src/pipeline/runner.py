"""Stream runner: drives a source through the pipeline and produces output.

Everything with a side effect lives here -- rendering, the preview window,
snapshots, recording, metadata, events -- so :class:`ReIDPipeline` stays pure
and reusable.
"""

from __future__ import annotations

import contextlib
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2

from src.config.schema import RecognitionMode, RecordingMode, SnapshotMode
from src.core.types import DetectionResult, FrameResult, RecognitionStatus
from src.events.types import Event, EventType
from src.output.metadata import MetadataWriter, summarize
from src.output.recorder import AnnotatedVideoWriter, VideoRecorder
from src.output.renderer import Renderer, RenderMetrics
from src.output.snapshot import SnapshotWriter
from src.pipeline.debug import DebugRecorder
from src.pipeline.engine import Engine
from src.pipeline.metrics import Stopwatch
from src.pipeline.processor import ProcessOutcome, ReIDPipeline
from src.sources.base import BaseSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RunSummary:
    """What a completed run produced."""

    source_id: str
    frames: int
    detections: int
    recognized: int
    unknown: int
    elapsed_s: float
    fps: float
    identities: dict[str, int] = field(default_factory=dict)
    video_path: str | None = None
    metadata_path: str | None = None
    clips: list[str] = field(default_factory=list)
    snapshots: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "frames": self.frames,
            "detections": self.detections,
            "recognized": self.recognized,
            "unknown": self.unknown,
            "elapsed_s": round(self.elapsed_s, 2),
            "fps": round(self.fps, 2),
            "identities": self.identities,
            "video": self.video_path,
            "metadata": self.metadata_path,
            "clips": self.clips,
            "snapshots": self.snapshots,
            "metrics": self.metrics,
        }


class StreamRunner:
    """Runs one source to completion (or until interrupted)."""

    def __init__(
        self,
        engine: Engine,
        pipeline: ReIDPipeline,
        *,
        show_window: bool | None = None,
        save_video: bool | None = None,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> None:
        self._engine = engine
        self._pipeline = pipeline
        self._config = engine.config
        self._renderer = Renderer(
            engine.config.display,
            score_label=(
                "Face"
                if engine.config.recognition.mode is RecognitionMode.FACE
                else "ReID"
            ),
        )
        self._on_frame = on_frame
        self._show_window = (
            engine.config.display.show_window if show_window is None else show_window
        )
        self._save_video = engine.config.output.save_video if save_video is None else save_video
        self._interrupted = False

        self._snapshots = SnapshotWriter(
            engine.config.output, engine.paths.snapshots_dir, engine.paths.crops_dir
        )
        self._debug = DebugRecorder(engine.config.debug, engine.paths.debug_dir)
        self._snapshot_count = 0

    # ------------------------------------------------------------------- run
    def run(self, source: BaseSource) -> RunSummary:
        """Process every frame of ``source`` and return a summary."""
        info = source.open()
        self._pipeline.reset()
        self._engine.events.emit(
            Event(
                type=EventType.SOURCE_STARTED,
                source_id=info.source_id,
                payload={
                    "kind": info.kind,
                    "width": info.width,
                    "height": info.height,
                    "fps": info.fps,
                    "frames": info.frame_count,
                },
            )
        )
        logger.info(
            "Processing started",
            extra={
                "source": info.source_id,
                "size": f"{info.width}x{info.height}",
                "fps": round(info.fps, 2),
                "identities": len(self._engine.gallery.active_identities),
            },
        )

        video_writer = self._make_video_writer(info)
        recorder = self._make_recorder(info)
        metadata = (
            MetadataWriter(
                self._engine.paths.output_metadata_dir,
                info.source_id,
                overwrite=self._config.output.overwrite,
            )
            if self._config.output.save_metadata
            else None
        )

        started = time.perf_counter()
        frames: list[FrameResult] = []
        previous_handler = self._install_interrupt_handler()

        try:
            while not self._interrupted:
                frame = source.read()
                if frame is None:
                    break
                with Stopwatch(self._pipeline.metrics, "end_to_end"):
                    outcome = self._pipeline.process(frame)
                frames.append(outcome.result)

                # Rendering copies the whole frame and draws on it, so it only
                # happens when something will actually look at the result. The
                # previous loop rendered unconditionally, paying that cost even
                # headless with every output disabled.
                render_metrics = self._render_metrics(outcome.result)
                needs_annotation = (
                    self._show_window
                    or video_writer is not None
                    or recorder is not None
                    or self._snapshots.enabled
                    or self._debug.enabled
                )
                annotated = (
                    self._renderer.render(frame.image, outcome.result, render_metrics)
                    if needs_annotation
                    else None
                )
                self._emit_events(outcome)
                if self._snapshots.enabled:
                    self._write_snapshots(frame.image, annotated, outcome.result)

                if video_writer is not None and annotated is not None:
                    video_writer.write(annotated)
                if recorder is not None:
                    trigger, reason = self._recording_trigger(outcome.result)
                    recorder.process(
                        annotated if annotated is not None else frame.image,
                        trigger=trigger,
                        reason=reason,
                    )
                if metadata is not None:
                    metadata.write(outcome.result)
                if self._debug.enabled and annotated is not None:
                    self._debug.record(
                        frame.image, annotated, outcome.result,
                        self._pipeline.track_manager.tracks,
                    )
                if self._on_frame is not None:
                    self._on_frame(outcome.result)
                if self._show_window and annotated is not None and not self._display(annotated):
                    logger.info("Preview window closed; stopping")
                    break
        finally:
            self._restore_interrupt_handler(previous_handler)
            if self._show_window:
                cv2.destroyAllWindows()
            source.close()
            video_path = video_writer.close() if video_writer is not None else None
            if recorder is not None:
                recorder.close()
            metadata_path = metadata.close() if metadata is not None else None
            self._emit_final_track_events(info.source_id)
            self._engine.events.emit(
                Event(
                    type=EventType.SOURCE_ENDED,
                    source_id=info.source_id,
                    payload={"frames": len(frames)},
                )
            )

        elapsed = time.perf_counter() - started
        stats = summarize(frames)
        summary = RunSummary(
            source_id=info.source_id,
            frames=len(frames),
            detections=int(stats["detections"]),
            recognized=int(stats["recognized"]),
            unknown=int(stats["unknown"]),
            elapsed_s=elapsed,
            fps=(len(frames) / elapsed) if elapsed > 0 else 0.0,
            identities=dict(stats["identities"]),
            video_path=str(video_path) if video_path else None,
            metadata_path=str(metadata_path) if metadata_path else None,
            clips=[str(c.path) for c in (recorder.clips if recorder else [])],
            snapshots=self._snapshot_count,
            metrics=self._pipeline.metrics.snapshot(),
        )
        logger.info(
            "Processing finished",
            extra={
                "source": summary.source_id,
                "frames": summary.frames,
                "fps": round(summary.fps, 2),
                "recognized": summary.recognized,
                "unknown": summary.unknown,
            },
        )
        return summary

    # --------------------------------------------------------------- outputs
    def _make_video_writer(self, info) -> AnnotatedVideoWriter | None:
        if not self._save_video or info.kind in ("image", "directory"):
            return None
        from src.output.gst_recorder import make_video_writer_factory  # noqa: PLC0415

        return AnnotatedVideoWriter(
            self._config.output,
            self._engine.paths.videos_dir,
            fps=info.fps or 25.0,
            size=(info.width, info.height),
            source_id=info.source_id,
            writer_factory=make_video_writer_factory(self._config),
        )

    def _make_recorder(self, info) -> VideoRecorder | None:
        if self._config.recording.mode is RecordingMode.DISABLED:
            return None
        if info.kind in ("image", "directory"):
            logger.debug("Recording is not applicable to still-image sources")
            return None
        from src.output.gst_recorder import make_video_writer_factory  # noqa: PLC0415

        return VideoRecorder(
            self._config.recording,
            self._config.output,
            self._engine.paths.videos_dir,
            fps=info.fps or 25.0,
            size=(info.width, info.height),
            source_id=info.source_id,
            writer_factory=make_video_writer_factory(self._config),
        )

    def _recording_trigger(self, result: FrameResult) -> tuple[bool, str]:
        recording = self._config.recording
        if recording.mode is RecordingMode.CONTINUOUS:
            return True, ""
        if recording.save_when_identity_detected:
            for detection in result.recognized:
                return True, detection.identity_id or "recognized"
        if recording.save_unknown and result.unknown:
            return True, "unknown"
        return False, ""

    def _write_snapshots(self, image, annotated, result: FrameResult) -> None:
        if not self._snapshots.enabled:
            return
        events_only = self._config.output.snapshot_mode is SnapshotMode.EVENTS_ONLY
        for detection in result.detections:
            is_event = self._is_snapshot_event(detection) if events_only else False
            record = self._snapshots.save(
                image, detection, result, annotated=annotated, is_event=is_event
            )
            if record is None:
                continue
            self._snapshot_count += 1
            self._engine.events.emit(
                Event(
                    type=EventType.SNAPSHOT_SAVED,
                    source_id=result.source_id,
                    frame_index=result.frame_index,
                    track_id=detection.track_id,
                    identity_id=detection.identity_id,
                    identity_name=detection.identity_name,
                    similarity=detection.reid_similarity,
                    bbox=detection.bbox.to_list(),
                    payload={"path": str(record.image_path)},
                )
            )

    def _identity_settled(self, state) -> bool:
        """Has this track seen enough frames for its identity to mean anything?

        The stabilizer needs ``minimum_recognized_frames`` observations before
        it will name anybody, so a brand-new track always reads as Unknown for
        its first frames. Acting on that would file a recognised person under
        "unknown", so conclusions wait until the window has filled.
        """
        if state is None:
            return True
        stability = self._config.tracking.identity_stability
        if not (self._config.tracking.enabled and stability.enabled):
            return True
        return state.frames_seen >= stability.minimum_recognized_frames

    def _is_snapshot_event(self, detection: DetectionResult) -> bool:
        """``events_only``: save once per track, when its identity has settled."""
        state = self._pipeline.track_manager.get(detection.track_id or -1)
        if detection.recognition_status is RecognitionStatus.PENDING:
            return False
        recognized = detection.effective.is_recognized
        if not recognized and not self._identity_settled(state):
            return False
        if state is None:
            return True
        key = f"snapshot:{detection.identity_id or 'unknown'}"
        if key in state.events_emitted:
            return False
        state.events_emitted.add(key)
        return True

    # ---------------------------------------------------------------- events
    def _emit_events(self, outcome: ProcessOutcome) -> None:
        emit_frame_events(self._engine, self._pipeline, outcome)

    @staticmethod
    def _detection_for(result: FrameResult, track_id: int) -> DetectionResult | None:
        for detection in result.detections:
            if detection.track_id == track_id:
                return detection
        return None

    def _emit_final_track_events(self, source_id: str) -> None:
        for state in self._pipeline.track_manager.reset():
            if state.identity_id:
                self._engine.events.emit(
                    Event(
                        type=EventType.PERSON_LOST,
                        source_id=source_id,
                        frame_index=state.last_seen_frame,
                        track_id=state.track_id,
                        identity_id=state.identity_id,
                        similarity=state.identity_similarity,
                    )
                )

    # -------------------------------------------------------------- display
    def _render_metrics(self, result: FrameResult) -> RenderMetrics | None:
        if not self._config.display.show_metrics:
            return None
        metrics = self._pipeline.metrics
        return RenderMetrics(
            fps=metrics.fps,
            persons=len(result.detections),
            recognized=len(result.recognized),
            unknown=len(result.unknown),
            tracks=self._pipeline.track_manager.active_count,
            detector_ms=metrics.stage_ms("detector"),
            reid_ms=metrics.stage_ms("reid"),
            total_ms=metrics.stage_ms("total"),
        )

    def _display(self, annotated) -> bool:
        """Show the preview window. Returns False when the user asked to quit."""
        window = self._config.display.window_name
        try:
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            logger.warning(
                "Preview window unavailable (%s); continuing headless. "
                "Set display.show_window: false to silence this.",
                exc,
            )
            self._show_window = False
            return True
        if key in (ord("q"), 27):
            return False
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:  # pragma: no cover - backend dependent
            pass
        return True

    # ------------------------------------------------------------- interrupt
    def _install_interrupt_handler(self):
        def handler(signum, frame):  # noqa: ANN001, ARG001
            if self._interrupted:  # second Ctrl-C: let it propagate
                raise KeyboardInterrupt
            logger.info("Interrupt received; finishing the current frame and closing files")
            self._interrupted = True

        try:
            return signal.signal(signal.SIGINT, handler)
        except ValueError:  # pragma: no cover - not on the main thread
            return None

    @staticmethod
    def _restore_interrupt_handler(previous) -> None:
        if previous is not None:
            with contextlib.suppress(ValueError):  # not on the main thread
                signal.signal(signal.SIGINT, previous)


def emit_frame_events(engine, pipeline, outcome: ProcessOutcome) -> None:
    events = engine.events
    if not events.enabled:
        return
    result = outcome.result

    for track_id in outcome.new_tracks:
        events.emit(
            Event(
                type=EventType.TRACK_STARTED,
                source_id=result.source_id,
                frame_index=result.frame_index,
                track_id=track_id,
            )
        )

    for detection in result.detections:
        state = pipeline.track_manager.get(detection.track_id or -1)
        emitted = state.events_emitted if state is not None else set()

        if "detected" not in emitted:
            emitted.add("detected")
            events.emit(
                Event(
                    type=EventType.PERSON_DETECTED,
                    source_id=result.source_id,
                    frame_index=result.frame_index,
                    track_id=detection.track_id,
                    bbox=detection.bbox.to_list(),
                    detector_confidence=detection.detector_confidence,
                )
            )
        # Only report an unknown person once the stabilizer has had enough
        # frames to decide; otherwise every new track would be announced as
        # unknown for its first few frames.
        if (
            detection.recognition_status
            in (RecognitionStatus.UNKNOWN, RecognitionStatus.REJECTED)
            and "unknown" not in emitted
            and _identity_settled(engine.config, state)
        ):
            emitted.add("unknown")
            events.emit(
                Event(
                    type=EventType.UNKNOWN_PERSON_DETECTED,
                    source_id=result.source_id,
                    frame_index=result.frame_index,
                    track_id=detection.track_id,
                    similarity=detection.reid_similarity,
                    bbox=detection.bbox.to_list(),
                    detector_confidence=detection.detector_confidence,
                )
            )

    for transition in outcome.transitions:
        detection = _detection_for(result, transition.state.track_id)
        events.emit(
            Event(
                type=(
                    EventType.PERSON_RECOGNIZED
                    if transition.kind == "identified"
                    else EventType.IDENTITY_CHANGED
                ),
                source_id=result.source_id,
                frame_index=result.frame_index,
                track_id=transition.state.track_id,
                identity_id=transition.state.identity_id,
                identity_name=detection.identity_name if detection else None,
                identity_title=detection.identity_title if detection else None,
                similarity=transition.state.identity_similarity,
                bbox=detection.bbox.to_list() if detection else None,
                payload={"previous_identity": transition.previous_identity}
                if transition.previous_identity
                else {},
            )
        )

    for track_id in outcome.ended_tracks:
        events.emit(
            Event(
                type=EventType.TRACK_ENDED,
                source_id=result.source_id,
                frame_index=result.frame_index,
                track_id=track_id,
            )
        )


def _detection_for(result: FrameResult, track_id: int) -> DetectionResult | None:
    for detection in result.detections:
        if detection.track_id == track_id:
            return detection
    return None


def _identity_settled(config, state) -> bool:
    """Has this track seen enough frames for its identity to mean anything?

    The stabilizer needs ``minimum_recognized_frames`` observations before it
    will name anybody, so a brand-new track always reads as Unknown for its
    first frames. Acting on that would file a recognised person under
    "unknown", so conclusions wait until the window has filled.
    """
    if state is None:
        return True
    stability = config.tracking.identity_stability
    if not (config.tracking.enabled and stability.enabled):
        return True
    return state.frames_seen >= stability.minimum_recognized_frames
