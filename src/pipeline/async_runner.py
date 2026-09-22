"""Asynchronous capture -> inference -> output pipeline.

    Camera -> capture queue -> Detection + Tracking + Recognition
           -> result queue  -> Rendering / Recording / Storage

Three properties matter more than raw throughput:

**Bounded queues.** Every queue has a fixed size. An unbounded queue in front of
a slow consumer does not prevent stalls, it converts them into latency and
memory growth -- on a live camera that means the operator is looking at a scene
from several seconds ago.

**Newest-frame preference.** When the inference stage falls behind a live
source, the useful frame is the most recent one. Older queued frames are
discarded and counted, rather than processed late.

**Clean shutdown.** Threads are joined with a timeout and writers are flushed,
so a Ctrl-C does not truncate a recording or lose the event log.

Track and identity state stay on a single thread: the inference worker owns the
:class:`~src.pipeline.processor.ReIDPipeline` exclusively, so no lock is needed
around track state and no race is possible around the identity hold.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.core.types import Frame, FrameResult
from src.events.types import Event, EventType
from src.output.async_dispatch import AsyncOutputDispatcher
from src.output.sink import OutputSink, OutputTask, recording_trigger, snapshot_event_map
from src.pipeline.engine import Engine
from src.pipeline.metrics import Stopwatch
from src.pipeline.processor import ProcessOutcome, ReIDPipeline
from src.sources.base import BaseSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CaptureStats:
    frames_read: int = 0
    frames_dropped: int = 0
    read_seconds: float = 0.0
    ended: bool = False
    error: str = ""

    @property
    def drop_rate_percent(self) -> float:
        total = self.frames_read + self.frames_dropped
        return 100.0 * self.frames_dropped / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "drop_rate_percent": round(self.drop_rate_percent, 3),
            "average_read_ms": round(
                1000.0 * self.read_seconds / self.frames_read, 3
            ) if self.frames_read else 0.0,
            "error": self.error,
        }


class CaptureThread:
    """Reads frames into a bounded queue, preferring the newest when full."""

    def __init__(self, source: BaseSource, capacity: int, *, drop_stale: bool) -> None:
        self._source = source
        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=max(1, capacity))
        self._drop_stale = drop_stale
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = CaptureStats()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                frame = self._source.read()
                self.stats.read_seconds += time.perf_counter() - started
                if frame is None:
                    self.stats.ended = True
                    break
                self.stats.frames_read += 1
                self._offer(frame)
        except Exception as exc:  # noqa: BLE001 - surface, do not crash silently
            self.stats.error = str(exc)
            logger.error("Capture thread failed: %s", exc)
        finally:
            # Sentinel so the consumer terminates even on an error path.
            with contextlib.suppress(queue.Full):  # consumer already gone
                self._queue.put(None, timeout=1.0)

    def _offer(self, frame: Frame) -> None:
        if not self._drop_stale:
            self._queue.put(frame)
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Discard the oldest queued frame rather than the newest one: on a
            # live source the latest frame is the only one worth showing.
            try:
                self._queue.get_nowait()
                self.stats.frames_dropped += 1
            except queue.Empty:  # pragma: no cover - raced with the consumer
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:  # pragma: no cover
                self.stats.frames_dropped += 1

    def read(self, timeout: float = 1.0) -> Frame | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():  # pragma: no cover - blocked in read()
                logger.warning("Capture thread did not stop within the timeout")
        self._thread = None


@dataclass(slots=True)
class AsyncRunSummary:
    """What an asynchronous run produced."""

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
    capture: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

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
            "capture": self.capture,
            "output": self.output,
            "metrics": self.metrics,
        }


class AsyncStreamRunner:
    """Drives a source through the pipeline with capture and output off-thread.

    The preview window stays on the calling thread: several GUI backends
    (notably macOS) require it, and it is also the natural place to shed frames
    when display is the bottleneck.
    """

    def __init__(
        self,
        engine: Engine,
        pipeline: ReIDPipeline,
        *,
        show_window: bool | None = None,
        save_video: bool | None = None,
        on_frame: Callable[[FrameResult], None] | None = None,
    ) -> None:
        from src.config.schema import RecognitionMode  # noqa: PLC0415
        from src.output.renderer import Renderer  # noqa: PLC0415

        self._engine = engine
        self._pipeline = pipeline
        self._config = engine.config
        self._on_frame = on_frame
        self._show_window = (
            engine.config.display.show_window if show_window is None else show_window
        )
        self._save_video = (
            engine.config.output.save_video if save_video is None else save_video
        )
        self._renderer = Renderer(
            engine.config.display,
            score_label=(
                "Face"
                if engine.config.recognition.mode is RecognitionMode.FACE
                else "ReID"
            ),
        )
        self._interrupted = False

    def run(self, source: BaseSource) -> AsyncRunSummary:
        config = self._config
        pipeline_config = config.pipeline
        info = source.open()
        self._pipeline.reset()

        sink = OutputSink(
            config,
            self._engine.paths,
            self._engine.events,
            renderer=self._renderer,
            save_video=self._save_video,
        )
        sink.open(info)
        dispatcher = AsyncOutputDispatcher(
            sink,
            queue_size=pipeline_config.output_queue_size,
            policy=pipeline_config.overload_policy,
            workers=pipeline_config.output_workers,
        )
        capture = CaptureThread(
            source,
            pipeline_config.capture_queue_size,
            drop_stale=pipeline_config.drop_stale_frames and info.is_stream,
        )

        self._engine.events.emit(
            Event(
                type=EventType.SOURCE_STARTED,
                source_id=info.source_id,
                payload={
                    "kind": info.kind, "width": info.width, "height": info.height,
                    "fps": info.fps, "frames": info.frame_count, "async": True,
                },
            )
        )
        logger.info(
            "Asynchronous processing started",
            extra={
                "source": info.source_id,
                "size": f"{info.width}x{info.height}",
                "capture_queue": pipeline_config.capture_queue_size,
                "output_queue": pipeline_config.output_queue_size,
                "drop_stale": pipeline_config.drop_stale_frames,
            },
        )

        metrics = self._pipeline.metrics
        started = time.perf_counter()
        frames: list[FrameResult] = []
        dispatcher.start()
        capture.start()

        try:
            while not self._interrupted:
                frame = capture.read(timeout=0.5)
                if frame is None:
                    if capture.stats.ended:
                        break
                    continue

                with Stopwatch(metrics, "end_to_end"):
                    outcome = self._pipeline.process(frame)
                    self._dispatch(outcome, frame, sink, dispatcher, metrics)

                frames.append(outcome.result)
                metrics.queue_depth = capture.depth + dispatcher.depth
                metrics.record_drop(0)

                if self._on_frame is not None:
                    self._on_frame(outcome.result)
                if self._show_window and not self._display(frame, outcome, sink):
                    break
        except KeyboardInterrupt:  # pragma: no cover - interactive
            logger.info("Interrupt received; shutting the pipeline down")
        finally:
            capture.stop(timeout=pipeline_config.shutdown_timeout_s)
            dispatcher.stop(timeout=pipeline_config.shutdown_timeout_s)
            if self._show_window:
                import cv2  # noqa: PLC0415

                cv2.destroyAllWindows()
            source.close()
            closed = sink.close()
            self._engine.events.emit(
                Event(
                    type=EventType.SOURCE_ENDED,
                    source_id=info.source_id,
                    payload={"frames": len(frames)},
                )
            )

        elapsed = time.perf_counter() - started
        metrics.dropped_frames = capture.stats.frames_dropped
        from src.output.metadata import summarize  # noqa: PLC0415

        stats = summarize(frames)
        return AsyncRunSummary(
            source_id=info.source_id,
            frames=len(frames),
            detections=int(stats["detections"]),
            recognized=int(stats["recognized"]),
            unknown=int(stats["unknown"]),
            elapsed_s=elapsed,
            fps=(len(frames) / elapsed) if elapsed > 0 else 0.0,
            identities=dict(stats["identities"]),
            video_path=closed["video"],
            metadata_path=closed["metadata"],
            clips=closed["clips"],
            snapshots=closed["snapshots"],
            metrics=metrics.snapshot(),
            capture=capture.stats.to_dict(),
            output=dispatcher.stats.to_dict(),
        )

    # --------------------------------------------------------------- helpers
    def _dispatch(self, outcome: ProcessOutcome, frame: Frame, sink: OutputSink,
                  dispatcher: AsyncOutputDispatcher, metrics) -> None:
        """Hand this frame's output work to the worker."""
        result = outcome.result
        # Decided here, on the thread that owns the track state, so the
        # snapshot/event decision cannot race with track updates.
        snapshot_events = snapshot_event_map(
            result, self._pipeline.track_manager, self._config
        )
        trigger, reason = recording_trigger(self._config, result)
        must_persist = any(snapshot_events.values()) or trigger

        from src.pipeline.runner import emit_frame_events  # noqa: PLC0415

        emit_frame_events(self._engine, self._pipeline, outcome)

        task = OutputTask(
            frame_image=frame.image,
            result=result,
            metrics=self._render_metrics(result, metrics),
            tracks=dict(self._pipeline.track_manager.tracks) if self._config.debug.enabled else None,
            trigger=trigger,
            trigger_reason=reason,
            snapshot_events=snapshot_events,
        )
        dispatcher.submit(task, must_persist=must_persist)

    def _render_metrics(self, result: FrameResult, metrics):
        if not self._config.display.show_metrics:
            return None
        from src.output.renderer import RenderMetrics  # noqa: PLC0415

        return RenderMetrics(
            fps=metrics.fps,
            persons=len(result.detections),
            recognized=len(result.recognized),
            unknown=len(result.unknown),
            tracks=self._pipeline.track_manager.active_count,
            detector_ms=metrics.stage_ms("detector"),
            reid_ms=metrics.stage_ms("encoder"),
            total_ms=metrics.stage_ms("end_to_end"),
        )

    def _display(self, frame: Frame, outcome: ProcessOutcome, sink: OutputSink) -> bool:
        import cv2  # noqa: PLC0415

        annotated = sink.render(
            frame.image, outcome.result, self._render_metrics(
                outcome.result, self._pipeline.metrics
            )
        )
        try:
            cv2.imshow(self._config.display.window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            logger.warning("Preview window unavailable (%s); continuing headless", exc)
            self._show_window = False
            return True
        return key not in (ord("q"), 27)
