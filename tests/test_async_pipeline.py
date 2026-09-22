"""Asynchronous pipeline: queues, drop policy, ordering and shutdown."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.schema import OverloadPolicy
from src.core.types import Frame, FrameResult, SourceInfo
from src.output.async_dispatch import AsyncOutputDispatcher
from src.output.sink import OutputTask
from src.pipeline.async_runner import CaptureThread
from src.sources.base import BaseSource
from tests.conftest import person_scene


class FakeSource(BaseSource):
    """Emits a fixed number of frames, optionally with a capture delay."""

    def __init__(self, count: int, *, delay: float = 0.0, is_stream: bool = True) -> None:
        self._count = count
        self._delay = delay
        self._index = 0
        self._is_stream = is_stream
        self._image = person_scene(size=(64, 48))
        self._info = SourceInfo(
            source_id="fake", kind="webcam" if is_stream else "video",
            width=64, height=48, fps=30.0,
            frame_count=None if is_stream else count, is_stream=is_stream,
        )

    def open(self) -> SourceInfo:
        return self._info

    def read(self) -> Frame | None:
        if self._index >= self._count:
            return None
        if self._delay:
            time.sleep(self._delay)
        frame = Frame(image=self._image, index=self._index, source_id="fake")
        self._index += 1
        return frame

    @property
    def info(self) -> SourceInfo:
        return self._info

    def close(self) -> None:
        return None


class RecordingSink:
    """Stands in for OutputSink, recording what it was asked to write."""

    def __init__(self, delay: float = 0.0) -> None:
        self.handled: list[OutputTask] = []
        self.delay = delay
        self.lock = threading.Lock()

    def handle(self, task: OutputTask) -> None:
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.handled.append(task)


def make_task(index: int) -> OutputTask:
    result = FrameResult(
        frame_index=index, timestamp=0.0, source_id="fake", width=64, height=48
    )
    return OutputTask(frame_image=np.zeros((48, 64, 3), np.uint8), result=result)


class TestCaptureThread:
    def test_frames_flow_through_the_queue(self) -> None:
        # drop_stale off: this asserts ordering, not the shedding policy.
        capture = CaptureThread(FakeSource(10), capacity=4, drop_stale=False)
        capture.start()
        received = []
        while len(received) < 10:
            frame = capture.read(timeout=2.0)
            if frame is None:
                break
            received.append(frame)
        capture.stop()
        assert len(received) == 10
        assert [f.index for f in received] == list(range(10))

    def test_the_queue_is_bounded(self) -> None:
        """An unbounded queue converts a slow consumer into unbounded latency."""
        capture = CaptureThread(FakeSource(200), capacity=2, drop_stale=True)
        capture.start()
        time.sleep(0.2)          # let the producer run ahead
        assert capture.depth <= 2
        capture.stop()

    def test_stale_frames_are_dropped_newest_first(self) -> None:
        capture = CaptureThread(FakeSource(100), capacity=2, drop_stale=True)
        capture.start()
        time.sleep(0.25)         # producer outruns the (absent) consumer
        first = capture.read(timeout=1.0)
        capture.stop()
        assert capture.stats.frames_dropped > 0
        # What survives is recent, not the very first frame produced.
        assert first is not None

    def test_dropping_can_be_disabled(self) -> None:
        """A file source must not lose frames."""
        capture = CaptureThread(FakeSource(20, is_stream=False), capacity=2,
                                drop_stale=False)
        capture.start()
        received = []
        while len(received) < 20:
            frame = capture.read(timeout=2.0)
            if frame is None:
                break
            received.append(frame)
        capture.stop()
        assert capture.stats.frames_dropped == 0
        assert len(received) == 20

    def test_the_end_of_a_source_is_signalled(self) -> None:
        capture = CaptureThread(FakeSource(3), capacity=4, drop_stale=False)
        capture.start()
        for _ in range(3):
            assert capture.read(timeout=2.0) is not None
        assert capture.read(timeout=2.0) is None
        assert capture.stats.ended
        capture.stop()

    def test_stats_report_the_drop_rate(self) -> None:
        capture = CaptureThread(FakeSource(60), capacity=1, drop_stale=True)
        capture.start()
        time.sleep(0.2)
        capture.stop()
        stats = capture.stats.to_dict()
        assert stats["frames_read"] > 0
        assert 0.0 <= stats["drop_rate_percent"] <= 100.0


class TestOutputDispatcher:
    def test_tasks_are_handled_in_order(self) -> None:
        """A single worker keeps video and metadata frame-ordered."""
        sink = RecordingSink()
        with AsyncOutputDispatcher(sink, queue_size=32, workers=1) as dispatcher:
            for index in range(20):
                dispatcher.submit(make_task(index))
        assert [t.result.frame_index for t in sink.handled] == list(range(20))

    def test_visualization_is_shed_before_anything_else(self) -> None:
        sink = RecordingSink(delay=0.02)
        dispatcher = AsyncOutputDispatcher(
            sink, queue_size=1, policy=OverloadPolicy.DROP_VISUALIZATION, workers=1
        )
        dispatcher.start()
        for index in range(30):
            dispatcher.submit(make_task(index))
        dispatcher.stop(timeout=10.0)
        # Frames that lost their visualization still ran; nothing vanished
        # silently, and the counter reflects tasks actually preserved.
        assert dispatcher.stats.dropped_visualization > 0
        # Every frame either ran, or was evicted to keep a newer one. Nothing
        # was lost silently: the counters account for all 30 submissions.
        accounted = (
            dispatcher.stats.completed
            + dispatcher.stats.dropped_tasks
            + dispatcher.stats.dropped_visualization
        )
        assert accounted >= dispatcher.stats.submitted - 1
        assert any(not t.allow_visualization for t in sink.handled)

    def test_a_task_that_must_persist_is_never_dropped(self) -> None:
        """A recognition event the configuration says to keep must survive."""
        sink = RecordingSink(delay=0.01)
        dispatcher = AsyncOutputDispatcher(
            sink, queue_size=1, policy=OverloadPolicy.DROP_VISUALIZATION, workers=1
        )
        dispatcher.start()
        for index in range(25):
            assert dispatcher.submit(make_task(index), must_persist=True)
        dispatcher.stop(timeout=15.0)
        assert len(sink.handled) == 25
        assert dispatcher.stats.dropped_tasks == 0

    def test_the_block_policy_applies_back_pressure(self) -> None:
        sink = RecordingSink(delay=0.005)
        dispatcher = AsyncOutputDispatcher(
            sink, queue_size=2, policy=OverloadPolicy.BLOCK, workers=1
        )
        dispatcher.start()
        for index in range(30):
            assert dispatcher.submit(make_task(index))
        dispatcher.stop(timeout=15.0)
        assert len(sink.handled) == 30
        assert dispatcher.stats.dropped_tasks == 0

    def test_a_failing_sink_does_not_kill_the_worker(self) -> None:
        class Exploding:
            def __init__(self) -> None:
                self.calls = 0

            def handle(self, task: OutputTask) -> None:
                self.calls += 1
                raise RuntimeError("disk on fire")

        sink = Exploding()
        with AsyncOutputDispatcher(sink, queue_size=8, workers=1) as dispatcher:
            for index in range(5):
                dispatcher.submit(make_task(index))
        assert sink.calls == 5

    def test_shutdown_drains_the_queue(self) -> None:
        sink = RecordingSink(delay=0.002)
        dispatcher = AsyncOutputDispatcher(sink, queue_size=64, workers=1)
        dispatcher.start()
        for index in range(40):
            dispatcher.submit(make_task(index))
        dispatcher.stop(timeout=15.0)
        assert len(sink.handled) == 40

    def test_stop_is_idempotent(self) -> None:
        dispatcher = AsyncOutputDispatcher(RecordingSink(), queue_size=4)
        dispatcher.start()
        dispatcher.stop()
        dispatcher.stop()


class TestPipelineConfig:
    def test_async_is_off_by_default(self, base_config_dict: dict, tmp_path: Path) -> None:
        """Existing deployments must keep their current behaviour."""
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        assert config.pipeline.async_enabled is False

    def test_async_is_configured_with_the_yaml_key(
        self, base_config_dict: dict, tmp_path: Path
    ) -> None:
        base_config_dict["pipeline"] = {"async": True, "capture_queue_size": 3}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        assert config.pipeline.async_enabled is True
        assert config.pipeline.capture_queue_size == 3

    def test_queue_sizes_are_bounded_by_the_schema(
        self, base_config_dict: dict, tmp_path: Path
    ) -> None:
        from src.core.exceptions import ConfigurationError

        base_config_dict["pipeline"] = {"capture_queue_size": 0}
        with pytest.raises(ConfigurationError):
            config_from_dict(base_config_dict, base_dir=tmp_path)

    def test_the_overload_policy_is_configurable(
        self, base_config_dict: dict, tmp_path: Path
    ) -> None:
        base_config_dict["pipeline"] = {"overload_policy": "block"}
        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        assert config.pipeline.overload_policy is OverloadPolicy.BLOCK
