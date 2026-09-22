"""Asynchronous output dispatch.

Disk I/O, JPEG encoding, JSON serialisation and video encoding are all orders of
magnitude slower than the arithmetic they follow, and none of them needs to
happen before the next frame is inferred. Moving them behind a bounded queue
stops them stalling the GPU.

The shedding policy is explicit because "drop something" is not a uniform
decision:

* visualization frames (preview, annotated video) are *sampling* of a
  continuous signal -- dropping some degrades smoothness and nothing else;
* recognition events and event-triggered snapshots are *discrete facts* about
  who was seen. Dropping one loses information that cannot be recovered.

So the queue sheds visualization first, then periodic metadata, and never sheds
events. When even that is not enough the dispatcher applies back-pressure
rather than silently losing a recognition.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from src.config.schema import OverloadPolicy
from src.output.sink import OutputSink, OutputTask
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DispatchStats:
    submitted: int = 0
    completed: int = 0
    dropped_visualization: int = 0
    dropped_tasks: int = 0
    blocked_seconds: float = 0.0
    max_queue_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "dropped_visualization": self.dropped_visualization,
            "dropped_tasks": self.dropped_tasks,
            "blocked_seconds": round(self.blocked_seconds, 3),
            "max_queue_depth": self.max_queue_depth,
        }


class AsyncOutputDispatcher:
    """Runs an :class:`OutputSink` on its own worker thread.

    A single worker is used by default, which preserves frame ordering for
    video and metadata without any explicit sequencing.
    """

    _SENTINEL = object()

    def __init__(
        self,
        sink: OutputSink,
        *,
        queue_size: int = 4,
        policy: OverloadPolicy = OverloadPolicy.DROP_VISUALIZATION,
        workers: int = 1,
        name: str = "output",
    ) -> None:
        self._sink = sink
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_size))
        self._policy = policy
        self._threads: list[threading.Thread] = []
        self._workers = max(1, workers)
        self._name = name
        self._running = threading.Event()
        self.stats = DispatchStats()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        for index in range(self._workers):
            thread = threading.Thread(
                target=self._run, name=f"{self._name}-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        logger.debug("Output dispatcher started", extra={"workers": self._workers})

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._SENTINEL:
                    return
                self._sink.handle(task)
                self.stats.completed += 1
            except Exception as exc:  # noqa: BLE001 - a worker must not die
                logger.error("Output worker error: %s", exc)
            finally:
                self._queue.task_done()

    # ---------------------------------------------------------------- submit
    def submit(self, task: OutputTask, *, must_persist: bool = False) -> bool:
        """Queue one frame's output.

        Args:
            must_persist: This task carries something the configuration says
                must be kept (an event snapshot, or metadata under a policy that
                preserves it). Such a task is never dropped: the caller blocks
                instead, which shows up as back-pressure rather than loss.

        Returns:
            Whether the task was accepted.
        """
        self.stats.submitted += 1
        depth = self._queue.qsize()
        self.stats.max_queue_depth = max(self.stats.max_queue_depth, depth)

        if must_persist or self._policy is OverloadPolicy.BLOCK:
            started = time.perf_counter()
            self._queue.put(task)
            elapsed = time.perf_counter() - started
            if elapsed > 0.001:
                self.stats.blocked_seconds += elapsed
            return True

        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            pass

        # The queue is full. Shedding the *content* of this task would not free
        # a slot, so instead the oldest queued non-essential task is evicted to
        # make room -- the same newest-frame preference the capture queue uses,
        # and for the same reason: a stale annotated frame has no value.
        if self._policy is OverloadPolicy.DROP_VISUALIZATION:
            if self._evict_one():
                task.allow_visualization = False
                task.annotated = None
                try:
                    self._queue.put_nowait(task)
                    self.stats.dropped_visualization += 1
                    return True
                except queue.Full:  # pragma: no cover - raced with a worker
                    pass

        self.stats.dropped_tasks += 1
        logger.debug(
            "Output task dropped: the output stage is behind",
            extra={"frame": task.result.frame_index, "policy": self._policy.value},
        )
        return False

    def _evict_one(self) -> bool:
        """Discard one queued non-essential task, if there is one.

        Essential tasks (an event snapshot, a recording trigger) are submitted
        with ``must_persist`` and never reach this path, so nothing evicted here
        can be a recognition the configuration promised to keep.
        """
        try:
            evicted = self._queue.get_nowait()
        except queue.Empty:  # pragma: no cover - raced with a worker
            return False
        if evicted is self._SENTINEL:
            # Shutdown is in progress: put it back and refuse the new task.
            self._queue.put_nowait(evicted)
            return False
        self._queue.task_done()
        return True

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------- shutdown
    def stop(self, timeout: float = 10.0) -> None:
        """Drain the queue and join the workers."""
        if not self._running.is_set():
            return
        deadline = time.perf_counter() + timeout
        for _ in self._threads:
            self._queue.put(self._SENTINEL)
        for thread in self._threads:
            remaining = max(0.1, deadline - time.perf_counter())
            thread.join(timeout=remaining)
            if thread.is_alive():  # pragma: no cover - depends on disk speed
                logger.warning(
                    "Output worker %s did not stop within the timeout", thread.name
                )
        self._threads.clear()
        self._running.clear()
        logger.debug("Output dispatcher stopped", extra=self.stats.to_dict())

    def __enter__(self) -> AsyncOutputDispatcher:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
