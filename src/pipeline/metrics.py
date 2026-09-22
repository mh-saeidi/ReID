"""Runtime metrics: FPS, stage latencies and detection counters."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class StageTimer:
    """Rolling latency statistics for one pipeline stage."""

    name: str
    window: int = 300
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=300))
    total: float = 0.0
    count: int = 0

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)
        self.total += seconds
        self.count += 1

    @property
    def average_ms(self) -> float:
        return (self.total / self.count * 1000.0) if self.count else 0.0

    @property
    def recent_ms(self) -> float:
        return float(np.mean(self.samples) * 1000.0) if self.samples else 0.0

    def percentile_ms(self, percentile: float) -> float:
        if not self.samples:
            return 0.0
        return float(np.percentile(np.asarray(self.samples) * 1000.0, percentile))

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "average_ms": round(self.average_ms, 3),
            "recent_ms": round(self.recent_ms, 3),
            "p50_ms": round(self.percentile_ms(50), 3),
            "p95_ms": round(self.percentile_ms(95), 3),
            "p99_ms": round(self.percentile_ms(99), 3),
        }


class MetricsCollector:
    """Aggregates throughput and per-stage latency for a run."""

    def __init__(self, *, window: int = 120) -> None:
        self._window = window
        self._frame_times: deque[float] = deque(maxlen=window)
        self._stages: dict[str, StageTimer] = {}
        self._started = time.perf_counter()
        self._last_frame_at: float | None = None

        self.frames = 0
        self.detections = 0
        self.recognized = 0
        self.unknown = 0
        self.no_face = 0
        self.active_tracks = 0
        self.reid_calls = 0
        self.reid_crops = 0
        self.face_detect_calls = 0
        self.faces_found = 0
        self.dropped_frames = 0
        self.queue_depth = 0
        self.batch_sizes: deque[int] = deque(maxlen=window)
        self.batches = 0
        self._people_per_frame: deque[int] = deque(maxlen=window)
        self._faces_per_frame: deque[int] = deque(maxlen=window)
        self._embeddings = 0
        self.scheduler_stats: dict[str, object] = {}
        self.search_stats: dict[str, int] = {}
        self.system: dict[str, object] = {}

    def stage(self, name: str) -> StageTimer:
        timer = self._stages.get(name)
        if timer is None:
            timer = StageTimer(name, window=self._window)
            timer.samples = deque(maxlen=self._window)
            self._stages[name] = timer
        return timer

    def record_stage(self, name: str, seconds: float) -> None:
        self.stage(name).add(seconds)

    def record_frame(
        self,
        *,
        detections: int,
        recognized: int,
        tracks: int,
        no_face: int = 0,
        faces: int = 0,
    ) -> None:
        now = time.perf_counter()
        if self._last_frame_at is not None:
            self._frame_times.append(now - self._last_frame_at)
        self._last_frame_at = now
        self.frames += 1
        self.detections += detections
        self.recognized += recognized
        self.no_face += no_face
        self.unknown += detections - recognized - no_face
        self.active_tracks = tracks
        self._people_per_frame.append(detections)
        self._faces_per_frame.append(faces)
        self.faces_found += faces

    def record_batch(self, size: int) -> None:
        """One encoder invocation of ``size`` chips."""
        if size <= 0:
            return
        self.batch_sizes.append(size)
        self.batches += 1
        self._embeddings += size

    def record_drop(self, count: int = 1) -> None:
        self.dropped_frames += count

    @property
    def average_batch_size(self) -> float:
        return float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0

    @property
    def average_people_per_frame(self) -> float:
        return float(np.mean(self._people_per_frame)) if self._people_per_frame else 0.0

    @property
    def average_faces_per_frame(self) -> float:
        return float(np.mean(self._faces_per_frame)) if self._faces_per_frame else 0.0

    @property
    def embeddings_per_second(self) -> float:
        return self._embeddings / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def drop_rate_percent(self) -> float:
        total = self.frames + self.dropped_frames
        return 100.0 * self.dropped_frames / total if total else 0.0

    def latency_percentile_ms(self, percentile: float, stage: str = "end_to_end") -> float:
        """Percentile of the chosen stage, defaulting to end-to-end latency."""
        timer = self._stages.get(stage) or self._stages.get("total")
        return timer.percentile_ms(percentile) if timer else 0.0

    @property
    def fps(self) -> float:
        """Instantaneous FPS over the recent window."""
        if not self._frame_times:
            return 0.0
        mean = float(np.mean(self._frame_times))
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started

    @property
    def average_fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else 0.0

    def stage_ms(self, name: str) -> float:
        timer = self._stages.get(name)
        return timer.recent_ms if timer else 0.0

    def snapshot(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "elapsed_s": round(self.elapsed, 3),
            "fps_recent": round(self.fps, 2),
            "fps_average": round(self.average_fps, 2),
            "latency_p50_ms": round(self.latency_percentile_ms(50), 3),
            "latency_p95_ms": round(self.latency_percentile_ms(95), 3),
            "latency_p99_ms": round(self.latency_percentile_ms(99), 3),
            "detections": self.detections,
            "recognized": self.recognized,
            "unknown": self.unknown,
            "no_face": self.no_face,
            "active_tracks": self.active_tracks,
            "average_people_per_frame": round(self.average_people_per_frame, 3),
            "average_faces_per_frame": round(self.average_faces_per_frame, 3),
            "embeddings_per_second": round(self.embeddings_per_second, 2),
            "reid_calls": self.reid_calls,
            "reid_crops": self.reid_crops,
            "face_detect_calls": self.face_detect_calls,
            "batches": self.batches,
            "average_batch_size": round(self.average_batch_size, 2),
            "dropped_frames": self.dropped_frames,
            "drop_rate_percent": round(self.drop_rate_percent, 3),
            "queue_depth": self.queue_depth,
            "scheduler": dict(self.scheduler_stats),
            "face_search": dict(self.search_stats),
            "system": dict(self.system),
            "stages": {name: timer.to_dict() for name, timer in sorted(self._stages.items())},
        }

    @property
    def stages(self) -> dict[str, StageTimer]:
        """Read-only view of the per-stage timers."""
        return dict(self._stages)


class Stopwatch:
    """Context manager that records elapsed time into a collector."""

    __slots__ = ("_collector", "_name", "_start", "elapsed")

    def __init__(self, collector: MetricsCollector | None, name: str) -> None:
        self._collector = collector
        self._name = name
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self._collector is not None:
            self._collector.record_stage(self._name, self.elapsed)
