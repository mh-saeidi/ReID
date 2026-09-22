"""Benchmark mode.

Reports only measured values: stage latencies come from the same
:class:`MetricsCollector` the live pipeline uses, so the numbers correspond to
the real code path rather than a synthetic micro-benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.pipeline.engine import Engine
from src.pipeline.metrics import Stopwatch
from src.pipeline.processor import ReIDPipeline
from src.sources.base import BaseSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class StageBenchmark:
    name: str
    calls: int
    average_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "calls": self.calls,
            "average_ms": round(self.average_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


class Verdict(str, Enum):
    """Outcome of comparing a run against its configured targets."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NOT_CHECKED = "NOT_CHECKED"


@dataclass(slots=True)
class TargetCheck:
    """One target and what was measured against it."""

    name: str
    limit: float
    measured: float
    verdict: Verdict
    comparison: str
    """``min`` (measured must be at least the limit) or ``max``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.name,
            "limit": round(self.limit, 3),
            "measured": round(self.measured, 3),
            "comparison": self.comparison,
            "verdict": self.verdict.value,
        }


def evaluate_targets(report: BenchmarkReport, targets) -> tuple[Verdict, list[TargetCheck]]:
    """Compare a run against its profile's targets.

    Targets are per-profile on purpose: a number that makes sense for an Orin
    Nano is meaningless on a workstation, so an unconfigured target reports
    NOT_CHECKED rather than inventing a universal requirement.
    """
    if targets is None or not targets.enabled:
        return Verdict.NOT_CHECKED, []

    margin = targets.warn_margin_percent / 100.0
    checks: list[TargetCheck] = []

    def add(name: str, limit: float | None, measured: float, comparison: str) -> None:
        if limit is None:
            return
        if comparison == "min":
            if measured >= limit:
                verdict = Verdict.PASS if measured >= limit * (1 + margin) else Verdict.WARN
            else:
                verdict = Verdict.FAIL
        else:
            if measured <= limit:
                verdict = Verdict.PASS if measured <= limit * (1 - margin) else Verdict.WARN
            else:
                verdict = Verdict.FAIL
        checks.append(TargetCheck(name, limit, measured, verdict, comparison))

    add("minimum_fps", targets.minimum_fps, report.end_to_end_fps, "min")
    add("maximum_p95_latency_ms", targets.maximum_p95_latency_ms, report.latency_p95_ms, "max")
    add("maximum_p99_latency_ms", targets.maximum_p99_latency_ms, report.latency_p99_ms, "max")
    add(
        "maximum_drop_rate_percent",
        targets.maximum_drop_rate_percent,
        report.drop_rate_percent,
        "max",
    )

    if not checks:
        return Verdict.NOT_CHECKED, []
    if any(c.verdict is Verdict.FAIL for c in checks):
        return Verdict.FAIL, checks
    if any(c.verdict is Verdict.WARN for c in checks):
        return Verdict.WARN, checks
    return Verdict.PASS, checks


@dataclass(slots=True)
class BenchmarkReport:
    """Measured throughput and latency for one input."""

    source_id: str
    frames: int
    detections: int
    recognized: int
    warmup_frames: int
    elapsed_s: float
    end_to_end_fps: float
    detector_fps: float
    reid_fps: float
    stages: list[StageBenchmark] = field(default_factory=list)
    device: str = ""
    detector_model: str = ""
    reid_model: str = ""
    embedding_dimension: int = 0
    gallery_size: int = 0

    # --- extended metrics ---------------------------------------------------
    label: str = ""
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    average_people_per_frame: float = 0.0
    average_faces_per_frame: float = 0.0
    embeddings_per_second: float = 0.0
    average_batch_size: float = 0.0
    dropped_frames: int = 0
    drop_rate_percent: float = 0.0
    queue_depth: int = 0
    recognition_attempts: int = 0
    recognition_skips: int = 0
    face_search: dict[str, int] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    backends: dict[str, Any] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)
    verdict: Verdict = Verdict.NOT_CHECKED
    target_checks: list[TargetCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source": self.source_id,
            "device": self.device,
            "detector_model": self.detector_model,
            "reid_model": self.reid_model,
            "embedding_dimension": self.embedding_dimension,
            "gallery_size": self.gallery_size,
            "frames_measured": self.frames,
            "warmup_frames": self.warmup_frames,
            "detections": self.detections,
            "recognized": self.recognized,
            "elapsed_s": round(self.elapsed_s, 3),
            "throughput": {
                "end_to_end_fps": round(self.end_to_end_fps, 2),
                "detector_fps": round(self.detector_fps, 2),
                "reid_fps": round(self.reid_fps, 2),
                "embeddings_per_second": round(self.embeddings_per_second, 2),
            },
            "latency_ms": {
                "p50": round(self.latency_p50_ms, 3),
                "p95": round(self.latency_p95_ms, 3),
                "p99": round(self.latency_p99_ms, 3),
            },
            "workload": {
                "average_people_per_frame": round(self.average_people_per_frame, 3),
                "average_faces_per_frame": round(self.average_faces_per_frame, 3),
                "average_batch_size": round(self.average_batch_size, 2),
                "recognition_attempts": self.recognition_attempts,
                "recognition_skips": self.recognition_skips,
            },
            "pipeline": {
                "dropped_frames": self.dropped_frames,
                "drop_rate_percent": round(self.drop_rate_percent, 3),
                "queue_depth": self.queue_depth,
            },
            "face_search": self.face_search,
            "scheduler": self.scheduler,
            "backends": self.backends,
            "config": self.config_summary,
            "telemetry": self.telemetry,
            "targets": {
                "verdict": self.verdict.value,
                "checks": [c.to_dict() for c in self.target_checks],
            },
            "stages": [s.to_dict() for s in self.stages],
        }

    def render(self) -> str:
        lines = [
            f"Benchmark: {self.label or self.source_id}",
            "  (all values measured on this machine; nothing is extrapolated)",
            "",
            f"  source              : {self.source_id}",
            f"  device              : {self.device}",
            f"  detector            : {self.detector_model}",
            f"  encoder             : {self.reid_model} (dim {self.embedding_dimension})",
            f"  gallery identities  : {self.gallery_size}",
            f"  frames measured     : {self.frames} (after {self.warmup_frames} warm-up)",
            f"  detections          : {self.detections} ({self.recognized} recognized)",
            f"  people / frame      : {self.average_people_per_frame:.2f}"
            f"   faces / frame: {self.average_faces_per_frame:.2f}",
            "",
            "  Stage timing (per call, not per frame -- a stage that runs on",
            "  some frames only has a higher per-call cost than per-frame cost)",
        ]
        for stage in self.stages:
            lines.append(
                f"    {stage.name:<16}: n={stage.calls:<6} avg={stage.average_ms:8.2f} ms "
                f"p50={stage.p50_ms:8.2f} p95={stage.p95_ms:8.2f}"
            )
        lines += [
            "",
            "  Throughput",
            f"    end-to-end FPS  : {self.end_to_end_fps:8.2f}",
            f"    detector FPS    : {self.detector_fps:8.2f}   (model stage only)",
            f"    encoder FPS     : {self.reid_fps:8.2f}   (model stage only)",
            f"    embeddings/s    : {self.embeddings_per_second:8.2f}",
            f"    avg batch size  : {self.average_batch_size:8.2f}",
            "",
            "  End-to-end latency",
            f"    p50 : {self.latency_p50_ms:8.2f} ms",
            f"    p95 : {self.latency_p95_ms:8.2f} ms",
            f"    p99 : {self.latency_p99_ms:8.2f} ms",
            "",
            "  Pipeline",
            f"    dropped frames  : {self.dropped_frames} "
            f"({self.drop_rate_percent:.2f}%)",
            f"    recognition     : {self.recognition_attempts} attempts, "
            f"{self.recognition_skips} skips",
        ]
        if self.face_search:
            used = ", ".join(f"{k}={v}" for k, v in self.face_search.items() if v)
            lines.append(f"    face search     : {used or 'none'}")

        telemetry = self.telemetry or {}
        if telemetry.get("available"):
            lines += ["", "  System"]
            for key, label in (
                ("cpu_percent_avg", "CPU avg %"),
                ("cpu_percent_peak", "CPU peak %"),
                ("gpu_percent_avg", "GPU avg %"),
                ("gpu_percent_peak", "GPU peak %"),
                ("ram_used_mb_peak", "RAM peak MB"),
            ):
                value = telemetry.get(key)
                if value is not None:
                    lines.append(f"    {label:<16}: {value}")
            for key, label in (("temperature_peak_c", "temp peak C"),
                               ("power_peak_mw", "power peak mW")):
                value = telemetry.get(key)
                if value:
                    top = sorted(value.items(), key=lambda kv: -kv[1])[:3]
                    lines.append(
                        f"    {label:<16}: "
                        + ", ".join(f"{name}={reading}" for name, reading in top)
                    )

        if self.target_checks:
            lines += ["", f"  Targets: {self.verdict.value}"]
            for check in self.target_checks:
                arrow = ">=" if check.comparison == "min" else "<="
                lines.append(
                    f"    [{check.verdict.value:<4}] {check.name:<28} "
                    f"measured {check.measured:.2f} {arrow} {check.limit:.2f}"
                )
        elif self.verdict is Verdict.NOT_CHECKED:
            lines += [
                "",
                "  Targets: not configured for this profile "
                "(set benchmark.targets.enabled)",
            ]
        return "\n".join(lines)


def run_benchmark(
    engine: Engine,
    pipeline: ReIDPipeline,
    source: BaseSource,
    *,
    max_frames: int = 200,
    warmup: int = 5,
    label: str = "",
    collect_telemetry: bool = True,
    telemetry_every: int = 10,
) -> BenchmarkReport:
    """Measure the pipeline on a real input.

    The first ``warmup`` frames are discarded: they include lazy kernel
    compilation, allocator warm-up and (on TensorRT) the first-run profile
    selection, which would otherwise dominate the average.

    Stage timings are per *call*, not per frame. A stage that only runs on some
    frames -- the encoder under the recognition scheduler, for instance -- has a
    higher per-call cost than its per-frame contribution, and conflating the two
    is the easiest way to misread a profile.
    """
    from src.tools.telemetry import TelemetryCollector  # noqa: PLC0415

    info = source.open()
    pipeline.reset()
    telemetry = TelemetryCollector(enabled=collect_telemetry)

    processed = 0
    detections = recognized = 0
    started = 0.0
    metrics = pipeline.metrics
    try:
        for index in range(max_frames + warmup):
            frame = source.read()
            if frame is None:
                break
            if index == warmup:
                # Discard warm-up measurements and start the clock.
                pipeline.metrics.__init__()  # noqa: PLC2801 - deliberate counter reset
                metrics = pipeline.metrics
                started = time.perf_counter()
            with Stopwatch(metrics if index >= warmup else None, "end_to_end"):
                outcome = pipeline.process(frame)
            if index >= warmup:
                processed += 1
                detections += len(outcome.result.detections)
                recognized += len(outcome.result.recognized)
                if telemetry.enabled and processed % max(1, telemetry_every) == 0:
                    telemetry.sample()
    finally:
        source.close()

    if processed == 0:
        raise ValueError(
            f"benchmark produced no measured frames from {info.source_id}; "
            f"the input has fewer than {warmup + 1} frames. Lower --warmup."
        )

    elapsed = time.perf_counter() - started
    snapshot = metrics.snapshot()
    stages = [
        StageBenchmark(
            name=name,
            calls=timer.count,
            average_ms=timer.average_ms,
            p50_ms=timer.percentile_ms(50),
            p95_ms=timer.percentile_ms(95),
            p99_ms=timer.percentile_ms(99),
        )
        for name, timer in sorted(metrics.stages.items())
    ]

    detector_ms = metrics.stage("detector").average_ms
    encoder_ms = metrics.stage("encoder").average_ms or metrics.stage("reid").average_ms
    scheduler = snapshot.get("scheduler", {}) or {}

    report = BenchmarkReport(
        source_id=info.source_id,
        frames=processed,
        detections=detections,
        recognized=recognized,
        warmup_frames=warmup,
        elapsed_s=elapsed,
        end_to_end_fps=processed / elapsed if elapsed > 0 else 0.0,
        detector_fps=1000.0 / detector_ms if detector_ms > 0 else 0.0,
        reid_fps=1000.0 / encoder_ms if encoder_ms > 0 else 0.0,
        stages=stages,
        device=engine.device.device,
        detector_model=engine.detector.info.name,
        reid_model=engine.encoder.info.name,
        embedding_dimension=engine.encoder.info.embedding_dimension,
        gallery_size=len(engine.gallery.active_identities),
        label=label or info.source_id,
        latency_p50_ms=float(snapshot.get("latency_p50_ms", 0.0)),
        latency_p95_ms=float(snapshot.get("latency_p95_ms", 0.0)),
        latency_p99_ms=float(snapshot.get("latency_p99_ms", 0.0)),
        average_people_per_frame=float(snapshot.get("average_people_per_frame", 0.0)),
        average_faces_per_frame=float(snapshot.get("average_faces_per_frame", 0.0)),
        embeddings_per_second=float(snapshot.get("embeddings_per_second", 0.0)),
        average_batch_size=float(snapshot.get("average_batch_size", 0.0)),
        dropped_frames=int(snapshot.get("dropped_frames", 0)),
        drop_rate_percent=float(snapshot.get("drop_rate_percent", 0.0)),
        queue_depth=int(snapshot.get("queue_depth", 0)),
        recognition_attempts=int(scheduler.get("recognition_attempts", 0)),
        recognition_skips=int(scheduler.get("recognition_skips", 0)),
        face_search=dict(snapshot.get("face_search", {}) or {}),
        scheduler=scheduler,
        telemetry=telemetry.summary(),
        backends=dict(engine.backends),
        config_summary=_config_summary(engine.config),
    )
    report.verdict, report.target_checks = evaluate_targets(
        report, engine.config.benchmark.targets
    )
    logger.info(
        "Benchmark complete",
        extra={
            "label": report.label,
            "frames": processed,
            "fps": round(report.end_to_end_fps, 2),
            "p95_ms": round(report.latency_p95_ms, 2),
            "verdict": report.verdict.value,
        },
    )
    return report


def _config_summary(config) -> dict[str, Any]:
    """The settings that actually move the numbers, for the JSON report."""
    return {
        "recognition_mode": config.recognition.mode.value,
        "detector_imgsz": config.detector.imgsz,
        "scheduler_enabled": config.recognition_scheduler.enabled,
        "stable_interval": config.recognition_scheduler.stable_interval,
        "unknown_interval": config.recognition_scheduler.unknown_interval,
        "reid_interval": config.performance.reid_interval,
        "batch_enabled": config.recognition.batch.enabled,
        "batch_max_size": config.recognition.batch.max_size,
        "async_pipeline": config.pipeline.async_enabled,
        "face_search_region": config.face.search_region.value,
        "adaptive_search": config.face.adaptive_search.enabled,
        "tracking_enabled": config.tracking.enabled,
        "recognition_threshold": config.matching.recognition_threshold,
    }
