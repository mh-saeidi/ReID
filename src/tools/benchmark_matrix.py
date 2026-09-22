"""Benchmark matrix: sweep configurations and compare them on one input.

Each scenario is a named configuration override applied to a copy of the loaded
configuration, benchmarked against the same source. That keeps the comparison
honest -- nothing varies between runs except the setting under test.

The scenarios are grouped so a deployment question maps to one command:

``recognition``  how much does the scheduler actually save, and at what cost to
                 the number of recognitions?
``batch``        what batch size suits this hardware? 8 is a guess until it is
                 measured; an 8 GB Orin Nano is not a datacentre GPU.
``encoder``      ArcFace or SFace for this accuracy/latency budget?
``search``       is whole-frame face detection faster than per-person ROI here?
``output``       what do display and recording actually cost?
``resolution``   what does the detector input size buy?
"""

from __future__ import annotations

import copy
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.schema import AppConfig
from src.tools.benchmark import BenchmarkReport, Verdict, run_benchmark
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One configuration under test."""

    name: str
    group: str
    description: str
    apply: Callable[[AppConfig], None]
    requires: Callable[[AppConfig, Any], bool] | None = None
    """Optional predicate; the scenario is skipped when it returns False."""


@dataclass(slots=True)
class MatrixResult:
    scenario: str
    group: str
    description: str
    report: BenchmarkReport | None = None
    skipped: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "group": self.group,
            "description": self.description,
            "skipped": self.skipped,
            "report": self.report.to_dict() if self.report else None,
        }


@dataclass(slots=True)
class MatrixReport:
    """Every scenario's result plus the environment they were measured in."""

    input_path: str
    frames: int
    warmup: int
    results: list[MatrixResult] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input_path,
            "frames_per_scenario": self.frames,
            "warmup_frames": self.warmup,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 1),
            "environment": self.environment,
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        lines = [
            "Benchmark matrix",
            f"  input   : {self.input_path}",
            f"  frames  : {self.frames} per scenario (after {self.warmup} warm-up)",
            f"  machine : {self.environment.get('platform', '?')}"
            f"  |  device: {self.environment.get('device', '?')}",
            "",
            f"  {'SCENARIO':<26} {'FPS':>7} {'p50':>8} {'p95':>8} {'RECOG':>7} "
            f"{'BATCH':>6}  VERDICT",
        ]
        current_group = None
        for result in self.results:
            if result.group != current_group:
                current_group = result.group
                lines.append(f"  -- {current_group} " + "-" * (58 - len(current_group)))
            if result.skipped:
                lines.append(f"  {result.scenario:<26} {'skipped':>7}  {result.skipped}")
                continue
            report = result.report
            assert report is not None  # noqa: S101
            verdict = report.verdict.value if report.verdict is not Verdict.NOT_CHECKED else ""
            lines.append(
                f"  {result.scenario:<26} {report.end_to_end_fps:>7.1f} "
                f"{report.latency_p50_ms:>7.1f}m {report.latency_p95_ms:>7.1f}m "
                f"{report.recognized:>7} {report.average_batch_size:>6.2f}  {verdict}"
            )
        lines += [
            "",
            "  FPS is end-to-end (capture -> inference -> output), not model-only.",
            "  RECOG counts recognised detections, so a faster scenario that",
            "  recognises far fewer people is not automatically better.",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Scenario definitions
# --------------------------------------------------------------------------- #


def _set_scheduler(config: AppConfig, **values: Any) -> None:
    for key, value in values.items():
        setattr(config.recognition_scheduler, key, value)


def recognition_scenarios() -> list[Scenario]:
    def every_frame(config: AppConfig) -> None:
        config.recognition_scheduler.enabled = False
        config.performance.reid_interval = 1

    def interval_three(config: AppConfig) -> None:
        config.recognition_scheduler.enabled = False
        config.performance.reid_interval = 3

    def adaptive(config: AppConfig) -> None:
        config.recognition_scheduler.enabled = True

    def adaptive_aggressive(config: AppConfig) -> None:
        config.recognition_scheduler.enabled = True
        _set_scheduler(config, stable_interval=12, unknown_interval=6)

    return [
        Scenario("recog-every-frame", "recognition",
                 "no scheduler, embed on every frame", every_frame),
        Scenario("recog-interval-3", "recognition",
                 "legacy fixed interval of 3", interval_three),
        Scenario("recog-adaptive", "recognition",
                 "adaptive scheduler (default)", adaptive),
        Scenario("recog-adaptive-slow", "recognition",
                 "adaptive scheduler, longer intervals", adaptive_aggressive),
    ]


def batch_scenarios() -> list[Scenario]:
    scenarios = []
    for size in (1, 2, 4, 8):
        def apply(config: AppConfig, size: int = size) -> None:
            config.recognition.batch.enabled = size > 1
            config.recognition.batch.max_size = size

        scenarios.append(
            Scenario(
                f"batch-{size}", "batch",
                f"face encoder batch size {size}", apply,
            )
        )
    return scenarios


def encoder_scenarios(config: AppConfig, paths) -> list[Scenario]:  # noqa: ARG001 - uniform factory signature
    sface = paths.resolve("models/face_recognition_sface_2021dec.onnx")
    arcface = paths.resolve("models/w600k_r50.onnx")

    def use_sface(target: AppConfig) -> None:
        target.face.recognition_model = str(sface)

    def use_arcface(target: AppConfig) -> None:
        target.face.recognition_model = str(arcface)

    return [
        Scenario("encoder-arcface", "encoder", "ArcFace w600k_r50 (512-D)",
                 use_arcface, requires=lambda c, p: arcface.exists()),
        Scenario("encoder-sface", "encoder", "SFace (128-D, smaller)",
                 use_sface, requires=lambda c, p: sface.exists()),
    ]


def search_scenarios() -> list[Scenario]:
    def upper_body(config: AppConfig) -> None:
        from src.config.schema import FaceSearchRegion

        config.face.adaptive_search.enabled = False
        config.face.search_region = FaceSearchRegion.UPPER_BODY

    def whole_frame(config: AppConfig) -> None:
        from src.config.schema import FaceSearchRegion

        config.face.adaptive_search.enabled = False
        config.face.search_region = FaceSearchRegion.FRAME

    def adaptive(config: AppConfig) -> None:
        config.face.adaptive_search.enabled = True

    return [
        Scenario("search-upper-body", "face search", "per-person ROI (default)", upper_body),
        Scenario("search-frame", "face search", "one whole-frame pass", whole_frame),
        Scenario("search-adaptive", "face search", "ROI when sparse, frame when crowded",
                 adaptive),
    ]


def output_scenarios() -> list[Scenario]:
    def minimal(config: AppConfig) -> None:
        config.output.save_video = False
        config.output.save_snapshots = False
        config.output.save_metadata = False
        config.recording.mode = config.recording.mode.DISABLED
        config.display.show_window = False

    def with_metadata(config: AppConfig) -> None:
        minimal(config)
        config.output.save_metadata = True

    def with_video(config: AppConfig) -> None:
        minimal(config)
        config.output.save_video = True

    def with_recording(config: AppConfig) -> None:
        from src.config.schema import RecordingMode

        minimal(config)
        config.recording.mode = RecordingMode.CONTINUOUS

    return [
        Scenario("output-none", "output", "no output at all", minimal),
        Scenario("output-metadata", "output", "JSONL metadata only", with_metadata),
        Scenario("output-video", "output", "annotated video (software encoder)", with_video),
        Scenario("output-recording", "output", "continuous recording", with_recording),
    ]


def resolution_scenarios() -> list[Scenario]:
    scenarios = []
    for size in (480, 640, 960):
        def apply(config: AppConfig, size: int = size) -> None:
            config.detector.imgsz = size

        scenarios.append(
            Scenario(f"imgsz-{size}", "detector resolution",
                     f"detector input {size}px", apply)
        )
    return scenarios


def pipeline_scenarios() -> list[Scenario]:
    def sync(config: AppConfig) -> None:
        config.pipeline.async_enabled = False

    def asynchronous(config: AppConfig) -> None:
        config.pipeline.async_enabled = True

    return [
        Scenario("pipeline-sync", "pipeline", "synchronous loop", sync),
        Scenario("pipeline-async", "pipeline", "async capture/inference/output",
                 asynchronous),
    ]


GROUPS: dict[str, Callable[..., list[Scenario]]] = {
    "recognition": lambda config, paths: recognition_scenarios(),
    "batch": lambda config, paths: batch_scenarios(),
    "encoder": encoder_scenarios,
    "search": lambda config, paths: search_scenarios(),
    "output": lambda config, paths: output_scenarios(),
    "resolution": lambda config, paths: resolution_scenarios(),
    "pipeline": lambda config, paths: pipeline_scenarios(),
}


def build_scenarios(config: AppConfig, paths, groups: list[str] | None = None) -> list[Scenario]:
    wanted = groups or list(GROUPS)
    scenarios: list[Scenario] = []
    for name in wanted:
        factory = GROUPS.get(name)
        if factory is None:
            raise ValueError(
                f"unknown scenario group '{name}'. Available: {', '.join(GROUPS)}"
            )
        scenarios.extend(factory(config, paths))
    return scenarios


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_matrix(
    base_config: AppConfig,
    scenarios: list[Scenario],
    *,
    source_factory: Callable[[AppConfig], Any],
    frames: int = 120,
    warmup: int = 10,
    input_label: str = "",
    progress: Callable[[str, int, int], None] | None = None,
) -> MatrixReport:
    """Run every scenario against the same input and collect the results.

    A fresh engine is built per scenario because several settings (encoder,
    detector resolution, backend) are decided at load time. That makes the
    matrix slow but comparable, which is the point.
    """
    from src.pipeline.engine import build_engine  # noqa: PLC0415

    started = time.perf_counter()
    report = MatrixReport(
        input_path=input_label,
        frames=frames,
        warmup=warmup,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        environment=_environment(base_config),
    )

    for index, scenario in enumerate(scenarios, start=1):
        if progress:
            progress(scenario.name, index, len(scenarios))

        config = copy.deepcopy(base_config)
        # Benchmarks must not be skewed by artefacts, and must not overwrite
        # a deployment's real output.
        config.display.show_window = False
        config.events.enabled = False
        config.debug.enabled = False
        scenario.apply(config)

        if scenario.requires is not None and not scenario.requires(config, None):
            report.results.append(
                MatrixResult(scenario.name, scenario.group, scenario.description,
                             skipped="requirements not met")
            )
            continue

        engine = None
        try:
            engine = build_engine(config)
            engine.ensure_gallery()
            pipeline = engine.pipeline(use_tracking=config.tracking.enabled)
            source = source_factory(config)
            outcome = run_benchmark(
                engine, pipeline, source,
                max_frames=frames, warmup=warmup, label=scenario.name,
                collect_telemetry=config.benchmark.collect_system_telemetry,
            )
            report.results.append(
                MatrixResult(scenario.name, scenario.group, scenario.description, outcome)
            )
        except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the sweep
            logger.error(
                "Scenario failed", extra={"scenario": scenario.name, "error": str(exc)}
            )
            report.results.append(
                MatrixResult(scenario.name, scenario.group, scenario.description,
                             skipped=f"failed: {exc}")
            )
        finally:
            if engine is not None:
                engine.close()

    report.duration_s = time.perf_counter() - started
    return report


def _environment(config: AppConfig) -> dict[str, Any]:
    from src.hardware.capabilities import detect_capabilities  # noqa: PLC0415

    caps = detect_capabilities()
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": caps.host.cpu_model,
        "cores": caps.host.cpu_count,
        "memory_gb": caps.host.total_memory_gb,
        "device": config.device.device.value,
        "jetson": caps.jetson.model if caps.is_jetson else None,
        "jetpack": caps.jetson.jetpack_version,
        "cuda": caps.cuda.runtime_version,
        "tensorrt": caps.tensorrt.version,
        "torch": caps.torch_version,
        "opencv": caps.opencv_version,
    }


def write_report(report: MatrixReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
