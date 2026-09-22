"""Benchmark reporting, target verdicts and the scenario matrix."""

from __future__ import annotations

import json

import pytest

from src.config.loader import config_from_dict
from src.config.schema import BenchmarkTargetConfig
from src.tools.benchmark import (
    BenchmarkReport,
    StageBenchmark,
    Verdict,
    evaluate_targets,
)
from src.tools.benchmark_matrix import (
    GROUPS,
    MatrixReport,
    MatrixResult,
    build_scenarios,
)
from src.tools.telemetry import TelemetryCollector


def make_report(**overrides) -> BenchmarkReport:
    defaults: dict = {
        "source_id": "video:test.mp4",
        "frames": 100,
        "detections": 200,
        "recognized": 180,
        "warmup_frames": 10,
        "elapsed_s": 4.0,
        "end_to_end_fps": 25.0,
        "detector_fps": 40.0,
        "reid_fps": 20.0,
        "latency_p50_ms": 35.0,
        "latency_p95_ms": 70.0,
        "latency_p99_ms": 90.0,
        "drop_rate_percent": 1.0,
    }
    defaults.update(overrides)
    return BenchmarkReport(**defaults)


class TestTargetEvaluation:
    def test_no_targets_means_no_verdict(self) -> None:
        """A laptop must not be judged against a Jetson's numbers."""
        verdict, checks = evaluate_targets(make_report(), BenchmarkTargetConfig())
        assert verdict is Verdict.NOT_CHECKED
        assert checks == []

    def test_comfortably_meeting_every_target_passes(self) -> None:
        targets = BenchmarkTargetConfig(
            enabled=True, minimum_fps=20.0, maximum_p95_latency_ms=100.0,
            maximum_drop_rate_percent=5.0, warn_margin_percent=10.0,
        )
        verdict, checks = evaluate_targets(make_report(), targets)
        assert verdict is Verdict.PASS
        assert len(checks) == 3
        assert all(c.verdict is Verdict.PASS for c in checks)

    def test_scraping_a_target_warns(self) -> None:
        targets = BenchmarkTargetConfig(
            enabled=True, minimum_fps=24.0, warn_margin_percent=10.0
        )
        verdict, _ = evaluate_targets(make_report(end_to_end_fps=25.0), targets)
        assert verdict is Verdict.WARN

    def test_missing_a_target_fails(self) -> None:
        targets = BenchmarkTargetConfig(enabled=True, minimum_fps=30.0)
        verdict, checks = evaluate_targets(make_report(end_to_end_fps=25.0), targets)
        assert verdict is Verdict.FAIL
        assert checks[0].verdict is Verdict.FAIL

    def test_latency_is_an_upper_bound(self) -> None:
        targets = BenchmarkTargetConfig(enabled=True, maximum_p95_latency_ms=50.0)
        verdict, _ = evaluate_targets(make_report(latency_p95_ms=70.0), targets)
        assert verdict is Verdict.FAIL

    def test_the_drop_rate_is_checked(self) -> None:
        targets = BenchmarkTargetConfig(enabled=True, maximum_drop_rate_percent=1.0)
        verdict, _ = evaluate_targets(make_report(drop_rate_percent=8.0), targets)
        assert verdict is Verdict.FAIL

    def test_the_worst_check_decides_the_overall_verdict(self) -> None:
        targets = BenchmarkTargetConfig(
            enabled=True, minimum_fps=10.0, maximum_p95_latency_ms=10.0
        )
        verdict, checks = evaluate_targets(make_report(), targets)
        assert verdict is Verdict.FAIL
        assert any(c.verdict is Verdict.PASS for c in checks)

    def test_disabled_targets_are_not_checked(self) -> None:
        targets = BenchmarkTargetConfig(enabled=False, minimum_fps=1000.0)
        verdict, _ = evaluate_targets(make_report(), targets)
        assert verdict is Verdict.NOT_CHECKED


class TestReportSerialisation:
    def test_the_report_is_machine_readable(self) -> None:
        report = make_report(
            stages=[StageBenchmark("detector", 100, 25.0, 24.0, 40.0, 45.0)],
            face_search={"upper_body": 50},
            scheduler={"recognition_attempts": 50, "recognition_skips": 150},
        )
        data = report.to_dict()
        payload = json.loads(json.dumps(data))   # must be JSON-serialisable

        assert payload["throughput"]["end_to_end_fps"] == 25.0
        assert payload["latency_ms"]["p95"] == 70.0
        assert payload["pipeline"]["drop_rate_percent"] == 1.0
        assert payload["stages"][0]["stage"] == "detector"
        assert payload["targets"]["verdict"] == "NOT_CHECKED"

    def test_the_human_report_separates_model_from_end_to_end(self) -> None:
        """Conflating the two is the easiest way to misread a profile."""
        report = make_report(
            stages=[StageBenchmark("encoder", 20, 60.0, 58.0, 80.0, 90.0)]
        )
        text = report.render()
        assert "end-to-end FPS" in text
        assert "model stage only" in text
        assert "per call, not per frame" in text

    def test_the_verdict_is_rendered_when_targets_exist(self) -> None:
        targets = BenchmarkTargetConfig(enabled=True, minimum_fps=30.0)
        report = make_report(end_to_end_fps=20.0)
        report.verdict, report.target_checks = evaluate_targets(report, targets)
        text = report.render()
        assert "Targets: FAIL" in text
        assert "minimum_fps" in text

    def test_telemetry_appears_when_collected(self) -> None:
        report = make_report(
            telemetry={"available": True, "cpu_percent_avg": 55.0, "ram_used_mb_peak": 900.0}
        )
        text = report.render()
        assert "System" in text
        assert "CPU avg %" in text


class TestScenarioMatrix:
    def test_every_group_produces_scenarios(self, base_config_dict, tmp_path) -> None:
        from src.config.paths import ProjectPaths

        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        paths = ProjectPaths.from_config(config)
        for group in GROUPS:
            scenarios = build_scenarios(config, paths, [group])
            assert scenarios, group
            assert all(s.group for s in scenarios)

    def test_the_batch_group_covers_the_documented_sizes(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.config.paths import ProjectPaths

        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        scenarios = build_scenarios(config, ProjectPaths.from_config(config), ["batch"])
        names = {s.name for s in scenarios}
        assert names == {"batch-1", "batch-2", "batch-4", "batch-8"}

    def test_scenarios_actually_change_the_configuration(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.config.paths import ProjectPaths

        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        scenarios = build_scenarios(config, ProjectPaths.from_config(config), ["batch"])
        scenario = next(s for s in scenarios if s.name == "batch-4")
        scenario.apply(config)
        assert config.recognition.batch.max_size == 4

    def test_the_recognition_group_spans_every_strategy(
        self, base_config_dict, tmp_path
    ) -> None:
        from src.config.paths import ProjectPaths

        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        scenarios = build_scenarios(
            config, ProjectPaths.from_config(config), ["recognition"]
        )
        names = {s.name for s in scenarios}
        assert "recog-every-frame" in names
        assert "recog-adaptive" in names

    def test_an_unknown_group_is_actionable(self, base_config_dict, tmp_path) -> None:
        from src.config.paths import ProjectPaths

        config = config_from_dict(base_config_dict, base_dir=tmp_path)
        with pytest.raises(ValueError, match="unknown scenario group"):
            build_scenarios(config, ProjectPaths.from_config(config), ["nonsense"])

    def test_the_matrix_report_serialises_and_renders(self) -> None:
        report = MatrixReport(
            input_path="clip.mp4", frames=100, warmup=10,
            environment={"platform": "Linux aarch64", "device": "cuda"},
            results=[
                MatrixResult("batch-1", "batch", "size 1", make_report()),
                MatrixResult("batch-2", "batch", "size 2", skipped="requirements not met"),
            ],
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["frames_per_scenario"] == 100
        assert len(payload["results"]) == 2

        text = report.render()
        assert "batch-1" in text
        assert "skipped" in text
        assert "end-to-end" in text


class TestTelemetry:
    def test_collection_never_raises(self) -> None:
        collector = TelemetryCollector()
        for _ in range(3):
            collector.sample()
        summary = collector.summary()
        assert summary["available"] is True
        assert summary["samples"] == 3

    def test_it_can_be_disabled(self) -> None:
        collector = TelemetryCollector(enabled=False)
        assert collector.sample() is None
        assert collector.summary() == {"available": False}

    def test_missing_counters_report_none_rather_than_failing(self) -> None:
        """A benchmark must not refuse to run because a sysfs node is absent."""
        collector = TelemetryCollector()
        collector.sample()
        summary = collector.summary()
        # GPU counters are absent on a machine without NVIDIA hardware.
        assert "gpu_percent_avg" in summary
