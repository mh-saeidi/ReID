"""CLI surface: command wiring, argument handling and error presentation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.common import build_overrides
from src.cli.main import app

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path, base_config_dict: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(base_config_dict), encoding="utf-8")
    return path


class TestCommandSurface:
    def test_root_help_lists_every_documented_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in (
            "webcam", "video", "image", "images", "stream", "run",
            "gallery", "config", "benchmark", "evaluate", "events", "retention",
        ):
            assert command in result.output

    @pytest.mark.parametrize(
        "command",
        [
            ["gallery", "--help"],
            ["gallery", "build", "--help"],
            ["gallery", "rebuild", "--help"],
            ["gallery", "list", "--help"],
            ["config", "validate", "--help"],
            ["webcam", "--help"],
            ["video", "--help"],
            ["image", "--help"],
            ["images", "--help"],
            ["benchmark", "--help"],
            ["evaluate", "--help"],
        ],
    )
    def test_each_subcommand_has_help(self, command) -> None:
        assert runner.invoke(app, command).exit_code == 0

    def test_version_reports_dependencies(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "ultralytics" in result.output


class TestConfigCommands:
    def test_validate_accepts_a_good_file(self, config_file: Path) -> None:
        result = runner.invoke(app, ["config", "validate", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "Configuration OK" in result.output

    def test_validate_emits_json_on_request(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["config", "validate", "--config", str(config_file), "--json"]
        )
        assert result.exit_code == 0
        assert json.loads(result.output)["valid"] is True

    def test_validate_warns_about_a_missing_reference_image(
        self, tmp_path: Path, base_config_dict: dict
    ) -> None:
        base_config_dict["people"] = [
            {"id": "ghost", "name": "Ghost", "image_path": "does/not/exist.jpg"}
        ]
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(base_config_dict), encoding="utf-8")
        result = runner.invoke(app, ["config", "validate", "--config", str(path)])
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_validate_rejects_a_bad_file_without_a_traceback(
        self, tmp_path: Path, base_config_dict: dict
    ) -> None:
        base_config_dict["matching"] = {
            "recognition_threshold": 0.9,
            "high_confidence_threshold": 0.1,
        }
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(base_config_dict), encoding="utf-8")
        result = runner.invoke(app, ["config", "validate", "--config", str(path)])
        assert result.exit_code == 1
        assert "error:" in result.output
        assert "Traceback" not in result.output

    def test_missing_config_file_is_reported_cleanly(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["config", "validate", "--config", str(tmp_path / "nope.yaml")]
        )
        assert result.exit_code == 1
        assert "configuration file not found" in result.output

    def test_show_prints_the_effective_configuration(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["config", "show", "--config", str(config_file), "--section", "matching"]
        )
        assert result.exit_code == 0
        assert json.loads(result.output)["matching"]["recognition_threshold"] == 0.70

    def test_show_rejects_an_unknown_section(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["config", "show", "--config", str(config_file), "--section", "bogus"]
        )
        assert result.exit_code == 1
        assert "unknown section" in result.output


class TestGalleryCommands:
    def test_list_with_no_people_is_not_an_error(self, config_file: Path) -> None:
        result = runner.invoke(app, ["gallery", "list", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "No identities configured" in result.output

    def test_list_emits_json(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["gallery", "list", "--config", str(config_file), "--json"]
        )
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_show_for_an_absent_identity_is_actionable(self, config_file: Path) -> None:
        result = runner.invoke(app, ["gallery", "show", "ghost", "--config", str(config_file)])
        assert result.exit_code == 1
        assert "no gallery entry" in result.output

    def test_remove_reports_when_nothing_was_removed(self, config_file: Path) -> None:
        result = runner.invoke(
            app, ["gallery", "remove", "ghost", "--config", str(config_file)]
        )
        assert result.exit_code == 0
        assert "No gallery entry" in result.output


class TestOverrides:
    def test_threshold_override_keeps_the_ordering_invariant(self) -> None:
        overrides = build_overrides(threshold=0.9)
        assert overrides["matching"]["recognition_threshold"] == 0.9
        assert overrides["matching"]["high_confidence_threshold"] >= 0.9

    def test_debug_flag_turns_on_debug_output_and_verbose_logs(self) -> None:
        overrides = build_overrides(debug=True)
        assert overrides["debug"]["enabled"] is True
        assert overrides["application"]["log_level"] == "DEBUG"

    def test_device_and_log_level_overrides(self) -> None:
        overrides = build_overrides(log_level="warning", device="CPU")
        assert overrides["application"]["log_level"] == "WARNING"
        assert overrides["device"]["device"] == "cpu"

    def test_no_flags_means_no_overrides(self) -> None:
        assert build_overrides() == {}


class TestProcessingErrors:
    def test_video_with_a_missing_file_is_reported_cleanly(
        self, config_file: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "video", "--input", str(tmp_path / "absent.mp4"),
                "--config", str(config_file), "--no-show",
            ],
        )
        assert result.exit_code == 1
        assert "Traceback" not in result.output
