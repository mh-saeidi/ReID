"""Configuration loading, layering and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.loader import config_from_dict, deep_merge, load_config, load_yaml, resolve_path
from src.config.paths import ProjectPaths
from src.config.schema import AppConfig, RecognitionMode
from src.core.exceptions import ConfigurationError


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestYamlLoading:
    def test_loads_a_valid_file(self, tmp_path: Path, base_config_dict: dict) -> None:
        path = write_yaml(tmp_path / "config.yaml", base_config_dict)
        config = load_config(path)
        assert config.application.log_level == "WARNING"
        assert config.config_path == str(path.resolve())
        assert config.base_dir == str(tmp_path.resolve())

    def test_empty_file_yields_schema_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        config = load_config(path)
        assert config.application.name
        assert config.people == []

    def test_missing_file_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="configuration file not found"):
            load_config(tmp_path / "nope.yaml")

    def test_malformed_yaml_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("people:\n  - id: 'unterminated\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid YAML"):
            load_config(path)

    def test_top_level_must_be_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping at the top level"):
            load_yaml(path)


class TestValidation:
    def test_unknown_key_is_rejected(self, base_config_dict: dict) -> None:
        base_config_dict["matching"]["totally_made_up"] = 1
        with pytest.raises(ConfigurationError, match="totally_made_up"):
            config_from_dict(base_config_dict)

    def test_threshold_ordering_is_enforced(self, base_config_dict: dict) -> None:
        base_config_dict["matching"] = {
            "recognition_threshold": 0.8,
            "high_confidence_threshold": 0.5,
        }
        with pytest.raises(ConfigurationError, match="high_confidence_threshold"):
            config_from_dict(base_config_dict)

    def test_duplicate_person_ids_are_rejected(self, base_config_dict: dict) -> None:
        base_config_dict["people"] = [
            {"id": "dup", "name": "One", "image_path": "a.jpg"},
            {"id": "dup", "name": "Two", "image_path": "b.jpg"},
        ]
        with pytest.raises(ConfigurationError, match="duplicate people"):
            config_from_dict(base_config_dict)

    def test_person_requires_a_name(self, base_config_dict: dict) -> None:
        base_config_dict["people"] = [{"id": "x", "name": "   ", "image_path": "a.jpg"}]
        with pytest.raises(ConfigurationError):
            config_from_dict(base_config_dict)

    def test_person_requires_a_reference_image(self, base_config_dict: dict) -> None:
        base_config_dict["people"] = [{"id": "x", "name": "X"}]
        with pytest.raises(ConfigurationError, match="no reference image"):
            config_from_dict(base_config_dict)

    def test_person_id_must_be_filename_safe(self, base_config_dict: dict) -> None:
        base_config_dict["people"] = [
            {"id": "bad/id", "name": "X", "image_path": "a.jpg"}
        ]
        with pytest.raises(ConfigurationError, match="unsafe in"):
            config_from_dict(base_config_dict)

    def test_both_recognition_modes_are_selectable(self, base_config_dict: dict) -> None:
        base_config_dict["recognition"] = {"mode": "face"}
        assert config_from_dict(base_config_dict).recognition.mode is RecognitionMode.FACE

        base_config_dict["recognition"] = {"mode": "person_reid"}
        assert (
            config_from_dict(base_config_dict).recognition.mode
            is RecognitionMode.PERSON_REID
        )

    def test_an_unknown_recognition_mode_is_rejected(self, base_config_dict: dict) -> None:
        base_config_dict["recognition"] = {"mode": "fingerprint"}
        with pytest.raises(ConfigurationError):
            config_from_dict(base_config_dict)

    def test_face_is_the_shipped_default(self) -> None:
        """Identity should be clothing-independent unless told otherwise."""
        from src.config.schema import RecognitionSectionConfig

        assert RecognitionSectionConfig().mode is RecognitionMode.FACE


class TestModeAwareThresholds:
    """Face and body embeddings occupy different similarity regimes."""

    def test_unset_thresholds_follow_the_mode(self, base_config_dict: dict) -> None:
        base_config_dict.pop("matching", None)

        base_config_dict["recognition"] = {"mode": "face"}
        face = config_from_dict(base_config_dict)

        base_config_dict["recognition"] = {"mode": "person_reid"}
        body = config_from_dict(base_config_dict)

        assert face.matching.recognition_threshold < body.matching.recognition_threshold
        assert face.matching.high_confidence_threshold >= face.matching.recognition_threshold

    def test_an_explicit_threshold_always_wins(self, base_config_dict: dict) -> None:
        base_config_dict["recognition"] = {"mode": "face"}
        base_config_dict["matching"] = {
            "recognition_threshold": 0.91,
            "high_confidence_threshold": 0.95,
        }
        config = config_from_dict(base_config_dict)
        assert config.matching.recognition_threshold == 0.91

    def test_half_specified_thresholds_stay_consistent(self, base_config_dict: dict) -> None:
        base_config_dict["recognition"] = {"mode": "face"}
        base_config_dict["matching"] = {"recognition_threshold": 0.50}
        config = config_from_dict(base_config_dict)
        assert config.matching.recognition_threshold == 0.50
        assert config.matching.high_confidence_threshold >= 0.50

    def test_reid_interval_requires_tracking(self, base_config_dict: dict) -> None:
        base_config_dict["performance"] = {"reid_interval": 3}
        base_config_dict["tracking"] = {"enabled": False}
        with pytest.raises(ConfigurationError, match="requires tracking.enabled"):
            config_from_dict(base_config_dict)

    def test_metric_mismatch_is_rejected(self, base_config_dict: dict) -> None:
        base_config_dict["matching"]["metric"] = "euclidean"
        base_config_dict["reid"] = {"similarity_metric": "cosine"}
        with pytest.raises(ConfigurationError, match="must agree"):
            config_from_dict(base_config_dict)

    def test_non_person_classes_are_rejected(self, base_config_dict: dict) -> None:
        base_config_dict["detector"] = {"classes": ["person", "car"]}
        with pytest.raises(ConfigurationError, match="person"):
            config_from_dict(base_config_dict)

    def test_minimum_frames_cannot_exceed_history(self, base_config_dict: dict) -> None:
        base_config_dict["tracking"] = {
            "identity_stability": {"history_size": 3, "minimum_recognized_frames": 5}
        }
        with pytest.raises(ConfigurationError, match="cannot exceed history_size"):
            config_from_dict(base_config_dict)


class TestLayeringAndPaths:
    def test_deep_merge_overrides_nested_keys_only(self) -> None:
        merged = deep_merge(
            {"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20, "z": 30}, "c": 4}
        )
        assert merged == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}

    def test_defaults_are_layered_under_the_main_file(
        self, tmp_path: Path, base_config_dict: dict
    ) -> None:
        defaults = write_yaml(
            tmp_path / "defaults.yaml", {"detector": {"imgsz": 1280, "confidence": 0.9}}
        )
        main = write_yaml(tmp_path / "config.yaml", {**base_config_dict, "detector": {"imgsz": 640}})
        config = load_config(main, defaults=defaults)
        assert config.detector.imgsz == 640          # the main file wins
        assert config.detector.confidence == 0.9     # the default survives

    def test_cli_overrides_win(self, tmp_path: Path, base_config_dict: dict) -> None:
        main = write_yaml(tmp_path / "config.yaml", base_config_dict)
        config = load_config(main, overrides={"matching": {"recognition_threshold": 0.42}})
        assert config.matching.recognition_threshold == 0.42

    def test_env_expansion(self, tmp_path: Path, base_config_dict: dict, monkeypatch) -> None:
        monkeypatch.setenv("TEST_REID_MODEL", "models/custom-reid.onnx")
        base_config_dict["models"] = {
            "detector": "${TEST_MISSING:-models/yolo26n.pt}",
            "reid": "${TEST_REID_MODEL}",
        }
        config = load_config(write_yaml(tmp_path / "c.yaml", base_config_dict))
        assert config.models.reid == "models/custom-reid.onnx"
        assert config.models.detector == "models/yolo26n.pt"

    def test_missing_env_without_default_fails(
        self, tmp_path: Path, base_config_dict: dict, monkeypatch
    ) -> None:
        monkeypatch.delenv("TEST_ABSENT", raising=False)
        base_config_dict["models"] = {"reid": "${TEST_ABSENT}"}
        with pytest.raises(ConfigurationError, match="TEST_ABSENT"):
            load_config(write_yaml(tmp_path / "c.yaml", base_config_dict))

    def test_relative_paths_resolve_against_the_config_file(self, tmp_path: Path) -> None:
        assert resolve_path("data/x", tmp_path) == (tmp_path / "data" / "x").resolve()
        absolute = Path("/tmp/abs")
        assert resolve_path(absolute, tmp_path) == absolute

    def test_project_paths_derive_every_directory(self, config: AppConfig) -> None:
        paths = ProjectPaths.from_config(config)
        assert paths.embeddings_dir.parent == paths.gallery_dir
        assert paths.snapshots_dir.parent == paths.output_dir
        assert paths.videos_dir.name == "videos"


class TestShippedConfigs:
    """The files in the repository must themselves be valid."""

    @pytest.mark.parametrize(
        "name", ["config.yaml", "configs/default.yaml", "configs/production.yaml"]
    )
    def test_repository_config_is_valid(self, name: str) -> None:
        path = Path(__file__).resolve().parent.parent / name
        config = load_config(path)
        assert isinstance(config, AppConfig)
        assert config.matching.high_confidence_threshold >= config.matching.recognition_threshold
