"""YAML configuration loading, layering, path resolution and validation."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.schema import AppConfig
from src.core.exceptions import ConfigurationError

__all__ = [
    "load_config",
    "load_yaml",
    "deep_merge",
    "format_validation_error",
    "resolve_path",
]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary with actionable error messages."""
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(
            f"configuration file not found: {path}. "
            "Pass --config <file> or create config.yaml in the project root."
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        raise ConfigurationError(f"cannot read configuration file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (lists are replaced wholesale)."""
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references inside string values."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            env_value = os.environ.get(name)
            if env_value is None:
                if default is None:
                    raise ConfigurationError(
                        f"environment variable '{name}' referenced in the configuration "
                        "is not set and has no default (use ${" + name + ":-default})"
                    )
                return default
            return env_value

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def resolve_path(path: str | Path, base_dir: str | Path | None) -> Path:
    """Resolve a configured path relative to the configuration file's directory.

    Absolute paths are returned untouched, so nothing in this project depends on
    the process working directory.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return (Path(base_dir) / candidate).resolve()


def format_validation_error(error: ValidationError, source: str) -> str:
    """Turn a pydantic error into a operator-readable, actionable message."""
    lines = [f"invalid configuration in {source}:"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"  - {location}: {item['msg']}")
    return "\n".join(lines)


def load_config(
    path: str | Path,
    *,
    defaults: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load, layer, expand and validate the application configuration.

    Layering order (later wins): ``defaults`` file -> ``path`` file -> ``overrides``.
    """
    config_path = Path(path).expanduser().resolve()
    base_dir = config_path.parent

    merged: dict[str, Any] = {}
    if defaults is not None:
        merged = deep_merge(merged, load_yaml(defaults))
    merged = deep_merge(merged, load_yaml(config_path))
    if overrides:
        merged = deep_merge(merged, overrides)

    merged = expand_env(merged)
    merged.pop("config_path", None)
    merged.pop("base_dir", None)

    try:
        config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationError(format_validation_error(exc, str(config_path))) from exc

    config.config_path = str(config_path)
    config.base_dir = str(base_dir)
    return config


def config_from_dict(data: Mapping[str, Any], base_dir: str | Path | None = None) -> AppConfig:
    """Validate an in-memory mapping (used by tests and the API layer)."""
    payload = dict(data)
    payload.pop("config_path", None)
    payload.pop("base_dir", None)
    try:
        config = AppConfig.model_validate(expand_env(payload))
    except ValidationError as exc:
        raise ConfigurationError(format_validation_error(exc, "<in-memory configuration>")) from exc
    config.base_dir = str(base_dir) if base_dir is not None else None
    return config
