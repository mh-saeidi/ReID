"""Shared CLI plumbing: option types, config loading and error presentation."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from src.config.loader import load_config
from src.config.schema import AppConfig
from src.core.exceptions import ReIDSystemError
from src.pipeline.engine import Engine, build_engine
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_CONFIG = "config.yaml"

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Path to the YAML configuration file.",
        show_default=True,
        rich_help_panel="Global",
    ),
]
LogLevelOption = Annotated[
    str | None,
    typer.Option(
        "--log-level",
        help="Override application.log_level (DEBUG, INFO, WARNING, ERROR).",
        rich_help_panel="Global",
    ),
]
DeviceOption = Annotated[
    str | None,
    typer.Option("--device-type", help="Override device.device (auto, cpu, cuda, mps).",
                 rich_help_panel="Global"),
]
DebugOption = Annotated[
    bool,
    typer.Option("--debug", help="Enable debug artefacts (crops, matches, track state).",
                 rich_help_panel="Global"),
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit machine-readable JSON on stdout."),
]


def build_overrides(
    *,
    log_level: str | None = None,
    device: str | None = None,
    debug: bool = False,
    threshold: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate global CLI flags into a configuration override mapping."""
    overrides: dict[str, Any] = {}
    if log_level:
        overrides.setdefault("application", {})["log_level"] = log_level.upper()
    if device:
        overrides.setdefault("device", {})["device"] = device.lower()
    if debug:
        overrides["debug"] = {"enabled": True}
        overrides.setdefault("application", {}).setdefault("log_level", "DEBUG")
    if threshold is not None:
        matching = overrides.setdefault("matching", {})
        matching["recognition_threshold"] = threshold
        # Keep the ordering invariant satisfied when only the lower bound moves.
        matching.setdefault("high_confidence_threshold", max(threshold, threshold + 0.15))
    if extra:
        overrides.update(extra)
    return overrides


def load(config_path: Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load and validate configuration, configuring logging as a side effect."""
    config = load_config(config_path, overrides=overrides)
    setup_logging(
        config.application.log_level,
        log_file=config.application.log_file,
        json_logs=config.application.json_logs,
        force=True,
    )
    return config


def engine_for(config: AppConfig, *, load_models: bool = True) -> Engine:
    return build_engine(config, load_models=load_models)


def fail(message: str, code: int = 1) -> None:
    """Print an actionable error and exit."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def handle_errors(func):
    """Turn application exceptions into clean CLI errors instead of tracebacks."""

    @functools.wraps(func)  # keeps the signature Typer introspects for options
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ReIDSystemError as exc:
            fail(str(exc))
        except KeyboardInterrupt:  # pragma: no cover - interactive
            typer.secho("\ninterrupted", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(130) from None
        except FileNotFoundError as exc:
            fail(f"file not found: {exc}")

    return wrapper


def emit(payload: Any, json_output: bool, text: str | None = None) -> None:
    """Print either JSON or human-readable text."""
    if json_output:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    elif text is not None:
        typer.echo(text)
