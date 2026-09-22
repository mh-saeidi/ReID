"""``system`` and ``models`` subcommands: capability inspection and engine builds."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from src.cli.common import (
    DEFAULT_CONFIG,
    ConfigOption,
    JsonOption,
    build_overrides,
    emit,
    fail,
    handle_errors,
    load,
)

system_app = typer.Typer(
    help="Inspect the platform and its acceleration capabilities.", no_args_is_help=True
)
models_app = typer.Typer(help="Manage model artefacts and TensorRT engines.",
                         no_args_is_help=True)


@system_app.command("info")
@handle_errors
def system_info(json_output: JsonOption = False) -> None:
    """Report platform, GPU, TensorRT, GStreamer and encoder availability.

    Read-only: nothing here changes nvpmodel, jetson_clocks, fan settings or any
    other system-wide configuration, and none of it requires root.
    """
    from src.hardware.capabilities import detect_capabilities, render_capabilities

    caps = detect_capabilities()
    emit(caps.to_dict(), json_output, render_capabilities(caps))


@system_app.command("backends")
@handle_errors
def system_backends(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    json_output: JsonOption = False,
) -> None:
    """Show which inference backend each model would use with this config."""
    from src.backends.selection import ModelRole, select_backend
    from src.config.paths import ProjectPaths
    from src.config.schema import RecognitionMode
    from src.hardware.capabilities import detect_capabilities

    app_config = load(config, build_overrides(log_level="WARNING"))
    paths = ProjectPaths.from_config(app_config)
    caps = detect_capabilities()

    targets = [(ModelRole.DETECTOR, app_config.models.detector)]
    if app_config.recognition.mode is RecognitionMode.FACE:
        targets += [
            (ModelRole.FACE_DETECTOR, app_config.face.detector_model),
            (ModelRole.FACE_ENCODER, app_config.face.recognition_model),
        ]
    else:
        targets.append((ModelRole.BODY_ENCODER, app_config.models.reid))

    rows = []
    for role, raw in targets:
        resolved = paths.resolve(raw)
        decision = select_backend(role, str(resolved), app_config, caps)
        rows.append(decision.to_dict() | {"model": str(resolved)})

    lines = ["Backend selection", ""]
    for row in rows:
        marker = "  (fallback)" if row["fallback_from"] else ""
        lines.append(f"  {row['role']:<14}: {row['backend']:<10}{marker}")
        lines.append(f"    model  : {Path(row['model']).name}")
        lines.append(f"    reason : {row['reason']}")
    emit(rows, json_output, "\n".join(lines))


@models_app.command("build-tensorrt")
@handle_errors
def build_tensorrt(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even when a valid engine exists.")
    ] = False,
    precision: Annotated[
        str | None,
        typer.Option("--precision", help="Override backend.tensorrt.precision."),
    ] = None,
    max_batch: Annotated[
        int | None, typer.Option("--max-batch", help="Override the maximum batch size.")
    ] = None,
    role: Annotated[
        list[str] | None,
        typer.Option("--role", help="Build only these roles (repeatable)."),
    ] = None,
    json_output: JsonOption = False,
) -> None:
    """Build the TensorRT engines this configuration needs.

    Engines are derived artefacts tied to this GPU, TensorRT version and CUDA
    version. They are rebuilt automatically when the source model or any build
    parameter changes, and are never assumed portable between machines.
    """
    from src.backends.engine_store import EngineStore
    from src.backends.tensorrt_builder import (
        TensorRTBuilder,
        TensorRTUnavailable,
        default_build_requests,
    )
    from src.config.paths import ProjectPaths
    from src.hardware.capabilities import detect_capabilities

    overrides: dict = {}
    if precision or max_batch:
        tensorrt: dict = {}
        if precision:
            tensorrt["precision"] = precision.lower()
        if max_batch:
            tensorrt["max_batch_size"] = max_batch
            tensorrt["optimal_batch_size"] = min(max_batch, 4)
        overrides["backend"] = {"tensorrt": tensorrt}

    app_config = load(config, build_overrides(log_level="INFO", extra=overrides))
    paths = ProjectPaths.from_config(app_config)
    caps = detect_capabilities()

    if not caps.has_tensorrt:
        fail(
            "TensorRT is not available on this system, so no engine can be "
            "built here. Engines are not portable, so they must be built on the "
            "target device. Run 'python main.py system info' to see what is "
            "detected. The ONNX Runtime backend works without TensorRT."
        )

    trt_config = app_config.backend.tensorrt
    store = EngineStore(
        paths.resolve(trt_config.engine_dir),
        strict_version_check=trt_config.strict_version_check,
    )
    builder = TensorRTBuilder(trt_config, store, caps)

    requests = default_build_requests(app_config, paths)
    if role:
        wanted = set(role)
        requests = [r for r in requests if r.role in wanted]
    if not requests:
        fail(
            "nothing to build: TensorRT engines are built from ONNX sources, "
            "and no configured model is an .onnx file. Export the detector with "
            "`yolo export model=... format=onnx dynamic=True` first."
        )

    results, failures = [], {}
    for request in requests:
        try:
            outcome = builder.ensure_engine(request, force=force)
        except TensorRTUnavailable as exc:
            failures[request.role] = str(exc)
            typer.secho(f"  {request.role}: FAILED - {exc}", fg=typer.colors.RED, err=True)
            continue
        results.append(
            {
                "role": request.role,
                "source": str(request.source),
                "engine": str(outcome.engine_path),
                "precision": outcome.metadata.precision,
                "rebuilt": outcome.rebuilt,
                "build_seconds": round(outcome.build_seconds, 2),
                "builder": outcome.backend,
            }
        )

    lines = [f"TensorRT engines in {store.directory}", ""]
    for row in results:
        state = "built" if row["rebuilt"] else "reused"
        lines.append(
            f"  {row['role']:<14} {state:<7} {row['precision']:<5} "
            f"{Path(row['engine']).name}"
            + (f"  ({row['build_seconds']:.1f}s)" if row["rebuilt"] else "")
        )
    for failed_role, message in failures.items():
        lines.append(f"  {failed_role:<14} FAILED   {message}")
    if failures:
        lines.append("")
        lines.append("  Roles that failed will use their ONNX/PyTorch backend instead.")

    emit(
        {"engines": results, "failed": failures, "directory": str(store.directory)},
        json_output,
        "\n".join(lines),
    )
    if failures:
        raise typer.Exit(1)


@models_app.command("list-engines")
@handle_errors
def list_engines(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    json_output: JsonOption = False,
) -> None:
    """List the TensorRT engines on disk and their provenance."""
    from src.backends.engine_store import EngineStore
    from src.config.paths import ProjectPaths

    app_config = load(config, build_overrides(log_level="WARNING"))
    paths = ProjectPaths.from_config(app_config)
    store = EngineStore(paths.resolve(app_config.backend.tensorrt.engine_dir))
    rows = store.list_engines()

    if not rows:
        emit([], json_output, f"No engines in {store.directory}.")
        return
    lines = [f"{'ROLE':<14} {'PRECISION':<10} {'SIZE':>8}  {'TRT':<9} ENGINE"]
    for row in rows:
        lines.append(
            f"{str(row['role'] or '-'):<14} {str(row['precision'] or '-'):<10} "
            f"{row['size_mb']:>7.1f}M  {str(row['tensorrt'] or '-'):<9} "
            f"{Path(row['engine']).name}"
        )
    emit(rows, json_output, "\n".join(lines))


@models_app.command("clean-engines")
@handle_errors
def clean_engines(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not prompt.")] = False,
) -> None:
    """Delete every cached TensorRT engine (sources are untouched)."""
    from src.backends.engine_store import EngineStore
    from src.config.paths import ProjectPaths

    app_config = load(config, build_overrides(log_level="WARNING"))
    paths = ProjectPaths.from_config(app_config)
    store = EngineStore(paths.resolve(app_config.backend.tensorrt.engine_dir))
    engines = store.list_engines()
    if not engines:
        typer.echo(f"No engines in {store.directory}.")
        return
    if not yes:
        typer.confirm(f"Delete {len(engines)} engine(s) in {store.directory}?", abort=True)
    removed = sum(int(store.remove(Path(row["engine"]))) for row in engines)
    typer.echo(f"Removed {removed} engine(s). Source models were not touched.")


def register(app: typer.Typer) -> None:
    app.add_typer(system_app, name="system")
    app.add_typer(models_app, name="models")
