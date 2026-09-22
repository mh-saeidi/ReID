"""Auxiliary subcommands: config, benchmark, evaluate, events, retention, serve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from src.cli.common import (
    DEFAULT_CONFIG,
    ConfigOption,
    DeviceOption,
    JsonOption,
    LogLevelOption,
    build_overrides,
    emit,
    engine_for,
    fail,
    handle_errors,
    load,
)
from src.config.schema import RecognitionMode, SourceKind
from src.output.metadata import write_json
from src.output.retention import apply_retention
from src.sources.factory import build_source, infer_kind
from src.tools.benchmark import run_benchmark
from src.tools.calibration import evaluate_dataset

config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)


@config_app.command("validate")
@handle_errors
def validate(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    check_models: Annotated[
        bool, typer.Option("--check-models", help="Also load the detector and ReID models.")
    ] = False,
    json_output: JsonOption = False,
) -> None:
    """Validate the configuration file and report what it resolves to."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    issues: list[str] = []

    from src.config.paths import ProjectPaths

    paths = ProjectPaths.from_config(app_config)
    for person in app_config.people:
        for raw in person.all_image_paths:
            resolved = paths.resolve(raw)
            if not resolved.exists():
                issues.append(f"people[{person.id}].image_path does not exist: {resolved}")

    if app_config.recognition.mode is RecognitionMode.FACE:
        for label, raw in (
            ("face.detector_model", app_config.face.detector_model),
            ("face.recognition_model", app_config.face.recognition_model),
        ):
            if not paths.resolve(raw).exists() and not Path(raw).exists():
                issues.append(
                    f"{label} does not exist: {paths.resolve(raw)} "
                    "-- run: python scripts/fetch_face_models.py"
                )

    payload = {
        "config": str(config.resolve()),
        "valid": True,
        "application": app_config.application.model_dump(mode="json"),
        "recognition": app_config.recognition.model_dump(mode="json"),
        "face": app_config.face.model_dump(mode="json"),
        "models": app_config.models.model_dump(mode="json"),
        "people": [p.model_dump(mode="json") for p in app_config.people],
        "paths": {
            "gallery": str(paths.gallery_dir),
            "output": str(paths.output_dir),
            "snapshots": str(paths.snapshots_dir),
            "videos": str(paths.videos_dir),
        },
        "warnings": issues,
    }

    if check_models:
        engine = engine_for(app_config)
        try:
            payload["engine"] = engine.describe()
        finally:
            engine.close()

    face_mode = app_config.recognition.mode is RecognitionMode.FACE
    lines = [
        f"Configuration OK: {config}",
        f"  people          : {len(app_config.people)} "
        f"({len(app_config.enabled_people)} enabled)",
        f"  recognition     : {app_config.recognition.mode.value}"
        + ("  (face only -- independent of clothing)" if face_mode
           else "  (whole-body appearance -- depends on clothing)"),
        f"  person detector : {app_config.models.detector}",
    ]
    if face_mode:
        # Which identity path will actually run, and the models it will use.
        # Reporting only the legacy ones is misleading on a deployment that
        # has enrolled a face gallery, which is exactly when somebody reads
        # this output to check their configuration.
        identity = app_config.face_identity
        gallery = paths.resolve(identity.gallery_dir)
        enrolled = (
            sum(1 for entry in gallery.iterdir()
                if entry.is_dir() and (entry / "metadata.json").exists())
            if gallery.is_dir() else 0
        )
        active = identity.enabled and enrolled > 0
        lines.append(
            "  identity path   : "
            + ("face_identity (passport-photo engine)" if active
               else "person gallery -- the face gallery is empty; "
                    "run 'identity build'")
        )
        if active:
            calibration = paths.resolve(identity.calibration_file)
            lines += [
                f"  face detector   : {identity.face_detector_model} "
                f"(conf {identity.face_detector_confidence})",
                f"  face encoder    : {identity.face_encoder_model}",
                f"  face gallery    : {gallery} ({enrolled} enrolled)",
                "  calibration     : "
                + (str(calibration) if calibration.exists()
                   else f"ABSENT -- using fallback_threshold "
                        f"{identity.fallback_threshold}; run 'calibrate'"),
            ]
        else:
            lines += [
                f"  face detector   : {app_config.face.detector_model}",
                f"  face encoder    : {app_config.face.recognition_model}",
            ]
        lines.append(
            f"  min face size   : {app_config.face.min_face_size}px "
            f"(identity held {app_config.face.identity_hold_frames} frames "
            "while the face is hidden)"
        )
    else:
        lines.append(f"  reid model      : {app_config.models.reid}")
    lines += [
        f"  threshold       : {app_config.matching.recognition_threshold} "
        f"(high {app_config.matching.high_confidence_threshold})",
        f"  tracking        : {'on' if app_config.tracking.enabled else 'off'} "
        f"({app_config.tracking.tracker})",
        f"  output directory: {paths.output_dir}",
    ]
    for issue in issues:
        lines.append(f"  WARNING         : {issue}")
    emit(payload, json_output, "\n".join(lines))


@config_app.command("show")
@handle_errors
def show(
    config: ConfigOption = Path(DEFAULT_CONFIG),
    section: Annotated[
        str | None, typer.Option("--section", "-s", help="Only show one top-level section.")
    ] = None,
) -> None:
    """Print the effective configuration after defaults and overrides."""
    app_config = load(config, build_overrides(log_level="WARNING"))
    data = app_config.model_dump(mode="json")
    if section:
        if section not in data:
            fail(f"unknown section '{section}'. Available: {', '.join(sorted(data))}")
        data = {section: data[section]}
    typer.echo(json.dumps(data, indent=2, default=str))


def register(app: typer.Typer) -> None:
    """Attach the auxiliary commands to the root CLI application."""

    @app.command("benchmark")
    @handle_errors
    def benchmark(
        config: ConfigOption = Path(DEFAULT_CONFIG),
        input_path: Annotated[
            str | None,
            typer.Option("--input", "-i", help="Video, image directory or camera index."),
        ] = None,
        frames: Annotated[
            int, typer.Option("--frames", "-n", min=1, help="Frames to measure.")
        ] = 200,
        warmup: Annotated[
            int, typer.Option("--warmup", min=0, help="Frames discarded before measuring.")
        ] = 5,
        check_targets: Annotated[
            bool,
            typer.Option(
                "--check-targets",
                help="Exit non-zero when the profile's targets are not met.",
            ),
        ] = False,
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="Write the JSON report here.")
        ] = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        json_output: JsonOption = False,
    ) -> None:
        """Measure detector, ReID and end-to-end throughput on real input."""
        app_config = load(
            config,
            build_overrides(
                log_level=log_level or "WARNING",
                device=device,
                extra={
                    "display": {"show_window": False},
                    "output": {"save_video": False, "save_snapshots": False,
                               "save_metadata": False},
                    "recording": {"mode": "disabled"},
                },
            ),
        )
        engine = engine_for(app_config)
        try:
            engine.ensure_gallery()
            target = input_path if input_path is not None else app_config.source.path
            if target is None:
                fail("benchmark needs --input (a video, image directory or camera index)")
            kind = infer_kind(str(target), app_config)
            source = build_source(app_config, engine.paths, kind=kind, target=str(target))
            pipeline = engine.pipeline(
                use_tracking=app_config.tracking.enabled and kind is not SourceKind.DIRECTORY
            )
            report = run_benchmark(
                engine,
                pipeline,
                source,
                max_frames=frames,
                warmup=warmup,
                collect_telemetry=app_config.benchmark.collect_system_telemetry,
            )
            if output is not None:
                write_json(output, report.to_dict(), overwrite=True)
                typer.echo(f"Report written to {output}")
            emit(report.to_dict(), json_output, report.render())

            from src.tools.benchmark import Verdict  # noqa: PLC0415

            if check_targets:
                if report.verdict is Verdict.NOT_CHECKED:
                    fail(
                        "--check-targets was given but this profile defines no "
                        "targets. Set benchmark.targets.enabled and the limits "
                        "that matter for this deployment."
                    )
                if report.verdict is Verdict.FAIL:
                    raise typer.Exit(1)
        finally:
            engine.close()

    @app.command("evaluate")
    @handle_errors
    def evaluate(
        dataset: Annotated[
            Path, typer.Option("--dataset", "-d", help="Evaluation dataset directory.")
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="Write the JSON report here.")
        ] = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        json_output: JsonOption = False,
    ) -> None:
        """Measure similarity distributions and candidate thresholds on your data."""
        app_config = load(
            config,
            build_overrides(
                log_level=log_level or "WARNING",
                device=device,
                extra={
                    "display": {"show_window": False},
                    "output": {"save_snapshots": False, "save_metadata": False,
                               "save_video": False},
                },
            ),
        )
        engine = engine_for(app_config)
        try:
            engine.ensure_gallery()
            pipeline = engine.pipeline(use_tracking=False)
            report = evaluate_dataset(engine, pipeline, dataset)
            if output is not None:
                write_json(output, report.to_dict(), overwrite=True)
                typer.echo(f"Report written to {output}")
            emit(report.to_dict(), json_output, report.render())
        finally:
            engine.close()

    @app.command("events")
    @handle_errors
    def events(
        config: ConfigOption = Path(DEFAULT_CONFIG),
        limit: Annotated[int, typer.Option("--limit", "-n", min=1)] = 50,
        event_type: Annotated[
            str | None, typer.Option("--type", help="Filter by event type.")
        ] = None,
        json_output: JsonOption = False,
    ) -> None:
        """Show recent events from the persisted event log."""
        app_config = load(config, build_overrides(log_level="WARNING"))
        from src.config.paths import ProjectPaths

        paths = ProjectPaths.from_config(app_config)
        path = paths.events_dir / app_config.events.filename
        if not path.exists():
            emit([], json_output, f"No event log at {path}.")
            return
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event_type and record.get("type") != event_type:
                    continue
                records.append(record)
        records = records[-limit:]
        lines = [
            f"{r.get('iso_time', '')}  {r.get('type', ''):<26} "
            f"{r.get('identity_name') or r.get('track_id') or ''}"
            for r in records
        ]
        emit(records, json_output, "\n".join(lines) or "No matching events.")

    @app.command("retention")
    @handle_errors
    def retention(
        config: ConfigOption = Path(DEFAULT_CONFIG),
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="Report what would be deleted.")
        ] = False,
        json_output: JsonOption = False,
    ) -> None:
        """Apply the configured retention policy to stored output."""
        app_config = load(config, build_overrides(log_level="INFO"))
        from src.config.paths import ProjectPaths

        paths = ProjectPaths.from_config(app_config)
        if not app_config.retention.enabled:
            emit(
                {"enabled": False},
                json_output,
                "retention.enabled is false; nothing was removed.",
            )
            return
        report = apply_retention(app_config.retention, paths, dry_run=dry_run)
        emit(
            report.to_dict(),
            json_output,
            f"{'Would remove' if dry_run else 'Removed'} {report.removed_files} file(s), "
            f"{report.removed_bytes / (1024 * 1024):.1f} MB.",
        )

    @app.command("serve")
    @handle_errors
    def serve(
        config: ConfigOption = Path(DEFAULT_CONFIG),
        host: Annotated[str | None, typer.Option("--host")] = None,
        port: Annotated[int | None, typer.Option("--port")] = None,
        log_level: LogLevelOption = None,
    ) -> None:
        """Start the optional REST API (requires fastapi and uvicorn)."""
        app_config = load(config, build_overrides(log_level=log_level))
        try:
            from src.api.server import serve as serve_api
        except ImportError as exc:
            fail(
                f"the API layer needs extra packages ({exc}). "
                "Install them with: pip install 'fastapi' 'uvicorn[standard]' "
                "python-multipart"
            )
            return
        serve_api(
            app_config,
            host=host or app_config.api.host,
            port=port or app_config.api.port,
        )

    app.add_typer(config_app, name="config")
    app.add_typer(benchmark_app, name="benchmark-suite")


benchmark_app = typer.Typer(
    help="Benchmark scenarios and configuration sweeps.", no_args_is_help=True
)


@benchmark_app.command("matrix")
@handle_errors
def benchmark_matrix(
    input_path: Annotated[
        str, typer.Option("--input", "-i", help="Video, image directory or camera index.")
    ],
    config: ConfigOption = Path(DEFAULT_CONFIG),
    group: Annotated[
        list[str] | None,
        typer.Option("--group", "-g", help="Scenario groups to run (repeatable)."),
    ] = None,
    frames: Annotated[int, typer.Option("--frames", "-n", min=1)] = 120,
    warmup: Annotated[int, typer.Option("--warmup", min=0)] = 10,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the JSON report here.")
    ] = None,
    log_level: LogLevelOption = None,
    device: DeviceOption = None,
    json_output: JsonOption = False,
) -> None:
    """Sweep configurations against one input and compare them.

    Every scenario runs on the same frames with only the setting under test
    changed, so the comparison is attributable. Use it to choose a batch size,
    a recognition interval or a face-search strategy from measurements on the
    target hardware instead of from defaults.
    """
    from src.sources.factory import build_source as make_source
    from src.tools.benchmark_matrix import build_scenarios, run_matrix, write_report

    app_config = load(
        config,
        build_overrides(
            log_level=log_level or "WARNING",
            device=device,
            extra={
                "display": {"show_window": False},
                "events": {"enabled": False},
            },
        ),
    )
    from src.config.paths import ProjectPaths

    paths = ProjectPaths.from_config(app_config)
    kind = infer_kind(str(input_path), app_config)

    try:
        scenarios = build_scenarios(app_config, paths, group)
    except ValueError as exc:
        fail(str(exc))
        return

    typer.echo(
        f"Running {len(scenarios)} scenario(s) x {frames} frames. "
        "Each builds its own engine, so this takes a while."
    )

    def progress(name: str, index: int, total: int) -> None:
        typer.echo(f"  [{index}/{total}] {name}")

    report = run_matrix(
        app_config,
        scenarios,
        source_factory=lambda cfg: make_source(
            cfg, ProjectPaths.from_config(cfg), kind=kind, target=str(input_path)
        ),
        frames=frames,
        warmup=warmup,
        input_label=str(input_path),
        progress=progress,
    )

    if output is not None:
        write_report(report, output)
        typer.echo(f"\nJSON report written to {output}")
    emit(report.to_dict(), json_output, "\n" + report.render())


@benchmark_app.command("scenarios")
@handle_errors
def benchmark_scenarios() -> None:
    """List the available benchmark scenario groups."""
    from src.tools.benchmark_matrix import GROUPS

    typer.echo("Scenario groups:")
    for name in GROUPS:
        typer.echo(f"  {name}")
    typer.echo("\nRun with: python main.py benchmark matrix --input <video> -g <group>")
