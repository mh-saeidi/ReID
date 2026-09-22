"""Processing subcommands: webcam, video, image, images, stream, run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from src.cli.common import (
    DEFAULT_CONFIG,
    ConfigOption,
    DebugOption,
    DeviceOption,
    JsonOption,
    LogLevelOption,
    build_overrides,
    emit,
    engine_for,
    handle_errors,
    load,
)
from src.config.schema import SourceKind
from src.pipeline.engine import Engine
from src.pipeline.image_runner import ImageRunner
from src.pipeline.runner import StreamRunner
from src.sources.factory import build_source, infer_kind

ThresholdOption = Annotated[
    float | None,
    typer.Option(
        "--threshold",
        "-t",
        min=-1.0,
        max=1.0,
        help="Override matching.recognition_threshold for this run.",
    ),
]
ShowOption = Annotated[
    bool | None,
    typer.Option("--show/--no-show", help="Show (or suppress) the live preview window."),
]
SaveVideoOption = Annotated[
    bool | None,
    typer.Option("--save-video/--no-save-video", help="Override output.save_video."),
]
AsyncOption = Annotated[
    bool | None,
    typer.Option(
        "--async/--sync",
        help="Run capture, inference and output on separate threads.",
    ),
]
RecordOption = Annotated[
    str | None,
    typer.Option("--record", help="Override recording.mode (continuous, event, disabled)."),
]
SnapshotOption = Annotated[
    str | None,
    typer.Option(
        "--snapshots",
        help="Override output.snapshot_mode (all, recognized, unknown, events_only, disabled).",
    ),
]


def _overrides(
    log_level: str | None,
    device: str | None,
    debug: bool,
    threshold: float | None,
    record: str | None,
    snapshots: str | None,
    async_mode: bool | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if async_mode is not None:
        extra["pipeline"] = {"async": async_mode}
    if record:
        extra["recording"] = {"mode": record.lower()}
    if snapshots:
        mode = snapshots.lower()
        extra["output"] = {
            "snapshot_mode": mode,
            "save_snapshots": mode != "disabled",
        }
    return build_overrides(
        log_level=log_level, device=device, debug=debug, threshold=threshold, extra=extra
    )


def _run_stream(
    engine: Engine,
    kind: SourceKind,
    target: str | int | None,
    *,
    show: bool | None,
    save_video: bool | None,
    json_output: bool,
) -> None:
    engine.ensure_gallery()
    pipeline = engine.pipeline(use_tracking=engine.config.tracking.enabled)
    source = build_source(engine.config, engine.paths, kind=kind, target=target)

    # Both runners produce the same results and the same output artefacts; the
    # asynchronous one moves capture and output onto their own threads.
    if engine.config.pipeline.async_enabled:
        from src.pipeline.async_runner import AsyncStreamRunner  # noqa: PLC0415

        runner = AsyncStreamRunner(
            engine, pipeline, show_window=show, save_video=save_video
        )
    else:
        runner = StreamRunner(engine, pipeline, show_window=show, save_video=save_video)
    summary = runner.run(source)
    emit(summary.to_dict(), json_output, _format_stream_summary(summary))


def _format_stream_summary(summary) -> str:
    lines = [
        "",
        f"Source            : {summary.source_id}",
        f"Frames            : {summary.frames}  ({summary.fps:.1f} FPS end-to-end)",
        f"Detections        : {summary.detections} "
        f"({summary.recognized} recognized, {summary.unknown} unknown)",
    ]
    if summary.identities:
        top = ", ".join(f"{k}={v}" for k, v in list(summary.identities.items())[:8])
        lines.append(f"Identity frames   : {top}")
    if summary.video_path:
        lines.append(f"Annotated video   : {summary.video_path}")
    for clip in summary.clips:
        lines.append(f"Event clip        : {clip}")
    if summary.snapshots:
        lines.append(f"Snapshots saved   : {summary.snapshots}")
    if summary.metadata_path:
        lines.append(f"Detection metadata: {summary.metadata_path}")
    capture = getattr(summary, "capture", None)
    if capture and capture.get("frames_dropped"):
        lines.append(
            f"Frames dropped     : {capture['frames_dropped']} "
            f"({capture['drop_rate_percent']:.1f}% -- the pipeline preferred "
            "the newest frames)"
        )
    output = getattr(summary, "output", None)
    if output and output.get("dropped_visualization"):
        lines.append(
            f"Visualization shed : {output['dropped_visualization']} frame(s); "
            "events and metadata were preserved"
        )
    return "\n".join(lines)


def _format_image_summary(summary) -> str:
    lines = [
        "",
        f"Images processed  : {summary.images}",
        f"Detections        : {summary.detections} "
        f"({summary.recognized} recognized, {summary.unknown} unknown)",
    ]
    if summary.identities:
        lines.append(
            "Recognized        : "
            + ", ".join(f"{k}={v}" for k, v in summary.identities.items())
        )
    for outcome in summary.outcomes[:20]:
        people = [
            f"{d.identity_name or 'Unknown'}"
            + (f" ({d.reid_similarity:.2f})" if d.reid_similarity else "")
            for d in outcome.result.detections
        ]
        lines.append(
            f"  {Path(outcome.source_path).name}: "
            + (", ".join(people) if people else "no person detected")
        )
    if len(summary.outcomes) > 20:
        lines.append(f"  ... and {len(summary.outcomes) - 20} more")
    for path, message in summary.failures.items():
        lines.append(f"  FAILED {path}: {message}")
    return "\n".join(lines)


def register(app: typer.Typer) -> None:
    """Attach the processing commands to the root CLI application."""

    @app.command("webcam")
    @handle_errors
    def webcam(
        config: ConfigOption = Path(DEFAULT_CONFIG),
        device_index: Annotated[
            str | None,
            typer.Option("--device", "-d", help="Camera index, or 'auto' to probe."),
        ] = None,
        show: ShowOption = None,
        save_video: SaveVideoOption = None,
        threshold: ThresholdOption = None,
        record: RecordOption = None,
        snapshots: SnapshotOption = None,
        async_mode: AsyncOption = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        debug: DebugOption = False,
        json_output: JsonOption = False,
    ) -> None:
        """Process a live camera stream."""
        app_config = load(
            config,
            _overrides(log_level, device, debug, threshold, record, snapshots, async_mode),
        )
        engine = engine_for(app_config)
        try:
            _run_stream(
                engine,
                SourceKind.WEBCAM,
                device_index,
                show=show,
                save_video=save_video,
                json_output=json_output,
            )
        finally:
            engine.close()

    @app.command("video")
    @handle_errors
    def video(
        input_path: Annotated[
            Path, typer.Option("--input", "-i", help="Video file to process.")
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        show: ShowOption = None,
        save_video: SaveVideoOption = None,
        threshold: ThresholdOption = None,
        record: RecordOption = None,
        snapshots: SnapshotOption = None,
        async_mode: AsyncOption = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        debug: DebugOption = False,
        json_output: JsonOption = False,
    ) -> None:
        """Process a video file."""
        app_config = load(
            config,
            _overrides(log_level, device, debug, threshold, record, snapshots, async_mode),
        )
        engine = engine_for(app_config)
        try:
            _run_stream(
                engine,
                SourceKind.VIDEO,
                str(input_path),
                show=show,
                save_video=save_video,
                json_output=json_output,
            )
        finally:
            engine.close()

    @app.command("stream")
    @handle_errors
    def stream(
        url: Annotated[str, typer.Option("--input", "-i", help="RTSP/HTTP stream URL.")],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        show: ShowOption = None,
        save_video: SaveVideoOption = None,
        threshold: ThresholdOption = None,
        record: RecordOption = None,
        snapshots: SnapshotOption = None,
        async_mode: AsyncOption = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        debug: DebugOption = False,
        json_output: JsonOption = False,
    ) -> None:
        """Process a network video stream (RTSP / HTTP / IP camera)."""
        app_config = load(
            config,
            _overrides(log_level, device, debug, threshold, record, snapshots, async_mode),
        )
        engine = engine_for(app_config)
        try:
            _run_stream(
                engine,
                SourceKind.STREAM,
                url,
                show=show,
                save_video=save_video,
                json_output=json_output,
            )
        finally:
            engine.close()

    @app.command("image")
    @handle_errors
    def image(
        input_path: Annotated[Path, typer.Option("--input", "-i", help="Image file.")],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        threshold: ThresholdOption = None,
        snapshots: SnapshotOption = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        debug: DebugOption = False,
        json_output: JsonOption = False,
    ) -> None:
        """Process a single image."""
        app_config = load(config, _overrides(log_level, device, debug, threshold, None, snapshots))
        engine = engine_for(app_config)
        try:
            engine.ensure_gallery()
            pipeline = engine.pipeline(use_tracking=False)
            source = build_source(
                engine.config, engine.paths, kind=SourceKind.IMAGE, target=str(input_path)
            )
            summary = ImageRunner(engine, pipeline).run(source)
            emit(summary.to_dict(), json_output, _format_image_summary(summary))
        finally:
            engine.close()

    @app.command("images")
    @handle_errors
    def images(
        input_path: Annotated[
            Path, typer.Option("--input", "-i", help="Directory of images.")
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        recursive: Annotated[
            bool, typer.Option("--recursive", "-r", help="Descend into subdirectories.")
        ] = False,
        threshold: ThresholdOption = None,
        snapshots: SnapshotOption = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        debug: DebugOption = False,
        json_output: JsonOption = False,
    ) -> None:
        """Process every image in a directory."""
        overrides = _overrides(log_level, device, debug, threshold, None, snapshots)
        if recursive:
            overrides.setdefault("source", {})["recursive"] = True
        app_config = load(config, overrides)
        engine = engine_for(app_config)
        try:
            engine.ensure_gallery()
            pipeline = engine.pipeline(use_tracking=False)
            source = build_source(
                engine.config, engine.paths, kind=SourceKind.DIRECTORY, target=str(input_path)
            )
            summary = ImageRunner(engine, pipeline).run(source)
            emit(summary.to_dict(), json_output, _format_image_summary(summary))
        finally:
            engine.close()

    @app.command("run")
    @handle_errors
    def run(
        target: Annotated[
            str,
            typer.Argument(
                help="Camera index, video file, image file, directory or stream URL."
            ),
        ],
        config: ConfigOption = Path(DEFAULT_CONFIG),
        show: ShowOption = None,
        threshold: ThresholdOption = None,
        log_level: LogLevelOption = None,
        device: DeviceOption = None,
        debug: DebugOption = False,
        json_output: JsonOption = False,
    ) -> None:
        """Process any input, inferring the source kind from the argument."""
        app_config = load(config, _overrides(log_level, device, debug, threshold, None, None))
        engine = engine_for(app_config)
        try:
            kind = infer_kind(target, engine.config)
            if kind in (SourceKind.IMAGE, SourceKind.DIRECTORY):
                engine.ensure_gallery()
                pipeline = engine.pipeline(use_tracking=False)
                source = build_source(engine.config, engine.paths, kind=kind, target=target)
                summary = ImageRunner(engine, pipeline).run(source)
                emit(summary.to_dict(), json_output, _format_image_summary(summary))
            else:
                _run_stream(
                    engine, kind, target, show=show, save_video=None, json_output=json_output
                )
        finally:
            engine.close()
