"""Input source construction from configuration and CLI arguments."""

from __future__ import annotations

from pathlib import Path

from src.config.paths import ProjectPaths
from src.config.schema import AppConfig, SourceKind
from src.core.exceptions import ConfigurationError
from src.sources.base import BaseSource
from src.sources.directory import ImageDirectorySource
from src.sources.image import ImageSource
from src.sources.stream import NetworkStreamSource
from src.sources.video import VideoFileSource
from src.sources.webcam import WebcamSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_source(
    config: AppConfig,
    paths: ProjectPaths,
    *,
    kind: SourceKind | None = None,
    target: str | int | None = None,
) -> BaseSource:
    """Create the configured input source.

    Args:
        kind: Overrides ``source.type`` (set by the CLI subcommand).
        target: Overrides ``source.path`` / ``source.device``.
    """
    source = config.source
    resolved_kind = kind or source.type

    if resolved_kind is SourceKind.WEBCAM:
        device = target if target is not None else source.device
        return WebcamSource(
            device,
            width=source.width,
            height=source.height,
            fps=source.fps,
            buffer_size=source.buffer_size,
        )

    if resolved_kind is SourceKind.JETSON_CAMERA:
        from src.hardware.capabilities import detect_capabilities  # noqa: PLC0415
        from src.sources.jetson_camera import JetsonCameraSource  # noqa: PLC0415

        caps = detect_capabilities()
        if not caps.gstreamer.opencv_gstreamer:
            # Portability first: a configuration written for a Jetson must still
            # run on a laptop, just without the accelerated capture path.
            logger.warning(
                "source.type='jetson_camera' requested but GStreamer is "
                "unavailable in this OpenCV build; using the portable V4L2 "
                "webcam source instead"
            )
            return WebcamSource(
                target if target is not None else source.device,
                width=source.width,
                height=source.height,
                fps=source.fps,
                buffer_size=source.buffer_size,
            )
        overrides = source
        if target is not None:
            overrides = source.model_copy(update={"device": target})
        return JetsonCameraSource(overrides, caps)

    if resolved_kind is SourceKind.STREAM:
        url = str(target if target is not None else (source.path or source.device))
        if not url:
            raise ConfigurationError("a stream source needs source.path or --input <url>")
        return NetworkStreamSource(url, buffer_size=source.buffer_size)

    raw = target if target is not None else source.path
    if raw is None:
        raise ConfigurationError(
            f"source.type '{resolved_kind.value}' requires source.path "
            "(or --input on the command line)"
        )
    path = paths.resolve(str(raw))

    if resolved_kind is SourceKind.VIDEO:
        return VideoFileSource(
            path, stride=config.performance.frame_stride, loop=source.loop
        )
    if resolved_kind is SourceKind.IMAGE:
        return ImageSource(path)
    if resolved_kind is SourceKind.DIRECTORY:
        return ImageDirectorySource(
            path, extensions=source.extensions, recursive=source.recursive
        )
    raise ConfigurationError(  # pragma: no cover - enum is exhaustive
        f"unsupported source.type '{resolved_kind}'"
    )


def infer_kind(target: str | Path, config: AppConfig) -> SourceKind:
    """Guess the source kind from a path (used by ``main.py run``)."""
    text = str(target)
    if text.startswith(("rtsp://", "http://", "https://", "rtmp://")):
        return SourceKind.STREAM
    if text.isdigit():
        return SourceKind.WEBCAM
    path = Path(text)
    if path.is_dir():
        return SourceKind.DIRECTORY
    suffix = path.suffix.lower()
    if suffix in set(config.source.extensions):
        return SourceKind.IMAGE
    return SourceKind.VIDEO
