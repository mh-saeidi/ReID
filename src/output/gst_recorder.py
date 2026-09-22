"""Hardware-accelerated video writing via GStreamer.

On a Jetson the video encoders are fixed-function blocks (NVENC): using them
frees the GPU and the CPU for inference, which is the whole point on a board
where both are scarce. OpenCV can drive them through a GStreamer appsrc
pipeline when it was built with GStreamer support.

Falling back is mandatory, not optional. A missing element, an OpenCV built
without GStreamer, or a non-NVIDIA host all produce the existing software
writer, because losing the recording entirely is a worse failure than encoding
it on the CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config.schema import AppConfig, RecordingBackend, VideoCodec
from src.hardware.capabilities import SystemCapabilities, detect_capabilities
from src.utils.logging import get_logger

logger = get_logger(__name__)

# OpenCV writes BGR; nvvidconv wants NVMM-resident I420 for the encoder.
_H264_ELEMENT = "nvv4l2h264enc"
_H265_ELEMENT = "nvv4l2h265enc"


@dataclass(frozen=True, slots=True)
class WriterPlan:
    """Which writer will be used, and why."""

    backend: RecordingBackend
    codec: str
    hardware: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "codec": self.codec,
            "hardware": self.hardware,
            "reason": self.reason,
        }


def plan_writer(config: AppConfig, caps: SystemCapabilities | None = None) -> WriterPlan:
    """Decide between the hardware and software encoder."""
    caps = caps or detect_capabilities()
    recording = config.recording
    requested = recording.backend

    if requested is RecordingBackend.OPENCV:
        return WriterPlan(
            RecordingBackend.OPENCV, config.output.codec, False, "explicitly configured"
        )

    wants_hardware = (
        requested is RecordingBackend.GSTREAMER
        or (requested is RecordingBackend.AUTO and recording.hardware_acceleration)
    )
    if not wants_hardware:
        return WriterPlan(
            RecordingBackend.OPENCV, config.output.codec, False,
            "hardware acceleration not requested",
        )

    if not caps.gstreamer.opencv_gstreamer:
        reason = "this OpenCV build has no GStreamer support"
    elif recording.codec is VideoCodec.H265 and not caps.gstreamer.nvv4l2h265enc:
        reason = "nvv4l2h265enc is not available"
    elif recording.codec is VideoCodec.H264 and not caps.gstreamer.nvv4l2h264enc:
        reason = "nvv4l2h264enc is not available"
    elif recording.codec is VideoCodec.MP4V:
        reason = "mp4v has no hardware encoder; use h264 or h265"
    else:
        return WriterPlan(
            RecordingBackend.GSTREAMER, recording.codec.value, True,
            "NVIDIA hardware encoder available",
        )

    if requested is RecordingBackend.GSTREAMER:
        logger.warning(
            "Hardware recording was requested but is unavailable (%s); "
            "falling back to the software encoder",
            reason,
        )
    return WriterPlan(RecordingBackend.OPENCV, config.output.codec, False, reason)


def build_gst_pipeline(path: Path, *, fps: float, size: tuple[int, int],
                       codec: str, bitrate_kbps: int) -> str:
    """appsrc -> NVMM -> hardware encoder -> mp4 container."""
    width, height = size
    encoder = _H265_ELEMENT if codec == "h265" else _H264_ELEMENT
    parser = "h265parse" if codec == "h265" else "h264parse"
    return (
        "appsrc is-live=true do-timestamp=true ! "
        "video/x-raw,format=BGR,"
        f"width={width},height={height},framerate={int(round(fps))}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
        f"{encoder} bitrate={bitrate_kbps * 1000} insert-sps-pps=true ! "
        f"{parser} ! qtmux ! "
        f"filesink location={path} sync=false"
    )


class GStreamerVideoWriter:
    """Minimal ``cv2.VideoWriter``-compatible wrapper around a GStreamer sink."""

    def __init__(self, path: Path, *, fps: float, size: tuple[int, int],
                 codec: str, bitrate_kbps: int) -> None:
        self._path = Path(path)
        self._size = size
        pipeline = build_gst_pipeline(
            self._path, fps=fps, size=size, codec=codec, bitrate_kbps=bitrate_kbps
        )
        logger.debug("GStreamer writer pipeline", extra={"pipeline": pipeline})
        self._writer = cv2.VideoWriter(
            pipeline, cv2.CAP_GSTREAMER, 0, float(fps), size, True
        )

    def isOpened(self) -> bool:  # noqa: N802 - matches the cv2 API
        return bool(self._writer.isOpened())

    def write(self, frame) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


VideoWriterFactory = Callable[[Path, float, tuple[int, int]], object]


def make_video_writer_factory(
    config: AppConfig, caps: SystemCapabilities | None = None
) -> VideoWriterFactory:
    """Return a factory producing the best available writer.

    The factory is handed to the existing recorder classes, so event clips,
    pre/post-roll ring buffering and continuous recording all keep their current
    behaviour -- only the encoder underneath changes.
    """
    plan = plan_writer(config, caps)
    if plan.hardware:
        logger.info(
            "Recording will use the NVIDIA hardware encoder",
            extra={"codec": plan.codec, "bitrate_kbps": config.recording.bitrate_kbps},
        )
    else:
        logger.debug("Recording uses the software encoder: %s", plan.reason)

    def factory(path: Path, fps: float, size: tuple[int, int]):
        if plan.hardware:
            writer = GStreamerVideoWriter(
                path,
                fps=fps,
                size=size,
                codec=plan.codec,
                bitrate_kbps=config.recording.bitrate_kbps,
            )
            if writer.isOpened():
                return writer
            logger.warning(
                "The GStreamer writer could not be opened for %s; "
                "falling back to the software encoder",
                path.name,
            )
        return cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*config.output.codec),
            float(fps),
            size,
        )

    factory.plan = plan  # type: ignore[attr-defined]
    return factory
