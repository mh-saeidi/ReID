"""Jetson camera capture through NVIDIA-accelerated GStreamer.

Three capture paths, chosen by what the board actually exposes:

``nvarguscamerasrc``
    CSI/MIPI cameras via the Argus stack. Frames stay in NVMM (NVIDIA's
    zero-copy memory) until ``nvvidconv`` hands them over, so scaling and colour
    conversion run on the VIC rather than the CPU.

``v4l2src`` + ``nvvidconv``
    USB/V4L2 cameras, with the format conversion still offloaded.

plain V4L2
    The existing :class:`~src.sources.webcam.WebcamSource` behaviour, used
    whenever GStreamer is unavailable. Nothing here is mandatory.

Two decisions matter for latency:

* the sensor is asked for its capture resolution once, and GStreamer scales to
  the delivered size in the same pass -- rather than handing Python a
  full-resolution frame that then gets resized again for inference;
* the sink is configured with ``drop=true max-buffers=1``, so when inference
  falls behind the camera discards frames instead of building a queue. A live
  view that is three seconds late is worse than one that skipped frames.
"""

from __future__ import annotations

import time

import cv2

from src.core.exceptions import SourceError
from src.core.types import Frame, SourceInfo
from src.hardware.capabilities import SystemCapabilities, detect_capabilities
from src.sources.base import BaseSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_csi_pipeline(
    *,
    sensor_id: int,
    capture_width: int,
    capture_height: int,
    output_width: int,
    output_height: int,
    fps: int,
    flip_method: int = 0,
) -> str:
    """CSI camera: Argus -> NVMM -> scale/convert -> BGR appsink."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={capture_width},height={capture_height},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw,width={output_width},height={output_height},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def build_v4l2_pipeline(
    *,
    device: str,
    output_width: int,
    output_height: int,
    fps: int,
) -> str:
    """USB/V4L2 camera with the conversion offloaded to nvvidconv."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={output_width},height={output_height},framerate={fps}/1 ! "
        "jpegdec ! nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class JetsonCameraSource(BaseSource):
    """Low-latency Jetson camera capture.

    Args:
        source_config: The ``source`` configuration section.
        capabilities: Detected platform capabilities; probed when omitted.
    """

    def __init__(self, source_config, capabilities: SystemCapabilities | None = None) -> None:
        self._config = source_config
        self._caps = capabilities or detect_capabilities()
        self._capture: cv2.VideoCapture | None = None
        self._info: SourceInfo | None = None
        self._index = 0
        self._pipeline = ""
        self._mode = ""

    # --------------------------------------------------------------- pipeline
    def _resolve_mode(self) -> str:
        config = self._config
        if config.gst_pipeline:
            return "manual"
        if config.csi is True:
            return "csi"
        if config.csi is False:
            return "v4l2"
        return "csi" if self._caps.has_csi_camera else "v4l2"

    def _build_pipeline(self) -> str:
        config = self._config
        if config.gst_pipeline:
            return config.gst_pipeline

        width = config.width or 1280
        height = config.height or 720
        fps = int(config.fps or 30)
        capture_width = config.capture_width or width
        capture_height = config.capture_height or height

        if self._mode == "csi":
            return build_csi_pipeline(
                sensor_id=config.sensor_id,
                capture_width=capture_width,
                capture_height=capture_height,
                output_width=width,
                output_height=height,
                fps=fps,
                flip_method=config.flip_method,
            )
        device = config.device
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            device = f"/dev/video{device}"
        return build_v4l2_pipeline(
            device=str(device), output_width=width, output_height=height, fps=fps
        )

    # ------------------------------------------------------------- lifecycle
    def open(self) -> SourceInfo:
        if self._capture is not None and self._info is not None:
            return self._info

        if not self._caps.gstreamer.opencv_gstreamer:
            raise SourceError(
                "source.type='jetson_camera' needs an OpenCV built with "
                "GStreamer support, which this installation does not have. "
                "On JetPack use the system OpenCV (python3-opencv), or set "
                "source.type='webcam' to use the portable V4L2 path."
            )

        self._mode = self._resolve_mode()
        if self._mode == "csi" and not self._caps.gstreamer.nvarguscamerasrc:
            raise SourceError(
                "a CSI camera was requested but nvarguscamerasrc is not "
                "available. Check the camera is connected and the Argus daemon "
                "is running, or set source.csi: false for a USB camera."
            )

        self._pipeline = self._build_pipeline()
        logger.debug("Jetson camera pipeline", extra={"pipeline": self._pipeline})

        capture = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            raise SourceError(
                f"cannot open the Jetson camera ({self._mode} mode). Verify the "
                "pipeline with:\n  gst-launch-1.0 "
                f"{self._pipeline.replace('appsink drop=true max-buffers=1 sync=false', 'fakesink')}"
            )

        ok, image = capture.read()
        if not ok or image is None:
            capture.release()
            raise SourceError(
                f"the Jetson camera opened but delivered no frames ({self._mode} mode)"
            )

        height, width = image.shape[:2]
        fps = float(self._config.fps or 30.0)
        self._capture = capture
        self._info = SourceInfo(
            source_id=f"jetson_camera:{self._mode}:{self._config.sensor_id}",
            kind="jetson_camera",
            width=width,
            height=height,
            fps=fps,
            frame_count=None,
            is_stream=True,
        )
        logger.info(
            "Jetson camera opened",
            extra={
                "mode": self._mode,
                "size": f"{width}x{height}",
                "fps": fps,
                "nvmm": self._caps.has_nvmm,
            },
        )
        return self._info

    def read(self) -> Frame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None  # noqa: S101
        ok, image = self._capture.read()
        if not ok or image is None:
            logger.warning("Jetson camera returned no frame; treating the stream as ended")
            return None
        frame = Frame(
            image=image,
            index=self._index,
            timestamp=time.time(),
            source_id=self.info.source_id,
        )
        self._index += 1
        return frame

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            return self.open()
        return self._info

    @property
    def pipeline(self) -> str:
        """The GStreamer pipeline in use (diagnostics)."""
        return self._pipeline

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.debug("Jetson camera closed", extra={"frames_read": self._index})
