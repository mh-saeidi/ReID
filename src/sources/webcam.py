"""Webcam / capture-device source."""

from __future__ import annotations

import time

import cv2

from src.core.exceptions import SourceError
from src.core.types import Frame, SourceInfo
from src.sources.base import BaseSource
from src.utils.logging import get_logger

logger = get_logger(__name__)

_AUTO_PROBE_LIMIT = 5


class WebcamSource(BaseSource):
    """Live capture device.

    Args:
        device: Device index, ``"auto"`` to probe, or a backend URL string.
        width/height/fps: Requested capture settings; the driver may ignore them.
        buffer_size: Driver buffer depth; 1 keeps latency low for live use.
    """

    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        buffer_size: int = 1,
    ) -> None:
        self._device = device
        self._requested = (width, height, fps)
        self._buffer_size = buffer_size
        self._capture: cv2.VideoCapture | None = None
        self._info: SourceInfo | None = None
        self._index = 0
        self._resolved_device: int | str = device

    @staticmethod
    def probe_devices(limit: int = _AUTO_PROBE_LIMIT) -> list[int]:
        """Return indices that could be opened -- used by ``--device auto``."""
        found: list[int] = []
        for index in range(limit):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    found.append(index)
            capture.release()
        return found

    def _resolve_device(self) -> int | str:
        if isinstance(self._device, str) and self._device.strip().lower() == "auto":
            devices = self.probe_devices()
            if not devices:
                raise SourceError(
                    "no usable camera found (probed indices 0-"
                    f"{_AUTO_PROBE_LIMIT - 1}). Connect a camera, grant the "
                    "application camera permission, or pass --device <index>."
                )
            logger.info("Auto-selected camera", extra={"device": devices[0], "found": devices})
            return devices[0]
        if isinstance(self._device, str) and self._device.isdigit():
            return int(self._device)
        return self._device

    def open(self) -> SourceInfo:
        if self._capture is not None and self._info is not None:
            return self._info

        self._resolved_device = self._resolve_device()
        capture = cv2.VideoCapture(self._resolved_device)
        if not capture.isOpened():
            raise SourceError(
                f"cannot open camera '{self._resolved_device}'. Check that the device "
                "exists, is not in use by another application, and that this program "
                "has camera permission (macOS: System Settings > Privacy & Security > Camera)."
            )

        width, height, fps = self._requested
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            capture.set(cv2.CAP_PROP_FPS, float(fps))
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self._buffer_size))
        except cv2.error:  # pragma: no cover - backend dependent
            logger.debug("Capture backend does not support CAP_PROP_BUFFERSIZE")

        ok, image = capture.read()
        if not ok or image is None:
            capture.release()
            raise SourceError(
                f"camera '{self._resolved_device}' opened but returned no frames; "
                "another application may be holding it"
            )

        actual_h, actual_w = image.shape[:2]
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        if actual_fps <= 0.0 or actual_fps > 480.0:
            actual_fps = float(fps) if fps else 30.0

        self._capture = capture
        self._info = SourceInfo(
            source_id=f"webcam_{self._resolved_device}",
            kind="webcam",
            width=actual_w,
            height=actual_h,
            fps=actual_fps,
            frame_count=None,
            is_stream=True,
        )
        if width and int(width) != actual_w:
            logger.warning(
                "Camera ignored the requested resolution",
                extra={"requested": f"{width}x{height}", "actual": f"{actual_w}x{actual_h}"},
            )
        logger.info(
            "Webcam opened",
            extra={
                "device": self._resolved_device,
                "size": f"{actual_w}x{actual_h}",
                "fps": round(actual_fps, 2),
            },
        )
        return self._info

    def read(self) -> Frame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None  # noqa: S101
        ok, image = self._capture.read()
        if not ok or image is None:
            logger.warning("Camera returned no frame; treating the stream as ended")
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

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.debug("Webcam closed", extra={"frames_read": self._index})
