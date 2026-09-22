"""Network stream source (RTSP / HTTP / IP camera).

Present so network inputs are wired end-to-end today; it shares OpenCV's
capture path with the webcam source and adds reconnection, which is what
actually differs about a remote stream.
"""

from __future__ import annotations

import contextlib
import time

import cv2

from src.core.exceptions import SourceError
from src.core.types import Frame, SourceInfo
from src.sources.base import BaseSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NetworkStreamSource(BaseSource):
    """RTSP/HTTP stream with bounded automatic reconnection.

    Args:
        url: Stream URL (``rtsp://``, ``http://``, ``https://``).
        reconnect_attempts: Reconnect tries before the source reports the end.
        reconnect_delay: Seconds between attempts.
    """

    def __init__(
        self,
        url: str,
        *,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0,
        buffer_size: int = 1,
    ) -> None:
        self._url = url
        self._reconnect_attempts = max(0, reconnect_attempts)
        self._reconnect_delay = max(0.0, reconnect_delay)
        self._buffer_size = buffer_size
        self._capture: cv2.VideoCapture | None = None
        self._info: SourceInfo | None = None
        self._index = 0

    def _connect(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._url)
        if not capture.isOpened():
            raise SourceError(
                f"cannot open stream '{self._url}'. Verify the URL, credentials and "
                "that OpenCV was built with the required protocol support."
            )
        with contextlib.suppress(cv2.error):  # not every backend supports it
            capture.set(cv2.CAP_PROP_BUFFERSIZE, int(self._buffer_size))
        return capture

    def open(self) -> SourceInfo:
        if self._capture is not None and self._info is not None:
            return self._info
        capture = self._connect()
        ok, image = capture.read()
        if not ok or image is None:
            capture.release()
            raise SourceError(f"stream '{self._url}' opened but delivered no frames")
        height, width = image.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        if fps <= 0.0 or fps > 480.0:
            fps = 25.0
        self._capture = capture
        self._info = SourceInfo(
            source_id=f"stream:{self._url}",
            kind="stream",
            width=width,
            height=height,
            fps=fps,
            frame_count=None,
            is_stream=True,
        )
        logger.info(
            "Network stream opened",
            extra={"url": self._url, "size": f"{width}x{height}", "fps": round(fps, 2)},
        )
        return self._info

    def read(self) -> Frame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None  # noqa: S101
        ok, image = self._capture.read()
        if not ok or image is None:
            if not self._reconnect():
                return None
            assert self._capture is not None  # noqa: S101
            ok, image = self._capture.read()
            if not ok or image is None:
                return None
        frame = Frame(
            image=image, index=self._index, timestamp=time.time(), source_id=self.info.source_id
        )
        self._index += 1
        return frame

    def _reconnect(self) -> bool:
        for attempt in range(1, self._reconnect_attempts + 1):
            logger.warning(
                "Stream dropped; reconnecting",
                extra={"url": self._url, "attempt": attempt, "of": self._reconnect_attempts},
            )
            self.close()
            time.sleep(self._reconnect_delay)
            try:
                self._capture = self._connect()
                return True
            except SourceError:
                continue
        logger.error("Stream reconnection failed", extra={"url": self._url})
        return False

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            return self.open()
        return self._info

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
