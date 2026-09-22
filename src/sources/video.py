"""Video file source."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.core.exceptions import SourceError
from src.core.types import Frame, SourceInfo
from src.sources.base import BaseSource
from src.utils.image import DEFAULT_VIDEO_EXTENSIONS
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VideoFileSource(BaseSource):
    """Reads frames from a video file.

    Args:
        path: Video file path.
        stride: Process every Nth frame (1 = all frames).
        loop: Restart from the beginning when the file ends.
    """

    def __init__(self, path: str | Path, *, stride: int = 1, loop: bool = False) -> None:
        self._path = Path(path)
        self._stride = max(1, stride)
        self._loop = loop
        self._capture: cv2.VideoCapture | None = None
        self._info: SourceInfo | None = None
        self._index = 0
        self._raw_index = 0

    def open(self) -> SourceInfo:
        if self._capture is not None and self._info is not None:
            return self._info
        if not self._path.exists():
            raise SourceError(
                f"video file not found: {self._path} "
                f"(supported extensions: {', '.join(DEFAULT_VIDEO_EXTENSIONS)})"
            )
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise SourceError(
                f"cannot open video {self._path}: the file may be corrupt or use a "
                "codec OpenCV was not built with"
            )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0.0 or fps > 480.0:
            logger.warning(
                "Video reports an implausible FPS (%s); assuming 30.0 for timing", fps
            )
            fps = 30.0
        if width <= 0 or height <= 0:
            capture.release()
            raise SourceError(f"video {self._path} reports invalid dimensions {width}x{height}")

        self._capture = capture
        self._info = SourceInfo(
            source_id=f"video:{self._path.name}",
            kind="video",
            width=width,
            height=height,
            fps=fps,
            frame_count=(total // self._stride) if total > 0 else None,
            is_stream=False,
        )
        logger.info(
            "Video source opened",
            extra={
                "path": str(self._path),
                "size": f"{width}x{height}",
                "fps": round(fps, 2),
                "frames": total if total > 0 else "unknown",
                "stride": self._stride,
            },
        )
        return self._info

    def read(self) -> Frame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None  # noqa: S101
        while True:
            ok, image = self._capture.read()
            if not ok:
                if self._loop and self._raw_index > 0:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._raw_index = 0
                    continue
                return None
            raw_index = self._raw_index
            self._raw_index += 1
            if raw_index % self._stride:
                continue
            frame = Frame(
                image=image,
                index=self._index,
                timestamp=time.time(),
                source_id=self.info.source_id,
                path=str(self._path),
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
            logger.debug("Video source closed", extra={"frames_read": self._index})
