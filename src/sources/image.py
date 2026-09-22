"""Single-image source."""

from __future__ import annotations

import time
from pathlib import Path

from src.core.types import Frame, SourceInfo
from src.sources.base import BaseSource
from src.utils.image import imread
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ImageSource(BaseSource):
    """Yields exactly one frame, so a still image runs through the same engine."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._info: SourceInfo | None = None
        self._consumed = False
        self._image = None

    def open(self) -> SourceInfo:
        if self._info is not None:
            return self._info
        self._image = imread(self._path)
        height, width = self._image.shape[:2]
        self._info = SourceInfo(
            source_id=f"image:{self._path.name}",
            kind="image",
            width=width,
            height=height,
            fps=0.0,
            frame_count=1,
            is_stream=False,
        )
        logger.debug(
            "Image source opened", extra={"path": str(self._path), "size": f"{width}x{height}"}
        )
        return self._info

    def read(self) -> Frame | None:
        if self._info is None:
            self.open()
        if self._consumed:
            return None
        self._consumed = True
        return Frame(
            image=self._image,
            index=0,
            timestamp=time.time(),
            source_id=self.info.source_id,
            path=str(self._path),
        )

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            return self.open()
        return self._info

    def close(self) -> None:
        self._image = None

    def reset(self) -> None:
        """Allow the single frame to be read again."""
        self._consumed = False
