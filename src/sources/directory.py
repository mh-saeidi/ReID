"""Directory-of-images source."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from src.core.exceptions import SourceError
from src.core.types import Frame, SourceInfo
from src.sources.base import BaseSource
from src.utils.image import DEFAULT_IMAGE_EXTENSIONS, imread, iter_images
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ImageDirectorySource(BaseSource):
    """Iterates the images in a directory in sorted (reproducible) order.

    Unreadable files are skipped with a warning rather than aborting the batch,
    because a single corrupt file should not lose a whole run.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
        recursive: bool = False,
    ) -> None:
        self._directory = Path(directory)
        self._extensions = tuple(extensions)
        self._recursive = recursive
        self._files: list[Path] = []
        self._cursor = 0
        self._index = 0
        self._info: SourceInfo | None = None

    @property
    def files(self) -> list[Path]:
        return list(self._files)

    def open(self) -> SourceInfo:
        if self._info is not None:
            return self._info
        self._files = iter_images(self._directory, self._extensions, recursive=self._recursive)
        if not self._files:
            raise SourceError(
                f"no images found in {self._directory} "
                f"(looking for {', '.join(self._extensions)}"
                f"{', recursively' if self._recursive else ''}). "
                "Check source.extensions or pass --recursive."
            )
        first = imread(self._files[0])
        height, width = first.shape[:2]
        self._info = SourceInfo(
            source_id=f"images:{self._directory.name}",
            kind="directory",
            width=width,
            height=height,
            fps=0.0,
            frame_count=len(self._files),
            is_stream=False,
        )
        logger.info(
            "Image directory opened",
            extra={"path": str(self._directory), "images": len(self._files)},
        )
        return self._info

    def read(self) -> Frame | None:
        if self._info is None:
            self.open()
        while self._cursor < len(self._files):
            path = self._files[self._cursor]
            self._cursor += 1
            try:
                image = imread(path)
            except SourceError as exc:
                logger.warning("Skipping unreadable image", extra={"path": str(path), "error": str(exc)})
                continue
            frame = Frame(
                image=image,
                index=self._index,
                timestamp=time.time(),
                source_id=self.info.source_id,
                path=str(path),
            )
            self._index += 1
            return frame
        return None

    @property
    def info(self) -> SourceInfo:
        if self._info is None:
            return self.open()
        return self._info

    def close(self) -> None:
        self._files = []
        self._cursor = 0
