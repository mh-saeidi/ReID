"""Input source abstraction.

The processing engine never learns where a frame came from. Adding an RTSP or
HTTP stream later means adding a class here, not touching the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.core.types import Frame, SourceInfo


class BaseSource(ABC):
    """A sequence of frames with uniform metadata and lifecycle."""

    @abstractmethod
    def open(self) -> SourceInfo:
        """Acquire the underlying resource. Idempotent."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release the resource. Safe to call multiple times."""

    @property
    @abstractmethod
    def info(self) -> SourceInfo:
        """Static metadata; valid after :meth:`open`."""

    # -- convenience accessors required by the source contract ---------------
    def fps(self) -> float:
        return self.info.fps

    def width(self) -> int:
        return self.info.width

    def height(self) -> int:
        return self.info.height

    def source_name(self) -> str:
        return self.info.source_id

    def frame_count(self) -> int | None:
        return self.info.frame_count

    @property
    def is_stream(self) -> bool:
        """True for unbounded live sources (webcam, RTSP): no total length."""
        return self.info.is_stream

    def __iter__(self) -> Iterator[Frame]:
        self.open()
        try:
            while True:
                frame = self.read()
                if frame is None:
                    return
                yield frame
        finally:
            self.close()

    def __enter__(self) -> BaseSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
