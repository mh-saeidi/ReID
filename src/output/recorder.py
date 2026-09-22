"""Annotated video recording: continuous and event-triggered.

Event recording uses a ring buffer of recent frames, so a clip genuinely starts
``pre_event_seconds`` *before* the person appeared rather than at the moment of
detection. Without the buffer the interesting approach is always missing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src.config.schema import OutputConfig, RecordingConfig, RecordingMode
from src.core.exceptions import OutputError
from src.utils.image import sanitize_filename, timestamp_slug, unique_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RecorderState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRAILING = "trailing"
    """Still writing the post-event tail after the trigger disappeared."""


@dataclass(frozen=True, slots=True)
class ClipInfo:
    path: Path
    frames: int
    fps: float
    trigger: str


class VideoRecorder:
    """Writes annotated video according to the configured recording mode.

    Args:
        recording: Recording behaviour (mode, pre/post roll, triggers).
        output: Output settings (codec, directory policy, FPS override).
        directory: Where clips are written.
        fps: Source frame rate, used for buffer sizing and playback speed.
        size: ``(width, height)`` of the frames that will be written.
        source_id: Used in the generated filenames.
    """

    def __init__(
        self,
        recording: RecordingConfig,
        output: OutputConfig,
        directory: Path,
        *,
        fps: float,
        size: tuple[int, int],
        source_id: str = "source",
        writer_factory=None,
    ) -> None:
        self._config = recording
        self._output = output
        self._directory = directory
        # Injected so a hardware encoder can replace the software one without
        # touching the event-clip or ring-buffer logic.
        self._writer_factory = writer_factory
        self._fps = float(output.video_fps or fps or 25.0)
        self._size = size
        self._source_id = source_id

        self._writer: cv2.VideoWriter | None = None
        self._state = RecorderState.IDLE
        self._frames_written = 0
        self._trailing_frames = 0
        self._current_path: Path | None = None
        self._current_trigger = ""
        self._clips: list[ClipInfo] = []

        buffer_frames = int(round(self._fps * recording.pre_event_seconds))
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, buffer_frames))
        self._post_frames = int(round(self._fps * recording.post_event_seconds))
        self._max_frames = int(round(self._fps * recording.max_clip_seconds))

    # ------------------------------------------------------------- properties
    @property
    def mode(self) -> RecordingMode:
        return self._config.mode

    @property
    def enabled(self) -> bool:
        return self._config.mode is not RecordingMode.DISABLED

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    @property
    def clips(self) -> list[ClipInfo]:
        return list(self._clips)

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    # ------------------------------------------------------------- lifecycle
    def _open_writer(self, trigger: str) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        prefix = sanitize_filename(self._config.filename_prefix or "clip")
        name = f"{prefix}_{sanitize_filename(self._source_id)}_{timestamp_slug()}"
        if trigger:
            name += f"_{sanitize_filename(trigger)}"
        path = unique_path(self._directory / f"{name}.mp4", overwrite=self._output.overwrite)

        if self._writer_factory is not None:
            writer = self._writer_factory(path, self._fps, self._size)
        else:
            fourcc = cv2.VideoWriter_fourcc(*self._output.codec)
            writer = cv2.VideoWriter(str(path), fourcc, self._fps, self._size)
        if not writer.isOpened():
            raise OutputError(
                f"cannot open the video writer for {path} with codec "
                f"'{self._output.codec}' at {self._size[0]}x{self._size[1]}. "
                "Try output.codec: 'mp4v' or 'avc1'."
            )
        self._writer = writer
        self._current_path = path
        self._current_trigger = trigger
        self._frames_written = 0
        logger.info(
            "Recording started",
            extra={"path": str(path), "fps": round(self._fps, 2), "trigger": trigger or "-"},
        )
        return path

    def start(self, trigger: str = "") -> Path | None:
        """Begin a clip, flushing the pre-event ring buffer into it."""
        if not self.enabled or self._writer is not None:
            return self._current_path
        path = self._open_writer(trigger)
        if self._config.mode is RecordingMode.EVENT:
            for buffered in self._ring:
                self._write_raw(buffered)
            self._ring.clear()
        self._state = RecorderState.RECORDING
        return path

    def stop(self) -> ClipInfo | None:
        """Close the current clip."""
        if self._writer is None:
            self._state = RecorderState.IDLE
            return None
        self._writer.release()
        clip = ClipInfo(
            path=self._current_path or Path(),
            frames=self._frames_written,
            fps=self._fps,
            trigger=self._current_trigger,
        )
        self._clips.append(clip)
        logger.info(
            "Recording stopped",
            extra={"path": str(clip.path), "frames": clip.frames},
        )
        self._writer = None
        self._current_path = None
        self._state = RecorderState.IDLE
        self._trailing_frames = 0
        return clip

    # ----------------------------------------------------------------- frames
    def _write_raw(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        self._writer.write(frame)
        self._frames_written += 1

    def process(self, frame: np.ndarray, *, trigger: bool = False, reason: str = "") -> None:
        """Feed one annotated frame to the recorder.

        Args:
            frame: The annotated frame to write.
            trigger: Whether this frame satisfies the recording trigger
                (a recognised identity, or an unknown person when configured).
            reason: Short label used in the clip filename.
        """
        if not self.enabled:
            return

        if self._config.mode is RecordingMode.CONTINUOUS:
            if self._writer is None:
                self.start(reason)
            self._write_raw(frame)
            if self._max_frames and self._frames_written >= self._max_frames:
                # Roll over to a new file instead of producing one huge clip.
                self.stop()
            return

        # Event mode.
        if trigger:
            if self._writer is None:
                self.start(reason)
            self._state = RecorderState.RECORDING
            self._trailing_frames = 0
            self._write_raw(frame)
        elif self._writer is not None:
            self._state = RecorderState.TRAILING
            self._trailing_frames += 1
            self._write_raw(frame)
            if self._trailing_frames >= self._post_frames:
                self.stop()
        else:
            self._ring.append(frame.copy())

        if self._writer is not None and self._max_frames and self._frames_written >= self._max_frames:
            logger.warning(
                "Clip reached recording.max_clip_seconds; closing it",
                extra={"path": str(self._current_path)},
            )
            self.stop()

    def close(self) -> ClipInfo | None:
        """Flush and release everything (call when the source ends)."""
        clip = self.stop()
        self._ring.clear()
        return clip

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AnnotatedVideoWriter:
    """Plain annotated-video writer for ``output.save_video``.

    Separate from :class:`VideoRecorder`, which implements the *event* policy:
    this one simply mirrors the processed stream to a file.
    """

    def __init__(
        self,
        output: OutputConfig,
        directory: Path,
        *,
        fps: float,
        size: tuple[int, int],
        source_id: str,
        writer_factory=None,
    ) -> None:
        self._output = output
        self._directory = directory
        self._writer_factory = writer_factory
        self._fps = float(output.video_fps or fps or 25.0)
        self._size = size
        self._source_id = source_id
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._frames = 0

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def frames(self) -> int:
        return self._frames

    def open(self) -> Path:
        if self._writer is not None and self._path is not None:
            return self._path
        self._directory.mkdir(parents=True, exist_ok=True)
        name = f"{sanitize_filename(self._source_id)}_{timestamp_slug()}.mp4"
        path = unique_path(self._directory / name, overwrite=self._output.overwrite)
        if self._writer_factory is not None:
            writer = self._writer_factory(path, self._fps, self._size)
        else:
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*self._output.codec),
                self._fps, self._size,
            )
        if not writer.isOpened():
            raise OutputError(
                f"cannot open the video writer for {path} with codec "
                f"'{self._output.codec}'. Try output.codec: 'mp4v' or 'avc1'."
            )
        self._writer = writer
        self._path = path
        logger.info(
            "Annotated video opened",
            extra={"path": str(path), "fps": round(self._fps, 2),
                   "size": f"{self._size[0]}x{self._size[1]}"},
        )
        return path

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.open()
        assert self._writer is not None  # noqa: S101
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        self._writer.write(frame)
        self._frames += 1

    def close(self) -> Path | None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "Annotated video written",
                extra={"path": str(self._path), "frames": self._frames},
            )
        return self._path
