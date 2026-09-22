"""Detection metadata persistence (JSON / JSONL)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.core.exceptions import OutputError
from src.core.types import FrameResult
from src.utils.image import sanitize_filename, timestamp_slug, unique_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetadataWriter:
    """Appends per-frame results to a JSONL file, one object per frame.

    JSONL because a long video produces thousands of records and a streaming
    format can be tailed live and truncated by retention without re-parsing.
    """

    def __init__(self, directory: Path, source_id: str, *, overwrite: bool = False) -> None:
        self._directory = directory
        self._source_id = source_id
        self._overwrite = overwrite
        self._path: Path | None = None
        self._file = None
        self._records = 0

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def records(self) -> int:
        return self._records

    def open(self) -> Path:
        if self._file is not None and self._path is not None:
            return self._path
        self._directory.mkdir(parents=True, exist_ok=True)
        name = f"{sanitize_filename(self._source_id)}_{timestamp_slug()}.jsonl"
        path = unique_path(self._directory / name, overwrite=self._overwrite)
        try:
            self._file = path.open("w", encoding="utf-8")
        except OSError as exc:
            raise OutputError(f"cannot open the metadata file {path}: {exc}") from exc
        self._path = path
        return path

    def write(self, result: FrameResult) -> None:
        if self._file is None:
            self.open()
        assert self._file is not None  # noqa: S101
        try:
            self._file.write(json.dumps(result.to_dict(), default=str) + "\n")
            self._records += 1
        except OSError as exc:  # pragma: no cover - disk-full territory
            logger.error("Cannot write frame metadata: %s", exc)

    def close(self) -> Path | None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
            logger.info(
                "Detection metadata written",
                extra={"path": str(self._path), "records": self._records},
            )
        return self._path

    def __enter__(self) -> MetadataWriter:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> Path:
    """Write one JSON document, never silently clobbering an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = unique_path(path, overwrite=overwrite)
    try:
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"cannot write {target}: {exc}") from exc
    return target


def summarize(results: Iterable[FrameResult]) -> dict[str, Any]:
    """Aggregate frame results into a run summary."""
    frames = list(results)
    detections = sum(len(f.detections) for f in frames)
    recognized = sum(len(f.recognized) for f in frames)
    identities: dict[str, int] = {}
    for frame in frames:
        for detection in frame.recognized:
            key = detection.identity_id or "unknown"
            identities[key] = identities.get(key, 0) + 1
    return {
        "frames": len(frames),
        "detections": detections,
        "recognized": recognized,
        "unknown": detections - recognized,
        "identities": dict(sorted(identities.items(), key=lambda kv: -kv[1])),
    }
