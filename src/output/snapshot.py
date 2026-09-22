"""Snapshot (still image) capture with sidecar metadata.

Snapshots are date-partitioned so retention cleanup is a directory operation,
and every image gets a JSON sidecar describing exactly what was recognised.
A per-track cooldown keeps a live stream from filling the disk.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.schema import OutputConfig, SnapshotMode
from src.core.exceptions import OutputError
from src.core.types import DetectionResult, FrameResult, RecognitionStatus
from src.utils.image import crop_bbox, imwrite, sanitize_filename, unique_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """A written snapshot and its sidecar."""

    image_path: Path
    metadata_path: Path | None
    crop_path: Path | None
    detection: DetectionResult


class SnapshotWriter:
    """Decides what to save, where, and how often."""

    def __init__(
        self,
        config: OutputConfig,
        snapshot_dir: Path,
        crops_dir: Path,
        *,
        clock=_dt.datetime.now,
    ) -> None:
        self._config = config
        self._snapshot_dir = snapshot_dir
        self._crops_dir = crops_dir
        self._clock = clock
        self._last_saved: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._config.save_snapshots and self._config.snapshot_mode is not SnapshotMode.DISABLED

    # ------------------------------------------------------------- selection
    def should_save(self, detection: DetectionResult, *, is_event: bool = False) -> bool:
        """Apply the configured snapshot mode to one detection."""
        if not self.enabled:
            return False
        mode = self._config.snapshot_mode
        recognized = detection.recognition_status in (
            RecognitionStatus.RECOGNIZED,
            RecognitionStatus.LOW_CONFIDENCE,
        )
        if mode is SnapshotMode.ALL:
            return True
        if mode is SnapshotMode.RECOGNIZED:
            return recognized
        if mode is SnapshotMode.UNKNOWN:
            return not recognized
        if mode is SnapshotMode.EVENTS_ONLY:
            return is_event
        return False

    def _cooldown_key(self, detection: DetectionResult) -> str:
        if detection.track_id is not None and detection.track_id >= 0:
            return f"track:{detection.track_id}"
        return f"identity:{detection.identity_id or 'unknown'}"

    def _within_cooldown(self, detection: DetectionResult, now: float) -> bool:
        cooldown = self._config.snapshot_cooldown_seconds
        if cooldown <= 0.0:
            return False
        key = self._cooldown_key(detection)
        last = self._last_saved.get(key)
        return last is not None and (now - last) < cooldown

    # --------------------------------------------------------------- writing
    def save(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        frame: FrameResult,
        *,
        annotated: np.ndarray | None = None,
        is_event: bool = False,
        force: bool = False,
    ) -> SnapshotRecord | None:
        """Write a snapshot for one detection, honouring mode and cooldown."""
        if not force and not self.should_save(detection, is_event=is_event):
            return None
        if not force and self._within_cooldown(detection, detection.timestamp):
            return None

        moment = self._clock()
        day_dir = self._snapshot_dir / moment.strftime("%Y-%m-%d")
        label = sanitize_filename(detection.identity_id or "unknown")
        stem = (
            f"{moment.strftime('%H-%M-%S')}_{label}_"
            f"{detection.reid_similarity:.2f}"
        )
        target = unique_path(day_dir / f"{stem}.jpg", overwrite=self._config.overwrite)

        source_image = annotated if annotated is not None else image
        try:
            image_path = imwrite(target, source_image, jpeg_quality=self._config.jpeg_quality)
        except OutputError as exc:
            logger.error("Snapshot not saved", extra={"error": str(exc)})
            return None

        crop_path = self._save_crop(image, detection, target)
        metadata_path = self._save_metadata(image_path, detection, frame, crop_path)
        self._last_saved[self._cooldown_key(detection)] = detection.timestamp

        logger.debug(
            "Snapshot saved",
            extra={"path": str(image_path), "identity": detection.identity_id or "unknown"},
        )
        return SnapshotRecord(image_path, metadata_path, crop_path, detection)

    def _save_crop(
        self, image: np.ndarray, detection: DetectionResult, snapshot_path: Path
    ) -> Path | None:
        if not self._config.save_crops:
            return None
        crop = crop_bbox(image, detection.bbox)
        if crop is None:
            return None
        target = unique_path(
            self._crops_dir / snapshot_path.parent.name / f"{snapshot_path.stem}_crop.jpg",
            overwrite=self._config.overwrite,
        )
        try:
            return imwrite(target, crop, jpeg_quality=self._config.jpeg_quality)
        except OutputError as exc:
            logger.warning("Crop not saved: %s", exc)
            return None

    def _save_metadata(
        self,
        image_path: Path,
        detection: DetectionResult,
        frame: FrameResult,
        crop_path: Path | None,
    ) -> Path | None:
        if not self._config.save_metadata:
            return None
        payload = {
            "source": frame.source_id,
            "timestamp": detection.timestamp,
            "iso_time": _dt.datetime.fromtimestamp(detection.timestamp).isoformat(
                timespec="seconds"
            ),
            "frame_index": frame.frame_index,
            "image": str(image_path),
            "crop": str(crop_path) if crop_path else None,
            **detection.to_dict(),
        }
        target = image_path.with_suffix(".json")
        try:
            target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("Snapshot metadata not saved: %s", exc)
            return None
        return target
