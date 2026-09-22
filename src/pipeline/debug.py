"""Debug artefact dumping (disabled by default).

``--debug`` writes the intermediate state that makes a bad recognition
explainable: the annotated frame, the exact crop the encoder saw and the full
similarity vector behind each decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config.schema import DebugConfig
from src.core.types import DetectionResult, FrameResult
from src.tracking.tracker import TrackState
from src.utils.image import crop_bbox, imwrite
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DebugRecorder:
    """Writes per-frame debug artefacts under ``output/debug``."""

    def __init__(self, config: DebugConfig, directory: Path) -> None:
        self._config = config
        self._directory = directory
        self._frames_written = 0
        if config.enabled:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info("Debug output enabled", extra={"directory": str(directory)})

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._frames_written < self._config.max_frames

    def record(
        self,
        image: np.ndarray,
        annotated: np.ndarray,
        result: FrameResult,
        tracks: dict[int, TrackState] | None = None,
    ) -> None:
        if not self.enabled:
            return
        stem = f"frame_{result.frame_index:06d}"

        if self._config.save_frames:
            imwrite(self._directory / f"{stem}.jpg", annotated)

        for detection in result.detections:
            if self._config.save_crops:
                crop = crop_bbox(image, detection.bbox)
                if crop is not None and crop.size:
                    imwrite(
                        self._directory / f"crop_track_{detection.track_id}_{stem}.jpg", crop
                    )
            if self._config.save_matches:
                self._write_match(stem, detection)

        if self._config.save_track_state and tracks:
            self._write_json(
                self._directory / f"tracks_{stem}.json",
                {str(k): v.to_dict() for k, v in tracks.items()},
            )
        self._frames_written += 1

    def _write_match(self, stem: str, detection: DetectionResult) -> None:
        payload = {
            **detection.to_dict(),
            "instantaneous": {
                "status": detection.recognition.status.value,
                "identity_id": detection.recognition.identity_id,
                "similarity": round(detection.recognition.similarity, 6),
                "runner_up_id": detection.recognition.runner_up_id,
                "runner_up_similarity": round(detection.recognition.runner_up_similarity, 6),
                "all_similarities": detection.recognition.all_similarities,
            },
        }
        self._write_json(
            self._directory / f"match_track_{detection.track_id}_{stem}.json", payload
        )

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        try:
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning("Cannot write the debug artefact %s: %s", path, exc)
