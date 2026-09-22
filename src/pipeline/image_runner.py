"""Still-image processing.

Images and image directories need different outputs from a video stream: one
annotated file per input, optional per-person crops, and a JSON document per
image rather than a rolling JSONL. Tracking is not used -- there is no temporal
continuity between unrelated photos -- so every person in every image is matched
against the gallery independently. That is what makes an image containing
"John, Jane, a stranger and John again" resolve correctly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.schema import RecognitionMode
from src.core.types import Frame, FrameResult
from src.events.types import Event, EventType
from src.output.metadata import write_json
from src.output.renderer import Renderer
from src.output.snapshot import SnapshotWriter
from src.pipeline.engine import Engine
from src.pipeline.processor import ReIDPipeline
from src.sources.base import BaseSource
from src.utils.image import crop_bbox, imwrite, sanitize_filename, unique_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ImageOutcome:
    """Everything produced for one input image."""

    source_path: str
    result: FrameResult
    annotated_path: str | None = None
    metadata_path: str | None = None
    crop_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.source_path,
            "annotated": self.annotated_path,
            "metadata": self.metadata_path,
            "crops": self.crop_paths,
            **self.result.to_dict(),
        }


@dataclass(slots=True)
class ImageBatchSummary:
    """Aggregate result for an image or a directory of images."""

    images: int
    detections: int
    recognized: int
    unknown: int
    elapsed_s: float
    outcomes: list[ImageOutcome] = field(default_factory=list)
    identities: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "detections": self.detections,
            "recognized": self.recognized,
            "unknown": self.unknown,
            "elapsed_s": round(self.elapsed_s, 3),
            "identities": self.identities,
            "failures": self.failures,
            "results": [o.to_dict() for o in self.outcomes],
        }


class ImageRunner:
    """Processes still images through the pipeline and writes image outputs."""

    def __init__(self, engine: Engine, pipeline: ReIDPipeline, *, save: bool = True) -> None:
        self._engine = engine
        self._pipeline = pipeline
        self._config = engine.config
        self._renderer = Renderer(
            engine.config.display,
            score_label=(
                "Face"
                if engine.config.recognition.mode is RecognitionMode.FACE
                else "ReID"
            ),
        )
        self._save = save
        self._snapshots = SnapshotWriter(
            engine.config.output, engine.paths.snapshots_dir, engine.paths.crops_dir
        )

    def run(self, source: BaseSource) -> ImageBatchSummary:
        """Process every image the source yields."""
        info = source.open()
        self._pipeline.reset()
        started = time.perf_counter()

        outcomes: list[ImageOutcome] = []
        identities: dict[str, int] = {}
        failures: dict[str, str] = {}
        detections = recognized = 0

        self._engine.events.emit(
            Event(type=EventType.SOURCE_STARTED, source_id=info.source_id,
                  payload={"kind": info.kind, "images": info.frame_count})
        )
        try:
            while True:
                frame = source.read()
                if frame is None:
                    break
                try:
                    outcome = self._process_one(frame)
                except Exception as exc:  # noqa: BLE001 - one bad file must not kill a batch
                    failures[frame.path or f"frame_{frame.index}"] = str(exc)
                    logger.error(
                        "Image processing failed",
                        extra={"path": frame.path, "error": str(exc)},
                    )
                    continue
                outcomes.append(outcome)
                detections += len(outcome.result.detections)
                recognized += len(outcome.result.recognized)
                for detection in outcome.result.recognized:
                    key = detection.identity_id or "unknown"
                    identities[key] = identities.get(key, 0) + 1
        finally:
            source.close()
            self._engine.events.emit(
                Event(type=EventType.SOURCE_ENDED, source_id=info.source_id,
                      payload={"images": len(outcomes)})
            )

        summary = ImageBatchSummary(
            images=len(outcomes),
            detections=detections,
            recognized=recognized,
            unknown=detections - recognized,
            elapsed_s=time.perf_counter() - started,
            outcomes=outcomes,
            identities=dict(sorted(identities.items(), key=lambda kv: -kv[1])),
            failures=failures,
        )
        logger.info(
            "Image processing finished",
            extra={
                "images": summary.images,
                "detections": summary.detections,
                "recognized": summary.recognized,
                "unknown": summary.unknown,
                "failed": len(failures),
            },
        )
        return summary

    def _process_one(self, frame: Frame) -> ImageOutcome:
        result = self._pipeline.process_image(frame.image, source_id=frame.source_id)
        result.frame_index = frame.index
        for detection in result.detections:
            detection.frame_index = frame.index

        annotated = self._renderer.render(frame.image, result, None)
        stem = Path(frame.path).stem if frame.path else f"image_{frame.index:05d}"
        outcome = ImageOutcome(source_path=frame.path or stem, result=result)

        if self._save:
            outcome.annotated_path = str(self._write_annotated(stem, annotated))
            outcome.crop_paths = [str(p) for p in self._write_crops(stem, frame, result)]
            if self._config.output.save_metadata:
                outcome.metadata_path = str(
                    write_json(
                        self._engine.paths.output_metadata_dir / f"{sanitize_filename(stem)}.json",
                        {"input": frame.path, "annotated": outcome.annotated_path,
                         **result.to_dict()},
                        overwrite=self._config.output.overwrite,
                    )
                )
            for detection in result.detections:
                if self._snapshots.save(frame.image, detection, result, annotated=annotated):
                    self._engine.events.emit(
                        Event(
                            type=EventType.SNAPSHOT_SAVED,
                            source_id=result.source_id,
                            frame_index=frame.index,
                            identity_id=detection.identity_id,
                            identity_name=detection.identity_name,
                            similarity=detection.reid_similarity,
                        )
                    )

        for detection in result.detections:
            self._engine.events.emit(
                Event(
                    type=(
                        EventType.PERSON_RECOGNIZED
                        if detection.effective.is_recognized
                        else EventType.UNKNOWN_PERSON_DETECTED
                    ),
                    source_id=result.source_id,
                    frame_index=frame.index,
                    identity_id=detection.identity_id,
                    identity_name=detection.identity_name,
                    identity_title=detection.identity_title,
                    similarity=detection.reid_similarity,
                    bbox=detection.bbox.to_list(),
                    detector_confidence=detection.detector_confidence,
                    payload={"image": frame.path} if frame.path else {},
                )
            )
        return outcome

    def _write_annotated(self, stem: str, annotated) -> Path:
        directory = self._engine.paths.annotated_images_dir
        target = unique_path(
            directory / f"{sanitize_filename(stem)}_annotated.jpg",
            overwrite=self._config.output.overwrite,
        )
        return imwrite(target, annotated, jpeg_quality=self._config.output.jpeg_quality)

    def _write_crops(self, stem: str, frame: Frame, result: FrameResult) -> list[Path]:
        if not self._config.output.save_crops:
            return []
        written: list[Path] = []
        for index, detection in enumerate(result.detections):
            crop = crop_bbox(frame.image, detection.bbox)
            if crop is None:
                continue
            label = sanitize_filename(detection.identity_id or "unknown")
            target = unique_path(
                self._engine.paths.crops_dir / f"{sanitize_filename(stem)}_{index}_{label}.jpg",
                overwrite=self._config.output.overwrite,
            )
            written.append(imwrite(target, crop, jpeg_quality=self._config.output.jpeg_quality))
        return written
