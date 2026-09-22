"""One-shot enrollment: reference image -> normalised gallery embedding.

The operator supplies exactly one photo per person. Detection, person
selection, cropping, quality assessment, embedding and normalisation all happen
here; the original file on disk is never modified.

The result carries a list of embeddings even though one-shot enrollment
produces a single entry, so multi-image enrollment (average / medoid of several
references) can be added later without changing this module's callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config.schema import AppConfig, PersonConfig, SelectionStrategy
from src.core.exceptions import (
    AmbiguousEnrollmentError,
    EnrollmentError,
    NoFaceFoundError,
    NoPersonFoundError,
)
from src.core.types import BBox, Detection
from src.detection.detector import Detector
from src.identity.quality import QualityReport, assess_reference
from src.reid.encoder import ReIDEncoder, l2_normalize
from src.reid.preprocess import CropPreprocessor, CropResult, CropStatus
from src.utils.image import imread, imwrite, sanitize_filename
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EnrollmentResult:
    """Everything produced by enrolling one reference image."""

    person_id: str
    embedding: np.ndarray
    bbox: BBox
    detector_confidence: float
    image_path: str
    quality: QualityReport
    crop_path: str | None = None
    detections_found: int = 1
    embeddings: list[np.ndarray] = field(default_factory=list)
    """Per-reference embeddings; length 1 for one-shot enrollment."""
    face: Any | None = None
    """The FaceDetection the embedding came from, in face mode."""

    @property
    def dimension(self) -> int:
        return int(self.embedding.shape[-1])


def select_detection(
    detections: Sequence[Detection],
    strategy: SelectionStrategy,
    image_shape: tuple[int, int],
) -> Detection:
    """Pick which detected person the reference image is about."""
    if not detections:
        raise NoPersonFoundError("no detections to select from")
    if strategy is SelectionStrategy.LARGEST_PERSON:
        return max(detections, key=lambda d: d.bbox.area)
    if strategy is SelectionStrategy.HIGHEST_CONFIDENCE:
        return max(detections, key=lambda d: d.confidence)
    if strategy is SelectionStrategy.CENTER_MOST:
        height, width = image_shape
        cx, cy = width / 2.0, height / 2.0
        return min(
            detections,
            key=lambda d: (d.bbox.center[0] - cx) ** 2 + (d.bbox.center[1] - cy) ** 2,
        )
    raise EnrollmentError(  # pragma: no cover - enum is exhaustive
        f"unknown enrollment.selection_strategy '{strategy}'"
    )


class Enroller:
    """Turns reference images into normalised gallery embeddings."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        encoder: ReIDEncoder,
        preprocessor: CropPreprocessor,
        *,
        crop_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._encoder = encoder
        self._preprocessor = preprocessor
        self._crop_dir = crop_dir

    def enroll(self, person: PersonConfig, image_paths: Sequence[Path]) -> EnrollmentResult:
        """Enroll a person from one (or, in future, several) reference images.

        Raises:
            NoPersonFoundError: the detector found nobody in the reference image.
            AmbiguousEnrollmentError: several people were found and
                ``enrollment.require_single_person`` forbids guessing.
            EnrollmentError: the image is unreadable, or quality gates failed
                while ``fail_on_warnings`` is enabled.
        """
        if not image_paths:
            raise EnrollmentError(f"person '{person.id}' has no reference image configured")

        results = [self._enroll_single(person, Path(path)) for path in image_paths]
        primary = results[0]
        if len(results) == 1:
            primary.embeddings = [primary.embedding]
            return primary

        # Forward-compatible aggregation: mean of normalised references, re-normalised.
        stacked = np.stack([r.embedding for r in results], axis=0)
        primary.embedding = l2_normalize(stacked.mean(axis=0))
        primary.embeddings = [r.embedding for r in results]
        return primary

    def _enroll_single(self, person: PersonConfig, image_path: Path) -> EnrollmentResult:
        if not image_path.exists():
            raise EnrollmentError(
                f"reference image for '{person.id}' not found: {image_path}. "
                "Check people[].image_path in the configuration."
            )
        image = imread(image_path)  # the original file is only ever read
        detections = self._detect_people(image)

        if not detections:
            raise NoPersonFoundError(
                f"no person detected in the reference image for '{person.id}' "
                f"({image_path}). Try a clearer, closer photo, or lower "
                "detector.confidence / enrollment.detector_confidence."
            )
        if len(detections) > 1 and self._config.enrollment.require_single_person:
            boxes = ", ".join(
                f"[{d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f}] "
                f"conf={d.confidence:.2f}"
                for d in sorted(detections, key=lambda d: -d.bbox.area)[:5]
            )
            raise AmbiguousEnrollmentError(
                f"{len(detections)} people detected in the reference image for "
                f"'{person.id}' ({image_path}); refusing to guess which one to "
                f"register. Boxes: {boxes}. Either crop the image to a single "
                "person, or set enrollment.require_single_person: false to use "
                f"the '{self._config.enrollment.selection_strategy.value}' strategy."
            )

        selected = select_detection(
            detections, self._config.enrollment.selection_strategy, image.shape[:2]
        )
        extraction = self._preprocessor.extract(image, selected.bbox)
        if not extraction.ok:
            raise self._extraction_error(person, image_path, selected, extraction)
        crop = extraction.image

        report = assess_reference(
            image,
            crop,
            selected.bbox,
            self._config.enrollment.quality,
            person_count=len(detections),
            face=extraction.face,
        )
        for warning in report.warnings:
            logger.warning(
                "Reference image quality", extra={"identity": person.id, "issue": warning}
            )
        if report.warnings and self._config.enrollment.quality.fail_on_warnings:
            raise EnrollmentError(
                f"reference image for '{person.id}' rejected by quality gates: "
                + "; ".join(report.warnings)
                + ". Set enrollment.quality.fail_on_warnings: false to enroll anyway."
            )

        embedding = self._encoder.embed(crop)
        if not np.any(embedding):
            raise EnrollmentError(
                f"the ReID encoder produced an empty embedding for '{person.id}' "
                f"({image_path}); the crop may be degenerate"
            )
        embedding = l2_normalize(embedding)

        crop_path = self._save_crop(person, image_path, crop)
        face = extraction.face
        logger.info(
            "Person enrolled",
            extra={
                "identity": person.id,
                "person": person.name,
                "bbox": selected.bbox.to_int(),
                "det_conf": round(selected.confidence, 3),
                "dim": int(embedding.shape[-1]),
                "warnings": len(report.warnings),
            },
        )
        return EnrollmentResult(
            person_id=person.id,
            embedding=embedding,
            bbox=selected.bbox,
            detector_confidence=selected.confidence,
            image_path=str(image_path),
            quality=report,
            crop_path=str(crop_path) if crop_path else None,
            detections_found=len(detections),
            face=face,
        )

    def _extraction_error(
        self,
        person: PersonConfig,
        image_path: Path,
        selected: Detection,
        extraction: CropResult,
    ) -> EnrollmentError:
        """Turn a failed crop into a message that says how to fix the photo.

        In face mode this must never degrade into body-appearance enrollment:
        an identity registered from clothing would defeat the entire point of
        the mode, and would do so silently.
        """
        if extraction.status is CropStatus.NO_FACE:
            return NoFaceFoundError(
                f"no face detected in the reference image for '{person.id}' "
                f"({image_path}). recognition.mode is 'face', which identifies "
                "people from facial appearance only, so the reference photo must "
                "show the person's face. Use a clearer, more frontal photo, or "
                "lower face.detection_confidence."
            )
        if extraction.status is CropStatus.FACE_TOO_SMALL:
            return NoFaceFoundError(
                f"the face in the reference image for '{person.id}' ({image_path}) "
                f"is unusable: {extraction.detail}. Use a higher-resolution or "
                "closer photo, or lower face.min_face_size if you accept the "
                "reduced accuracy."
            )
        if extraction.status is CropStatus.LOW_QUALITY:
            return EnrollmentError(
                f"the reference image for '{person.id}' ({image_path}) was "
                f"rejected: {extraction.detail}"
            )
        return EnrollmentError(
            f"the person region in '{image_path}' is unusable "
            f"(box {selected.bbox.to_int()}): {extraction.detail or 'too small to crop'}. "
            "Use a higher-resolution reference image."
        )

    def _detect_people(self, image: np.ndarray) -> list[Detection]:
        """Detect with an optional enrollment-specific confidence threshold."""
        override = self._config.enrollment.detector_confidence
        if override is None:
            return self._detector.detect(image)

        original = self._config.detector.confidence
        try:
            self._config.detector.confidence = override
            return self._detector.detect(image)
        finally:
            self._config.detector.confidence = original

    def _save_crop(self, person: PersonConfig, image_path: Path, crop: np.ndarray) -> Path | None:
        """Persist the crop that was actually embedded (debugging / audit)."""
        if not self._config.enrollment.save_normalized_crop or self._crop_dir is None:
            return None
        name = f"{sanitize_filename(person.id)}_{sanitize_filename(image_path.stem)}.jpg"
        target = self._crop_dir / name
        try:
            return imwrite(target, crop, jpeg_quality=self._config.output.jpeg_quality)
        except Exception as exc:  # noqa: BLE001 - never fail enrollment over a debug artefact
            logger.warning("Could not save enrollment crop: %s", exc)
            return None
