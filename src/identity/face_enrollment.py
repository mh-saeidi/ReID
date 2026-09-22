"""One-shot enrollment from a passport photograph.

The previous pipeline required YOLO26 to find a *person* in the reference image
before it would look for a face. For a passport photo that is a pure liability:
the detector contributes nothing (there is exactly one subject, centred and
filling the frame) and it can only reject a valid reference. A head-and-
shoulders crop on a plain background is not what a person detector is trained
on, and the failure mode is silent -- a perfectly good passport photo rejected
with "no person detected".

So enrollment goes face-first. The face detector is the right tool for the job
and is the only detector involved.

The operator's original file is opened read-only and never written to. A copy
is placed in the gallery so the entry can be audited or rebuilt later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import EnrollmentError
from src.face.align import align_face
from src.face.detector import FaceDetector
from src.face.quality import FaceQualityScore, assess_face_quality
from src.face.types import FaceDetection
from src.face.visibility import FaceVisibility, classify_visibility
from src.reid.encoder import ReIDEncoder, l2_normalize
from src.utils.image import imread
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ReferenceWarning(str, Enum):
    """Machine-readable problems with a reference photograph."""

    REFERENCE_FACE_TOO_SMALL = "REFERENCE_FACE_TOO_SMALL"
    REFERENCE_TOO_BLURRY = "REFERENCE_TOO_BLURRY"
    REFERENCE_POSE_TOO_EXTREME = "REFERENCE_POSE_TOO_EXTREME"
    REFERENCE_POORLY_EXPOSED = "REFERENCE_POORLY_EXPOSED"
    REFERENCE_LOW_QUALITY = "REFERENCE_LOW_QUALITY"
    REFERENCE_OCCLUDED = "REFERENCE_OCCLUDED"
    REFERENCE_LOW_RESOLUTION = "REFERENCE_LOW_RESOLUTION"
    MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
    WEAK_LANDMARKS = "WEAK_LANDMARKS"


class FaceSelection(str, Enum):
    LARGEST_FACE = "largest_face"
    HIGHEST_CONFIDENCE = "highest_confidence"
    CENTER_MOST = "center_most"


@dataclass(slots=True)
class ReferenceQualityConfig:
    """Gates applied to a reference photograph.

    Deliberately stricter than the runtime gates. A weak reference degrades
    every future comparison against that person, permanently -- whereas a weak
    query degrades only that frame.
    """

    enabled: bool = True
    min_interocular_px: float = 28.0
    min_face_px: int = 70
    min_sharpness: float = 0.30
    min_exposure: float = 0.35
    max_yaw_deg: float = 25.0
    max_pitch_deg: float = 20.0
    min_landmark_score: float = 0.55
    min_overall_quality: float = 0.45
    require_full_face: bool = True
    fail_on_warnings: bool = False
    max_faces: int = 1
    selection: FaceSelection = FaceSelection.LARGEST_FACE


@dataclass(slots=True)
class ReferenceReport:
    """What was found in a reference photograph."""

    warnings: list[ReferenceWarning] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    faces_detected: int = 0
    quality: FaceQualityScore | None = None
    visibility: FaceVisibility = FaceVisibility.FULL_FACE

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "warnings": [w.value for w in self.warnings],
            "messages": list(self.messages),
            "faces_detected": self.faces_detected,
            "visibility": self.visibility.value,
            "quality": self.quality.to_dict() if self.quality else None,
        }


@dataclass(slots=True)
class FaceEnrollmentResult:
    """The embedding and provenance produced from one passport photo."""

    person_id: str
    embedding: np.ndarray
    face: FaceDetection
    report: ReferenceReport
    image_path: str
    aligned_chip: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        return int(self.embedding.shape[-1])


def select_face(
    faces: list[FaceDetection], strategy: FaceSelection, shape: tuple[int, int]
) -> FaceDetection:
    """Pick which detected face the reference is about."""
    if strategy is FaceSelection.HIGHEST_CONFIDENCE:
        return max(faces, key=lambda f: f.score)
    if strategy is FaceSelection.CENTER_MOST:
        height, width = shape
        cx, cy = width / 2.0, height / 2.0
        return min(
            faces,
            key=lambda f: (f.bbox.center[0] - cx) ** 2 + (f.bbox.center[1] - cy) ** 2,
        )
    return max(faces, key=lambda f: f.bbox.area)


class FaceEnroller:
    """Turns a passport photograph into a gallery-ready face embedding."""

    def __init__(
        self,
        detector: FaceDetector,
        encoder: ReIDEncoder,
        config: ReferenceQualityConfig,
        *,
        chip_size: int = 112,
    ) -> None:
        self._detector = detector
        self._encoder = encoder
        self._config = config
        self._chip_size = chip_size

    def enroll(self, person_id: str, image_path: Path) -> FaceEnrollmentResult:
        """Enroll one person from one image.

        Raises:
            EnrollmentError: no face, too many faces, or -- when
                ``fail_on_warnings`` is set -- a reference that fails its
                quality gates.
        """
        path = Path(image_path)
        if not path.exists():
            raise EnrollmentError(
                f"reference image for '{person_id}' not found: {path}"
            )

        image = imread(path)          # read-only; the original is never written
        faces = self._detector.detect(image)
        report = ReferenceReport(faces_detected=len(faces))

        if not faces:
            raise EnrollmentError(
                f"no face detected in the reference image for '{person_id}' "
                f"({path}). This system identifies people by face, so the "
                "reference must show one clearly. Check the photograph is not "
                "rotated, heavily cropped or very low resolution, or lower "
                "face.detection_confidence."
            )

        if len(faces) > self._config.max_faces:
            report.warnings.append(ReferenceWarning.MULTIPLE_FACES_DETECTED)
            boxes = ", ".join(
                f"[{f.bbox.x1:.0f},{f.bbox.y1:.0f},{f.bbox.x2:.0f},{f.bbox.y2:.0f}]"
                f" score={f.score:.2f}"
                for f in sorted(faces, key=lambda f: -f.bbox.area)[:4]
            )
            if self._config.max_faces <= 1:
                raise EnrollmentError(
                    f"{len(faces)} faces detected in the reference image for "
                    f"'{person_id}' ({path}); refusing to guess which person to "
                    f"register. Faces: {boxes}. Crop the image to one face, or "
                    "raise enrollment.reference_quality.max_faces to use the "
                    f"'{self._config.selection.value}' selection strategy."
                )
            report.messages.append(
                f"{len(faces)} faces found; selected by "
                f"{self._config.selection.value}"
            )

        face = select_face(faces, self._config.selection, image.shape[:2])
        if not face.has_landmarks:
            raise EnrollmentError(
                f"the face detected for '{person_id}' has no landmarks, so it "
                "cannot be aligned. Alignment is required: an unaligned face "
                "produces an embedding that is not comparable with the gallery."
            )

        visibility = classify_visibility(image, face.bbox, face.landmarks)
        report.visibility = visibility.visibility

        try:
            chip = align_face(image, face.landmarks, self._chip_size, face.bbox)
        except Exception as exc:  # noqa: BLE001 - cv2 raises broadly
            raise EnrollmentError(
                f"could not align the face for '{person_id}' ({path}): {exc}"
            ) from exc

        quality = assess_face_quality(chip, face, visibility)
        report.quality = quality
        self._apply_gates(report, face, quality, image.shape[:2])

        for message in report.messages:
            logger.warning(
                "Reference photograph", extra={"identity": person_id, "issue": message}
            )
        if report.warnings and self._config.fail_on_warnings:
            raise EnrollmentError(
                f"the reference image for '{person_id}' was rejected by the "
                "quality gates: " + "; ".join(report.messages) + ". Set "
                "enrollment.reference_quality.fail_on_warnings: false to enroll "
                "it anyway, accepting the reduced accuracy."
            )

        embedding = l2_normalize(self._encoder.embed(chip))
        if not np.any(embedding):
            raise EnrollmentError(
                f"the face encoder returned an empty embedding for '{person_id}'"
            )

        logger.info(
            "Person enrolled from passport photograph",
            extra={
                "identity": person_id,
                "face": face.bbox.to_int(),
                "face_score": round(face.score, 3),
                "iod_px": round(face.eye_distance, 1),
                "quality": round(quality.overall, 3),
                "visibility": visibility.visibility.value,
                "dim": int(embedding.shape[-1]),
                "warnings": len(report.warnings),
            },
        )
        return FaceEnrollmentResult(
            person_id=person_id,
            embedding=embedding,
            face=face,
            report=report,
            image_path=str(path),
            aligned_chip=chip,
        )

    def _apply_gates(
        self,
        report: ReferenceReport,
        face: FaceDetection,
        quality: FaceQualityScore,
        shape: tuple[int, int],
    ) -> None:
        config = self._config
        if not config.enabled:
            return

        def flag(warning: ReferenceWarning, message: str) -> None:
            report.warnings.append(warning)
            report.messages.append(message)

        if face.eye_distance < config.min_interocular_px:
            flag(
                ReferenceWarning.REFERENCE_FACE_TOO_SMALL,
                f"inter-ocular distance {face.eye_distance:.0f}px is below "
                f"{config.min_interocular_px:.0f}px; use a higher-resolution photo",
            )
        if face.size < config.min_face_px:
            flag(
                ReferenceWarning.REFERENCE_LOW_RESOLUTION,
                f"face is {face.size:.0f}px across (minimum {config.min_face_px}px)",
            )
        if quality.sharpness < config.min_sharpness:
            flag(
                ReferenceWarning.REFERENCE_TOO_BLURRY,
                f"reference is soft or blurred (sharpness {quality.sharpness:.2f})",
            )
        if quality.exposure < config.min_exposure:
            flag(
                ReferenceWarning.REFERENCE_POORLY_EXPOSED,
                f"reference is poorly exposed (score {quality.exposure:.2f}, "
                f"mean intensity {quality.mean_intensity:.0f})",
            )
        if abs(quality.yaw_deg) > config.max_yaw_deg or abs(quality.pitch_deg) > config.max_pitch_deg:
            flag(
                ReferenceWarning.REFERENCE_POSE_TOO_EXTREME,
                f"reference is non-frontal (yaw {quality.yaw_deg:.0f} deg, "
                f"pitch {quality.pitch_deg:.0f} deg); a passport-style frontal "
                "photograph matches a wider range of queries",
            )
        if quality.landmarks < config.min_landmark_score:
            flag(
                ReferenceWarning.WEAK_LANDMARKS,
                f"weak landmark fit (score {quality.landmarks:.2f}); alignment "
                "may be imprecise",
            )
        if config.require_full_face and report.visibility is not FaceVisibility.FULL_FACE:
            flag(
                ReferenceWarning.REFERENCE_OCCLUDED,
                f"reference face is {report.visibility.value}; an unobstructed "
                "reference is strongly preferred",
            )
        if quality.overall < config.min_overall_quality:
            flag(
                ReferenceWarning.REFERENCE_LOW_QUALITY,
                f"overall reference quality {quality.overall:.2f} is below "
                f"{config.min_overall_quality:.2f} "
                f"(weakest: {quality.weakest_component})",
            )
        _ = shape
