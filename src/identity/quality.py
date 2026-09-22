"""Reference-image quality assessment for enrollment.

One-shot enrollment lives or dies on the reference image, so the operator is
told what is wrong with it. The checks warn by default and only reject when
``enrollment.quality.fail_on_warnings`` is set: real deployments have plenty of
unusual-but-usable reference photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config.schema import QualityConfig
from src.core.types import BBox
from src.utils.image import measure_quality


@dataclass(slots=True)
class QualityReport:
    """Outcome of the quality gates for one reference crop."""

    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "warnings": list(self.warnings), "metrics": dict(self.metrics)}


def assess_reference(
    image: np.ndarray,
    crop: np.ndarray,
    bbox: BBox,
    config: QualityConfig,
    *,
    person_count: int = 1,
    face: Any | None = None,
) -> QualityReport:
    """Evaluate a reference image and the crop taken from it.

    In face mode ``crop`` is the aligned face chip and ``face`` describes the
    detected face, so the checks that matter change: face resolution and
    landmark quality replace body size and aspect ratio.
    """
    report = QualityReport()
    if not config.enabled:
        return report
    if face is not None:
        return _assess_face_reference(image, crop, config, face, person_count)

    image_h, image_w = image.shape[:2]
    quality = measure_quality(crop)
    area_ratio = bbox.area / float(image_h * image_w) if image_h and image_w else 0.0

    report.metrics = {
        **quality.to_dict(),
        "bbox_width": round(bbox.width, 1),
        "bbox_height": round(bbox.height, 1),
        "area_ratio": round(area_ratio, 4),
        "aspect_ratio": round(bbox.aspect_ratio, 3),
        "person_count": person_count,
        "image_width": image_w,
        "image_height": image_h,
    }

    if bbox.height < config.min_bbox_height:
        report.warnings.append(
            f"person is only {bbox.height:.0f}px tall (minimum {config.min_bbox_height}px); "
            "fine appearance detail may be lost"
        )
    if bbox.width < config.min_bbox_width:
        report.warnings.append(
            f"person is only {bbox.width:.0f}px wide (minimum {config.min_bbox_width}px)"
        )
    if area_ratio < config.min_crop_area_ratio:
        report.warnings.append(
            f"person covers {area_ratio:.1%} of the image "
            f"(minimum {config.min_crop_area_ratio:.1%}); use a closer reference photo"
        )
    if quality.blur_variance < config.min_blur_variance:
        report.warnings.append(
            f"reference crop looks blurry (sharpness {quality.blur_variance:.1f} < "
            f"{config.min_blur_variance})"
        )
    if quality.brightness < config.min_brightness:
        report.warnings.append(
            f"reference crop is very dark (mean intensity {quality.brightness:.1f} < "
            f"{config.min_brightness})"
        )
    elif quality.brightness > config.max_brightness:
        report.warnings.append(
            f"reference crop is over-exposed (mean intensity {quality.brightness:.1f} > "
            f"{config.max_brightness})"
        )
    if bbox.aspect_ratio < config.min_aspect_ratio:
        report.warnings.append(
            f"person box aspect ratio {bbox.aspect_ratio:.2f} is below "
            f"{config.min_aspect_ratio}; the body may be cropped or seated, which "
            "weakens whole-body appearance matching"
        )
    if bbox.touches_border(image_w, image_h, config.truncation_border_tolerance):
        report.warnings.append(
            "person touches the image border and is probably truncated; "
            "a fully visible body gives a stronger reference embedding"
        )
    if person_count > 1:
        report.warnings.append(
            f"{person_count} people detected in the reference image; "
            "only the selected one was enrolled"
        )
    return report


def _assess_face_reference(
    image: np.ndarray,
    chip: np.ndarray,
    config: QualityConfig,
    face: Any,
    person_count: int,
) -> QualityReport:
    """Quality gates for a face reference.

    Face recognition accuracy is driven almost entirely by how much real facial
    detail the reference carries, so the checks are about resolution, sharpness
    and exposure of the face itself -- not of the body around it.
    """
    report = QualityReport()
    image_h, image_w = image.shape[:2]
    quality = measure_quality(chip)

    report.metrics = {
        **quality.to_dict(),
        "modality": "face",
        "face_size": round(face.size, 1),
        "face_score": round(face.score, 4),
        "eye_distance": round(face.eye_distance, 1),
        "aligned": bool(face.has_landmarks),
        "person_count": person_count,
        "image_width": image_w,
        "image_height": image_h,
    }

    if face.size < config.min_face_size:
        report.warnings.append(
            f"face is only {face.size:.0f}px across (recommended at least "
            f"{config.min_face_size}px); recognition accuracy drops sharply "
            "below this"
        )
    if face.has_landmarks and face.eye_distance < config.min_eye_distance:
        report.warnings.append(
            f"inter-ocular distance is {face.eye_distance:.0f}px (recommended at "
            f"least {config.min_eye_distance:.0f}px); the face may be turned away"
        )
    if not face.has_landmarks:
        report.warnings.append(
            "no facial landmarks were available, so the reference could not be "
            "aligned; an aligned reference gives a materially better embedding"
        )
    if face.score < config.min_face_score:
        report.warnings.append(
            f"face detector confidence is only {face.score:.2f}; the reference "
            "may be partially occluded or strongly non-frontal"
        )
    if quality.blur_variance < config.min_blur_variance:
        report.warnings.append(
            f"face looks blurry (sharpness {quality.blur_variance:.1f} < "
            f"{config.min_blur_variance})"
        )
    if quality.brightness < config.min_brightness:
        report.warnings.append(
            f"face is very dark (mean intensity {quality.brightness:.1f})"
        )
    elif quality.brightness > config.max_brightness:
        report.warnings.append(
            f"face is over-exposed (mean intensity {quality.brightness:.1f})"
        )
    if person_count > 1:
        report.warnings.append(
            f"{person_count} people detected in the reference image; only the "
            "selected one was enrolled"
        )
    return report
