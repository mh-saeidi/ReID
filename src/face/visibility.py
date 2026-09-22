"""Face visibility and occlusion classification.

A recognition system that only knows "face / no face" cannot behave correctly
when someone is wearing a mask. The similarity a masked face produces is drawn
from a different distribution than a full face: it is systematically lower
*and* systematically less discriminative, because the visible region carries
less identity information. Comparing it against a threshold calibrated on full
faces produces false rejects; lowering that threshold to compensate produces
false accepts. Both are wrong.

So visibility is classified explicitly, and the decision layer is told which
distribution the score came from.

Classification uses the five-point landmark geometry plus regional appearance
statistics, not a learned occlusion model. That is a deliberate trade: it needs
no extra network, runs in microseconds, and is good enough to *route* a face to
the right threshold. It is not good enough to be an occlusion detector in its
own right, and is not presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np

from src.core.types import BBox


class FaceVisibility(str, Enum):
    """How much of the face is usable for recognition."""

    FULL_FACE = "full_face"
    """Eyes, nose and mouth all visible and consistent."""

    PARTIAL_FACE = "partial_face"
    """A meaningful region is missing -- strong pose, or an edge crop."""

    MASKED = "masked"
    """The lower face is covered; the periocular region remains."""

    HEAVILY_OCCLUDED = "heavily_occluded"
    """Too little visible to support a recognition claim."""

    NO_USABLE_FACE = "no_usable_face"
    """Nothing to work with."""

    @property
    def is_usable(self) -> bool:
        return self in (
            FaceVisibility.FULL_FACE,
            FaceVisibility.PARTIAL_FACE,
            FaceVisibility.MASKED,
        )

    @property
    def threshold_key(self) -> str:
        """Which calibrated threshold family this face belongs to."""
        if self is FaceVisibility.FULL_FACE:
            return "full"
        if self is FaceVisibility.MASKED:
            return "masked"
        return "partial"


@dataclass(slots=True)
class VisibilityReport:
    """The visibility decision and the evidence behind it."""

    visibility: FaceVisibility
    confidence: float
    """How sure the classifier is, in [0, 1]. Not an identity confidence."""
    lower_face_visible: float
    """Structure score of the mouth/chin region, in [0, 1]. Low means covered."""
    eye_region_visible: float
    yaw_estimate: float
    """Approximate yaw from landmark asymmetry, in degrees. Sign is arbitrary."""
    pitch_estimate: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "visibility": self.visibility.value,
            "confidence": round(self.confidence, 4),
            "lower_face_visible": round(self.lower_face_visible, 4),
            "eye_region_visible": round(self.eye_region_visible, 4),
            "yaw_deg": round(self.yaw_estimate, 1),
            "pitch_deg": round(self.pitch_estimate, 1),
            "reasons": list(self.reasons),
        }


# Landmark indices in the five-point convention.
RIGHT_EYE, LEFT_EYE, NOSE, RIGHT_MOUTH, LEFT_MOUTH = range(5)


def estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Rough yaw and pitch from five landmarks.

    Yaw comes from how far the nose sits from the midpoint between the eyes,
    relative to the inter-ocular distance: a frontal face has the nose centred,
    a turned face does not. Pitch comes from the nose's vertical position
    between the eye line and the mouth line.

    These are geometric approximations, not a pose network. They are accurate
    enough to gate a decision and are reported as estimates.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    right_eye, left_eye = points[RIGHT_EYE], points[LEFT_EYE]
    nose = points[NOSE]
    mouth = (points[RIGHT_MOUTH] + points[LEFT_MOUTH]) / 2.0

    eye_centre = (right_eye + left_eye) / 2.0
    interocular = float(np.linalg.norm(left_eye - right_eye))
    if interocular < 1e-3:
        return 90.0, 0.0

    # Horizontal nose offset, in units of half the inter-ocular distance.
    horizontal = float(nose[0] - eye_centre[0]) / (interocular / 2.0)
    yaw = float(np.clip(horizontal, -2.0, 2.0) * 45.0)

    eye_to_mouth = float(np.linalg.norm(mouth - eye_centre))
    if eye_to_mouth < 1e-3:
        return yaw, 0.0
    # A frontal face puts the nose around 45-55% of the way down.
    vertical = (float(nose[1] - eye_centre[1]) / eye_to_mouth) - 0.5
    pitch = float(np.clip(vertical, -1.0, 1.0) * 60.0)
    return yaw, pitch


def _region_structure(patch: np.ndarray) -> float:
    """How much facial structure a region contains, in roughly [0, 1].

    Deliberately *not* skin-colour detection. Colour-based skin segmentation
    (YCrCb bounds and friends) fails on beards, shadow and strong colour casts,
    and its error rate differs systematically across skin tones -- which would
    make masked-face handling work better for some people than others. That is
    not an acceptable property for a biometric system.

    Instead this measures texture: an uncovered face region contains edges
    (lips, nostrils, the shadow under the nose) and local contrast, while a
    mask, a scarf or a hand is comparatively flat and uniform in colour. Both
    of those hold regardless of the wearer's complexion.

    Two signals are combined:
      * gradient energy -- how much edge structure is present;
      * colour dispersion -- how far the region is from a single flat colour.
    """
    if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return 0.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Normalise out overall illumination so a dark scene is not read as flat.
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gradient = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())
    # ~200 is a typical variance for an uncovered face region at working sizes.
    edge_score = float(np.clip(gradient / 200.0, 0.0, 1.0))

    # Chroma spread: a flat fabric occupies a much tighter colour range.
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma_spread = float(lab[:, :, 1].std() + lab[:, :, 2].std())
    colour_score = float(np.clip(chroma_spread / 18.0, 0.0, 1.0))

    return float(0.65 * edge_score + 0.35 * colour_score)


def _safe_patch(image: np.ndarray, box: BBox) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.clip(width, height).to_int()
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def classify_visibility(
    image: np.ndarray,
    bbox: BBox,
    landmarks: np.ndarray | None,
    *,
    max_yaw: float = 45.0,
    max_pitch: float = 35.0,
    mask_structure_threshold: float = 0.30,
    eye_structure_threshold: float = 0.20,
) -> VisibilityReport:
    """Classify how much of this face is usable.

    Args:
        image: Frame or crop the face lives in.
        bbox: Face box in that image's coordinates.
        landmarks: Five-point landmarks, in the same coordinates.
        max_yaw/max_pitch: Beyond these the face counts as partial.
        mask_structure_threshold: Structure score in the mouth region below
            which the lower face is considered covered.
        eye_structure_threshold: The same test for the periocular region;
            failing it means there is nothing left to recognise from.
    """
    reasons: list[str] = []

    if landmarks is None or len(landmarks) < 5:
        return VisibilityReport(
            FaceVisibility.PARTIAL_FACE, 0.3, 0.5, 0.5, 0.0, 0.0,
            ["no landmarks, so visibility could not be assessed"],
        )

    points = np.asarray(landmarks, dtype=np.float32)
    yaw, pitch = estimate_pose(points)

    interocular = float(np.linalg.norm(points[LEFT_EYE] - points[RIGHT_EYE]))
    if interocular < 4.0:
        return VisibilityReport(
            FaceVisibility.HEAVILY_OCCLUDED, 0.6, 0.0, 0.0, yaw, pitch,
            ["inter-ocular distance is too small to assess"],
        )

    eye_centre = (points[LEFT_EYE] + points[RIGHT_EYE]) / 2.0
    mouth_centre = (points[LEFT_MOUTH] + points[RIGHT_MOUTH]) / 2.0

    # Periocular band around the eye line.
    eye_box = BBox(
        eye_centre[0] - interocular * 0.9, eye_centre[1] - interocular * 0.45,
        eye_centre[0] + interocular * 0.9, eye_centre[1] + interocular * 0.45,
    )
    # Mouth/chin band, which a mask covers and a bare face does not.
    mouth_box = BBox(
        mouth_centre[0] - interocular * 0.75, mouth_centre[1] - interocular * 0.35,
        mouth_centre[0] + interocular * 0.75, mouth_centre[1] + interocular * 0.55,
    )

    eye_visible = _region_structure(_safe_patch(image, eye_box))
    lower_landmark = _region_structure(_safe_patch(image, mouth_box))

    # Cross-check against the detected box, independent of the landmarks.
    #
    # Heavy occlusion moves the landmarks: when a scarf covers the mouth, the
    # detector places the "mouth" points on whatever skin is still visible, and
    # the landmark-derived region then reports a clear face. Sampling the lower
    # third of the box catches that, because the box still spans the covered
    # area. Taking the pessimistic reading of the two means an ambiguous face is
    # routed to the more conservative threshold rather than the looser one.
    lower_box = BBox(bbox.x1, bbox.y1 + bbox.height * 0.62, bbox.x2, bbox.y2)
    lower_region = _region_structure(_safe_patch(image, lower_box))
    lower_visible = min(lower_landmark, lower_region)
    if lower_region < lower_landmark - 0.25:
        reasons.append(
            f"landmark region reads {lower_landmark:.2f} but the lower face "
            f"box reads {lower_region:.2f}; treating it as occluded"
        )

    # Geometry cross-check. Heavy occlusion makes the detector box only the
    # visible upper face, and a box that no longer spans the covered area
    # cannot reveal the occlusion by sampling inside it. But the box becomes
    # abnormally short relative to the inter-ocular distance, which it can.
    height_ratio = bbox.height / max(interocular, 1e-3)
    if height_ratio < 1.75:
        reasons.append(
            f"face box is short for its inter-ocular distance "
            f"(height/IOD {height_ratio:.2f} < 1.75); the lower face is "
            "probably outside the detected box"
        )
        lower_visible = min(lower_visible, 0.15)

    # --- decide ------------------------------------------------------------
    if eye_visible < eye_structure_threshold and lower_visible < mask_structure_threshold:
        reasons.append(
            f"neither the eye region ({eye_visible:.2f}) nor the lower face "
            f"({lower_visible:.2f}) shows facial structure"
        )
        return VisibilityReport(
            FaceVisibility.HEAVILY_OCCLUDED, 0.7, lower_visible, eye_visible,
            yaw, pitch, reasons,
        )

    if lower_visible < mask_structure_threshold <= eye_visible + 1.0:
        if eye_visible >= eye_structure_threshold:
            reasons.append(
                f"lower face largely covered (structure {lower_visible:.2f} < "
                f"{mask_structure_threshold:.2f}) while the eye region is visible "
                f"({eye_visible:.2f}) -- consistent with a mask"
            )
            confidence = float(np.clip((mask_structure_threshold - lower_visible) * 4.0, 0.3, 0.95))
            return VisibilityReport(
                FaceVisibility.MASKED, confidence, lower_visible, eye_visible,
                yaw, pitch, reasons,
            )

    if abs(yaw) > max_yaw or abs(pitch) > max_pitch:
        reasons.append(
            f"pose beyond the usable range (yaw {yaw:.0f} deg, pitch {pitch:.0f} deg)"
        )
        return VisibilityReport(
            FaceVisibility.PARTIAL_FACE, 0.7, lower_visible, eye_visible,
            yaw, pitch, reasons,
        )

    confidence = float(np.clip(min(eye_visible, lower_visible) * 2.0, 0.4, 0.99))
    return VisibilityReport(
        FaceVisibility.FULL_FACE, confidence, lower_visible, eye_visible,
        yaw, pitch, reasons or ["eyes, nose and mouth regions all visible"],
    )
