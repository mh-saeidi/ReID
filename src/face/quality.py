"""Face quality assessment.

Quality is a *continuous weight on evidence*, not a gate. The distinction
matters: a small, blurry, half-turned face still carries some identity
information, and throwing it away entirely loses recall. Treating it as equal
to a clean frontal face loses precision. Weighting it is the only option that
does neither.

The score combines signals that each independently predict how reliable the
resulting embedding will be:

``resolution``   inter-ocular distance in pixels -- the single strongest
                 predictor, because everything downstream is resampled to
                 112x112 and detail that was never captured cannot be restored.
``sharpness``    variance of Laplacian, normalised for face size.
``exposure``     distance from mid-grey, penalising crushed shadows and clipping.
``pose``         frontality from the landmark geometry.
``landmarks``    detector confidence plus landmark self-consistency.
``visibility``   how much of the face is actually present.

The weights were chosen to reflect measured sensitivity on this system's own
data (resolution dominates; see docs), not copied from a paper. They are
configurable, and the composite is reported alongside its components so a bad
score is always explainable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

from src.face.types import FaceDetection
from src.face.visibility import FaceVisibility, VisibilityReport, estimate_pose


@dataclass(frozen=True, slots=True)
class QualityWeights:
    """Relative contribution of each signal to the composite score."""

    resolution: float = 0.34
    sharpness: float = 0.20
    exposure: float = 0.12
    pose: float = 0.18
    landmarks: float = 0.08
    visibility: float = 0.08

    def normalised(self) -> QualityWeights:
        total = sum(asdict(self).values()) or 1.0
        return QualityWeights(**{k: v / total for k, v in asdict(self).items()})


@dataclass(slots=True)
class FaceQualityScore:
    """A face's quality, component by component.

    Every field is in [0, 1] and higher is better, so the composite is
    interpretable and any single low component explains the result.
    """

    overall: float
    resolution: float
    sharpness: float
    exposure: float
    pose: float
    landmarks: float
    visibility: float

    interocular_px: float = 0.0
    laplacian_variance: float = 0.0
    mean_intensity: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    face_visibility: FaceVisibility = FaceVisibility.FULL_FACE
    notes: list[str] = field(default_factory=list)

    @property
    def weakest_component(self) -> str:
        components = {
            "resolution": self.resolution, "sharpness": self.sharpness,
            "exposure": self.exposure, "pose": self.pose,
            "landmarks": self.landmarks, "visibility": self.visibility,
        }
        return min(components, key=lambda key: components[key])

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "components": {
                "resolution": round(self.resolution, 4),
                "sharpness": round(self.sharpness, 4),
                "exposure": round(self.exposure, 4),
                "pose": round(self.pose, 4),
                "landmarks": round(self.landmarks, 4),
                "visibility": round(self.visibility, 4),
            },
            "measurements": {
                "interocular_px": round(self.interocular_px, 1),
                "laplacian_variance": round(self.laplacian_variance, 1),
                "mean_intensity": round(self.mean_intensity, 1),
                "yaw_deg": round(self.yaw_deg, 1),
                "pitch_deg": round(self.pitch_deg, 1),
                "roll_deg": round(self.roll_deg, 1),
            },
            "visibility": self.face_visibility.value,
            "weakest_component": self.weakest_component,
            "notes": list(self.notes),
        }


# Inter-ocular distance at which resolution stops limiting the embedding.
# Below ~24 px the face occupies too few pixels to survive the 112x112 resample.
_IOD_FLOOR_PX = 12.0
_IOD_SATURATION_PX = 55.0

_VISIBILITY_SCORE = {
    FaceVisibility.FULL_FACE: 1.0,
    FaceVisibility.PARTIAL_FACE: 0.55,
    FaceVisibility.MASKED: 0.45,
    FaceVisibility.HEAVILY_OCCLUDED: 0.10,
    FaceVisibility.NO_USABLE_FACE: 0.0,
}


def _resolution_score(interocular: float) -> float:
    """Saturating ramp: more pixels help until the model stops caring."""
    if interocular <= _IOD_FLOOR_PX:
        return 0.0
    span = _IOD_SATURATION_PX - _IOD_FLOOR_PX
    return float(np.clip((interocular - _IOD_FLOOR_PX) / span, 0.0, 1.0))


def _sharpness_score(chip: np.ndarray) -> tuple[float, float]:
    """Laplacian variance on the aligned chip, mapped to [0, 1]."""
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY) if chip.ndim == 3 else chip
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Measured on aligned 112x112 chips: a clean face sits well above 120,
    # a visibly blurred one below 40.
    return float(np.clip(variance / 140.0, 0.0, 1.0)), variance


def _exposure_score(chip: np.ndarray) -> tuple[float, float]:
    """Penalise both crushed shadows and clipped highlights."""
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY) if chip.ndim == 3 else chip
    mean = float(gray.mean())
    # Triangular around mid-grey, plus an explicit clipping penalty.
    centred = 1.0 - abs(mean - 128.0) / 128.0
    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    return float(np.clip(centred * (1.0 - clipped), 0.0, 1.0)), mean


def _pose_score(yaw: float, pitch: float, roll: float) -> float:
    """Frontality. Yaw costs most; roll costs least because alignment fixes it."""
    yaw_penalty = min(abs(yaw) / 50.0, 1.0)
    pitch_penalty = min(abs(pitch) / 40.0, 1.0)
    roll_penalty = min(abs(roll) / 60.0, 1.0)
    combined = 0.55 * yaw_penalty + 0.30 * pitch_penalty + 0.15 * roll_penalty
    return float(np.clip(1.0 - combined, 0.0, 1.0))


def _landmark_score(face: FaceDetection) -> float:
    """Detector confidence, tempered by landmark self-consistency.

    A plausible five-point set has the eyes above the mouth and the nose
    between them. A set that violates that is geometrically impossible and the
    alignment derived from it will be wrong, however confident the detector was.
    """
    confidence = float(np.clip(face.score, 0.0, 1.0))
    if not face.has_landmarks:
        return confidence * 0.4

    points = np.asarray(face.landmarks, dtype=np.float32)
    eye_y = (points[0][1] + points[1][1]) / 2.0
    mouth_y = (points[3][1] + points[4][1]) / 2.0
    nose_y = points[2][1]

    consistent = eye_y < nose_y < mouth_y
    interocular = float(np.linalg.norm(points[1] - points[0]))
    eye_to_mouth = abs(mouth_y - eye_y)
    # For a real face these are the same order of magnitude.
    proportion_ok = 0.5 < (eye_to_mouth / max(interocular, 1e-3)) < 2.5

    penalty = 1.0
    if not consistent:
        penalty *= 0.35
    if not proportion_ok:
        penalty *= 0.6
    return float(np.clip(confidence * penalty, 0.0, 1.0))


def _roll_from_landmarks(landmarks: np.ndarray | None) -> float:
    if landmarks is None or len(landmarks) < 2:
        return 0.0
    right_eye, left_eye = landmarks[0], landmarks[1]
    return float(np.degrees(np.arctan2(left_eye[1] - right_eye[1],
                                       left_eye[0] - right_eye[0])))


def assess_face_quality(
    chip: np.ndarray,
    face: FaceDetection,
    visibility: VisibilityReport | None = None,
    weights: QualityWeights | None = None,
) -> FaceQualityScore:
    """Score an aligned face chip.

    Args:
        chip: The aligned 112x112 image that will be embedded.
        face: The detection it came from, in frame coordinates.
        visibility: Occlusion analysis; assumed full-face when omitted.
        weights: Component weighting; the default is used when omitted.
    """
    weights = (weights or QualityWeights()).normalised()
    notes: list[str] = []

    interocular = face.eye_distance
    resolution = _resolution_score(interocular)
    if resolution < 0.25:
        notes.append(f"low resolution: inter-ocular distance {interocular:.0f}px")

    sharpness, laplacian = _sharpness_score(chip)
    if sharpness < 0.3:
        notes.append(f"soft or blurred (Laplacian variance {laplacian:.0f})")

    exposure, mean_intensity = _exposure_score(chip)
    if exposure < 0.4:
        notes.append(f"poor exposure (mean intensity {mean_intensity:.0f})")

    if face.has_landmarks:
        yaw, pitch = estimate_pose(face.landmarks)
    else:
        yaw = pitch = 0.0
    roll = _roll_from_landmarks(face.landmarks)
    pose = _pose_score(yaw, pitch, roll)
    if pose < 0.5:
        notes.append(f"non-frontal (yaw {yaw:.0f} deg, pitch {pitch:.0f} deg)")

    landmarks = _landmark_score(face)
    if landmarks < 0.5:
        notes.append("weak or geometrically inconsistent landmarks")

    face_visibility = visibility.visibility if visibility else FaceVisibility.FULL_FACE
    visibility_score = _VISIBILITY_SCORE.get(face_visibility, 0.5)
    if face_visibility is not FaceVisibility.FULL_FACE:
        notes.append(f"visibility: {face_visibility.value}")

    overall = (
        weights.resolution * resolution
        + weights.sharpness * sharpness
        + weights.exposure * exposure
        + weights.pose * pose
        + weights.landmarks * landmarks
        + weights.visibility * visibility_score
    )

    return FaceQualityScore(
        overall=float(np.clip(overall, 0.0, 1.0)),
        resolution=resolution,
        sharpness=sharpness,
        exposure=exposure,
        pose=pose,
        landmarks=landmarks,
        visibility=visibility_score,
        interocular_px=interocular,
        laplacian_variance=laplacian,
        mean_intensity=mean_intensity,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        face_visibility=face_visibility,
        notes=notes,
    )
