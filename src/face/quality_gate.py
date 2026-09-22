"""Runtime face quality gating.

A face that is too small, too blurry, badly lit or badly landmarked does not
merely produce a weak embedding -- it produces a *misleading* one. Degraded
inputs tend to collapse toward the centre of the embedding space, which puts
them closer to every gallery entry at once. That is the shape of a false accept.

So the gate runs before the encoder, and a rejection returns the existing
semantic states (``NO_FACE`` / ``FACE_TOO_SMALL`` / ``LOW_QUALITY``). It never
produces an identity, and it never downgrades to body appearance.

Every threshold defaults to the value already configured under ``face.*``, so
enabling this section changes no behaviour until something is overridden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config.schema import FaceConfig, RuntimeQualityConfig
from src.face.types import FaceDetection, FaceQuality
from src.reid.preprocess import CropStatus


@dataclass(frozen=True, slots=True)
class GateResult:
    """Whether a face may be embedded, and why not when it may not."""

    accepted: bool
    status: CropStatus = CropStatus.OK
    quality: FaceQuality = FaceQuality.OK
    reason: str = ""
    measured: dict[str, float] | None = None

    def __bool__(self) -> bool:
        return self.accepted


class FaceQualityGate:
    """Applies the configured runtime quality thresholds to a detected face.

    Args:
        config: The ``face`` section, used as the source of inherited defaults.
        runtime: The ``face.runtime_quality`` overrides.
    """

    def __init__(self, config: FaceConfig, runtime: RuntimeQualityConfig | None = None) -> None:
        runtime = runtime or config.runtime_quality
        self._enabled = runtime.enabled
        # Inheritance keeps one source of truth: an unset runtime threshold is
        # the corresponding face.* value, not a second independent default.
        self._min_size = _first(runtime.min_face_size, config.min_face_size)
        self._min_eye = _first(runtime.min_eye_distance, config.min_eye_distance)
        self._min_score = _first(runtime.min_detector_confidence, 0.0)
        self._min_sharpness = _first(runtime.min_sharpness, config.min_blur_variance)
        self._min_brightness = _first(runtime.min_brightness, 0.0)
        self._max_brightness = _first(runtime.max_brightness, 255.0)
        self._max_skew = runtime.max_landmark_skew
        self._require_landmarks = config.require_landmarks

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def needs_pixels(self) -> bool:
        """Do any enabled checks require the aligned chip?"""
        return self._enabled and (
            self._min_sharpness > 0.0
            or self._min_brightness > 0.0
            or self._max_brightness < 255.0
        )

    # ------------------------------------------------------------ geometry
    def check_detection(self, face: FaceDetection) -> GateResult:
        """Checks that need only the detection -- run before any warping.

        Rejecting here saves the alignment and the encoder pass entirely, which
        is the whole point of gating before rather than after.
        """
        if not self._enabled:
            return GateResult(True)

        measured = {
            "face_size": round(face.size, 1),
            "eye_distance": round(face.eye_distance, 1),
            "detector_confidence": round(face.score, 4),
        }

        if face.score < self._min_score:
            return GateResult(
                False,
                CropStatus.LOW_QUALITY,
                FaceQuality.LOW_CONFIDENCE,
                f"face detector confidence {face.score:.2f} is below "
                f"{self._min_score:.2f}",
                measured,
            )

        if face.size < self._min_size:
            return GateResult(
                False,
                CropStatus.FACE_TOO_SMALL,
                FaceQuality.TOO_SMALL,
                f"face is {face.size:.0f}px (minimum {self._min_size:.0f}px); "
                "too little detail to identify reliably",
                measured,
            )

        if face.has_landmarks:
            if face.eye_distance < self._min_eye:
                return GateResult(
                    False,
                    CropStatus.FACE_TOO_SMALL,
                    FaceQuality.EXTREME_POSE,
                    f"inter-ocular distance {face.eye_distance:.0f}px is below "
                    f"{self._min_eye:.0f}px (face turned away or too distant)",
                    measured,
                )
            if self._max_skew is not None:
                skew = _landmark_skew(face)
                measured["landmark_skew"] = round(skew, 3)
                if skew > self._max_skew:
                    return GateResult(
                        False,
                        CropStatus.LOW_QUALITY,
                        FaceQuality.EXTREME_POSE,
                        f"landmark skew {skew:.2f} exceeds {self._max_skew:.2f}; "
                        "extreme head roll or an unreliable landmark fit",
                        measured,
                    )
        elif self._require_landmarks:
            return GateResult(
                False,
                CropStatus.NO_FACE,
                FaceQuality.LOW_CONFIDENCE,
                "no facial landmarks, so the face cannot be aligned",
                measured,
            )

        return GateResult(True, measured=measured)

    # -------------------------------------------------------------- pixels
    def check_chip(self, chip: np.ndarray, face: FaceDetection) -> GateResult:
        """Checks that need the aligned chip (sharpness, exposure)."""
        if not self._enabled or not self.needs_pixels:
            return GateResult(True)

        import cv2  # noqa: PLC0415

        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY) if chip.ndim == 3 else chip
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        measured = {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
        }

        if self._min_sharpness > 0.0 and sharpness < self._min_sharpness:
            return GateResult(
                False,
                CropStatus.LOW_QUALITY,
                FaceQuality.BLURRY,
                f"face is blurry (sharpness {sharpness:.1f} < {self._min_sharpness:.1f})",
                measured,
            )
        if brightness < self._min_brightness:
            return GateResult(
                False,
                CropStatus.LOW_QUALITY,
                FaceQuality.LOW_CONFIDENCE,
                f"face is too dark (mean intensity {brightness:.1f})",
                measured,
            )
        if brightness > self._max_brightness:
            return GateResult(
                False,
                CropStatus.LOW_QUALITY,
                FaceQuality.LOW_CONFIDENCE,
                f"face is over-exposed (mean intensity {brightness:.1f})",
                measured,
            )
        return GateResult(True, measured=measured)


def _first(value: float | int | None, fallback: float | int) -> float:
    return float(fallback if value is None else value)


def _landmark_skew(face: FaceDetection) -> float:
    """Vertical eye offset relative to inter-ocular distance."""
    if not face.has_landmarks:
        return 0.0
    eyes = face.landmarks[:2]
    separation = float(np.linalg.norm(eyes[1] - eyes[0]))
    if separation < 1e-6:
        return float("inf")
    return abs(float(eyes[1][1] - eyes[0][1])) / separation
