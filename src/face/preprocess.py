"""Face extraction and alignment as a :class:`CropPreprocessor`.

This is the piece that makes identity clothing-independent. Where the
person-appearance preprocessor hands the encoder the whole body, this one:

1. takes the person box from YOLO26,
2. searches the head region for a face,
3. checks the face is big enough to be worth recognising,
4. warps it onto the canonical 112x112 template using the five landmarks,
5. hands the encoder *only* the face.

No pixel below the neck reaches the embedding, so a change of clothing cannot
change the identity decision.

When no usable face is found the result says so explicitly. It never falls back
to body appearance -- silently switching modality would reintroduce exactly the
clothing dependence this mode exists to remove.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.config.schema import FaceSearchRegion, RecognitionMode
from src.core.types import BBox
from src.face.align import align_face, crop_face_fallback
from src.face.detector import FaceDetector
from src.face.types import FaceDetection, FaceQuality
from src.reid.preprocess import CropPreprocessor, CropResult, CropStatus
from src.utils.image import crop_bbox, measure_quality
from src.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.schema import AdaptiveSearchConfig
    from src.face.quality_gate import FaceQualityGate

logger = get_logger(__name__)


class FaceCropPreprocessor(CropPreprocessor):
    """Produces aligned face chips from person boxes.

    Args:
        detector: Face detector supplying boxes and five landmarks.
        chip_size: Output chip edge length (112 for ArcFace/SFace).
        min_face_size: Minimum face box short side, in pixels. Faces below this
            carry too little detail to identify reliably, and guessing from them
            is how false accepts happen.
        min_eye_distance: Minimum inter-ocular distance; a pose-robust
            resolution check that catches strongly profile faces.
        search_region: Where to look for the face relative to the person box.
        head_fraction: Fraction of the person box height searched by
            ``upper_body``.
        require_landmarks: Refuse faces without landmarks instead of falling
            back to an unaligned crop.
        min_blur_variance: Reject faces blurrier than this (0 disables).
    """

    def __init__(
        self,
        detector: FaceDetector,
        *,
        chip_size: int = 112,
        min_face_size: int = 40,
        min_eye_distance: float = 14.0,
        search_region: FaceSearchRegion = FaceSearchRegion.UPPER_BODY,
        head_fraction: float = 0.55,
        margin: float = 0.12,
        require_landmarks: bool = True,
        min_blur_variance: float = 0.0,
        quality_gate: FaceQualityGate | None = None,
        adaptive_search: AdaptiveSearchConfig | None = None,
    ) -> None:
        self._detector = detector
        self._chip_size = chip_size
        self._min_face_size = min_face_size
        self._min_eye_distance = min_eye_distance
        self._search_region = search_region
        self._head_fraction = head_fraction
        self._margin = margin
        self._require_landmarks = require_landmarks
        self._min_blur_variance = min_blur_variance
        self._gate = quality_gate
        self._adaptive = adaptive_search
        self._frame_cache: tuple[int, list[FaceDetection]] | None = None
        # Telemetry: which search strategy actually ran, so the adaptive policy
        # can be evaluated rather than assumed.
        self.search_stats: dict[str, int] = {"upper_body": 0, "full_box": 0, "frame": 0}
        self._active_region = search_region
        self._candidate_count = 0

    @property
    def mode(self) -> RecognitionMode:
        return RecognitionMode.FACE

    @property
    def detector(self) -> FaceDetector:
        return self._detector

    @property
    def active_search_region(self) -> FaceSearchRegion:
        """The strategy in use for the current frame (telemetry)."""
        return self._active_region

    def begin_frame(self, candidate_count: int) -> None:
        """Announce how many people will be searched this frame.

        Per-person ROI search costs one detector call each but every call is
        small; whole-frame search is a single larger call. Which wins depends on
        person count and frame size, so the switch is measured rather than
        assumed -- and disabled by default.
        """
        self._candidate_count = candidate_count
        if self._adaptive is not None and self._adaptive.enabled:
            self._active_region = (
                self._adaptive.crowded_region
                if candidate_count >= self._adaptive.crowd_threshold
                else self._adaptive.sparse_region
            )
        else:
            self._active_region = self._search_region

    # ------------------------------------------------------------------ search
    def _search_box(self, bbox: BBox, width: int, height: int) -> BBox:
        """Region of the frame searched for this person's face."""
        if self._active_region is FaceSearchRegion.FULL_BOX:
            box = bbox
        else:
            # Heads sit in the top portion of a person box. Searching only there
            # is both faster and less prone to picking up a bystander's face
            # that overlaps the lower half of the box.
            box = BBox(bbox.x1, bbox.y1, bbox.x2, bbox.y1 + bbox.height * self._head_fraction)
        pad_x = box.width * self._margin
        pad_y = box.height * self._margin
        return BBox(box.x1 - pad_x, box.y1 - pad_y, box.x2 + pad_x, box.y2 + pad_y).clip(
            width, height
        )

    def _detect_in_frame(self, image: np.ndarray) -> list[FaceDetection]:
        """One detection pass per frame, cached by object identity."""
        key = id(image)
        if self._frame_cache is not None and self._frame_cache[0] == key:
            return self._frame_cache[1]
        faces = self._detector.detect(image)
        self._frame_cache = (key, faces)
        return faces

    def _faces_for(self, image: np.ndarray, bbox: BBox) -> tuple[list[FaceDetection], BBox | None]:
        """Faces for this person, plus the region they were found in.

        The region is returned so alignment can warp from the small patch
        instead of the full frame -- a 1280x720 warpAffine per face is pure
        waste when the face occupies a few hundred pixels.
        """
        height, width = image.shape[:2]
        if self._active_region is FaceSearchRegion.FRAME:
            # Detect once over the whole frame, then keep faces whose centre
            # falls inside this person's box.
            self.search_stats["frame"] += 1
            return (
                [
                    face
                    for face in self._detect_in_frame(image)
                    if _center_inside(face.bbox, bbox)
                ],
                None,
            )

        region = self._search_box(bbox, width, height)
        patch = crop_bbox(image, region, min_size=16)
        if patch is None:
            return [], None
        key = "full_box" if self._active_region is FaceSearchRegion.FULL_BOX else "upper_body"
        self.search_stats[key] += 1
        # Faces are returned in patch coordinates; they are offset into frame
        # coordinates for reporting, but alignment uses the patch directly.
        return (
            [face.offset(region.x1, region.y1) for face in self._detector.detect(patch)],
            region,
        )

    def _select(self, faces: list[FaceDetection], bbox: BBox) -> FaceDetection | None:
        """Pick this person's face: the largest one near the top of the box.

        Size is preferred over raw score because a bigger face yields a better
        embedding, and a bystander leaning into the box is usually smaller.
        """
        if not faces:
            return None
        inside = [f for f in faces if _center_inside(f.bbox, bbox, tolerance=0.2)]
        return max(inside or faces, key=lambda f: f.bbox.area)

    # --------------------------------------------------------------- interface
    def extract(self, image: np.ndarray, bbox: BBox) -> CropResult:
        height, width = image.shape[:2]
        if bbox.width < 8 or bbox.height < 8:
            return CropResult(None, CropStatus.DEGENERATE_BOX, detail="person box too small")

        faces, region = self._faces_for(image, bbox)
        face = self._select(faces, bbox)
        if face is None:
            return CropResult(
                None,
                CropStatus.NO_FACE,
                detail="no face detected in the person region",
            )

        # Geometry gate first: rejecting here skips both the warp and the
        # encoder, which is the entire reason it runs before them.
        if self._gate is not None and self._gate.enabled:
            verdict = self._gate.check_detection(face)
            if not verdict.accepted:
                face.quality = verdict.quality
                face.detail = verdict.reason
                return CropResult(None, verdict.status, face=face, detail=verdict.reason)
        else:
            legacy = self._legacy_geometry_check(face)
            if legacy is not None:
                return legacy

        chip = self._align(image, face, region)
        if chip is None:
            return CropResult(
                None, CropStatus.NO_FACE, face=face, detail="face alignment failed"
            )

        if self._gate is not None and self._gate.enabled:
            verdict = self._gate.check_chip(chip, face)
            if not verdict.accepted:
                face.quality = verdict.quality
                face.detail = verdict.reason
                return CropResult(None, verdict.status, face=face, detail=verdict.reason)
        elif self._min_blur_variance > 0.0:
            sharpness = measure_quality(chip).blur_variance
            if sharpness < self._min_blur_variance:
                face.quality = FaceQuality.BLURRY
                return CropResult(
                    None,
                    CropStatus.LOW_QUALITY,
                    face=face,
                    detail=f"face is blurry (sharpness {sharpness:.1f})",
                )

        return CropResult(chip, CropStatus.OK, face=face)

    def _legacy_geometry_check(self, face: FaceDetection) -> CropResult | None:
        """The pre-gate geometry rules, kept for when gating is disabled."""
        if face.size < self._min_face_size:
            face.quality = FaceQuality.TOO_SMALL
            return CropResult(
                None,
                CropStatus.FACE_TOO_SMALL,
                face=face,
                detail=(
                    f"face is {face.size:.0f}px (minimum {self._min_face_size}px); "
                    "too little detail to identify reliably"
                ),
            )
        if face.has_landmarks and face.eye_distance < self._min_eye_distance:
            face.quality = FaceQuality.EXTREME_POSE
            return CropResult(
                None,
                CropStatus.FACE_TOO_SMALL,
                face=face,
                detail=(
                    f"inter-ocular distance {face.eye_distance:.0f}px is below "
                    f"{self._min_eye_distance:.0f}px (face turned away or too distant)"
                ),
            )
        return None

    def _align(
        self,
        image: np.ndarray,
        face: FaceDetection,
        region: BBox | None = None,
    ) -> np.ndarray | None:
        """Warp the face onto the 112x112 template.

        When the face came from a person ROI, the warp runs on that small patch
        rather than the full frame. ``cv2.warpAffine`` costs time proportional
        to the *source* it samples from, so warping a 1280x720 frame to produce
        a 112x112 chip is work spent on pixels that are thrown away. The
        landmarks are shifted into patch coordinates first, which makes the
        output numerically identical.
        """
        source = image
        landmarks = face.landmarks
        bbox = face.bbox
        if region is not None:
            patch = crop_bbox(image, region, min_size=16)
            if patch is not None:
                shifted_bbox = BBox(
                    bbox.x1 - region.x1, bbox.y1 - region.y1,
                    bbox.x2 - region.x1, bbox.y2 - region.y1,
                )
                shifted_landmarks = (
                    landmarks - np.array([region.x1, region.y1], dtype=np.float32)
                    if landmarks is not None
                    else None
                )
                # Only take the fast path when the face genuinely lies inside
                # the patch. A detector that reports coordinates in a different
                # frame of reference would otherwise silently warp garbage.
                if _within(shifted_bbox, shifted_landmarks, patch.shape[1], patch.shape[0]):
                    source = patch
                    bbox = shifted_bbox
                    landmarks = shifted_landmarks

        if landmarks is not None and len(landmarks) >= 5:
            try:
                return align_face(source, landmarks, self._chip_size, bbox)
            except Exception as exc:  # noqa: BLE001 - cv2 raises broadly
                logger.debug("Face alignment failed, falling back to a crop: %s", exc)
                if self._require_landmarks:
                    return None
        elif self._require_landmarks:
            logger.debug("Face has no landmarks and require_landmarks is set")
            return None
        return crop_face_fallback(source, bbox, self._chip_size)

    def prepare(self, crop: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """Chips are already at the encoder's geometry; resize only if asked."""
        if crop.shape[0] == size_hw[0] and crop.shape[1] == size_hw[1]:
            return crop
        import cv2  # noqa: PLC0415

        return cv2.resize(crop, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)


def _within(
    bbox: BBox, landmarks: np.ndarray | None, width: int, height: int, slack: float = 2.0
) -> bool:
    """Does this face fit inside a patch of the given size?"""
    if bbox.x1 < -slack or bbox.y1 < -slack:
        return False
    if bbox.x2 > width + slack or bbox.y2 > height + slack:
        return False
    if landmarks is not None and len(landmarks):
        xs, ys = landmarks[:, 0], landmarks[:, 1]
        if xs.min() < -slack or ys.min() < -slack:
            return False
        if xs.max() > width + slack or ys.max() > height + slack:
            return False
    return True


def _center_inside(inner: BBox, outer: BBox, tolerance: float = 0.0) -> bool:
    """Is ``inner``'s centre inside ``outer`` (optionally expanded)?"""
    cx, cy = inner.center
    pad_x = outer.width * tolerance
    pad_y = outer.height * tolerance
    return (
        outer.x1 - pad_x <= cx <= outer.x2 + pad_x
        and outer.y1 - pad_y <= cy <= outer.y2 + pad_y
    )
