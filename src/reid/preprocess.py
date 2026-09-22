"""Crop preprocessing: the seam between detection and embedding.

Two modes exist and they are always explicit -- the system never switches
between them silently, because they answer the identity question from
completely different evidence:

``person_reid``
    Whole-body appearance. Fast, works at a distance and from behind, but the
    embedding is dominated by clothing, so a change of outfit breaks it.

``face``
    Facial appearance only. Invariant to clothing, but needs a visible face at
    sufficient resolution -- it cannot identify someone from behind.

Both return a :class:`CropResult`, so the pipeline can distinguish "nothing
usable here" from "no face visible", and report the reason instead of guessing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src.config.schema import AppConfig, RecognitionMode
from src.core.exceptions import ConfigurationError
from src.core.types import BBox
from src.utils.image import crop_bbox, letterbox, resize_crop

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.face.types import FaceDetection


class CropStatus(str, Enum):
    """Why a crop was or was not produced for a detection."""

    OK = "ok"
    DEGENERATE_BOX = "degenerate_box"
    """The person box is too small or lies outside the frame."""

    NO_FACE = "no_face"
    """Face mode: nobody's face is visible in this person region."""

    FACE_TOO_SMALL = "face_too_small"
    """Face mode: a face was found but carries too little detail to identify."""

    LOW_QUALITY = "low_quality"
    """The region was found but is too blurry or degraded to embed."""


@dataclass(slots=True)
class CropResult:
    """The image handed to the encoder, plus why it may be missing."""

    image: np.ndarray | None
    status: CropStatus = CropStatus.OK
    face: FaceDetection | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.image is not None and self.status is CropStatus.OK

    def __bool__(self) -> bool:
        return self.ok


class CropPreprocessor(ABC):
    """Extracts and normalises the image region handed to the encoder."""

    @property
    @abstractmethod
    def mode(self) -> RecognitionMode: ...

    @abstractmethod
    def extract(self, image: np.ndarray, bbox: BBox) -> CropResult:
        """Cut the region of interest out of a frame."""

    @abstractmethod
    def prepare(self, crop: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        """Resize a crop to the encoder's input geometry (still BGR uint8)."""

    @property
    def requires_visible_face(self) -> bool:
        """True when a detection without a usable face cannot be identified."""
        return self.mode is RecognitionMode.FACE

    def extract_many(
        self, image: np.ndarray, boxes: Sequence[BBox]
    ) -> tuple[list[np.ndarray], list[int], list[CropResult]]:
        """Extract several crops, returning them with their original indices."""
        crops: list[np.ndarray] = []
        indices: list[int] = []
        results: list[CropResult] = []
        for index, bbox in enumerate(boxes):
            result = self.extract(image, bbox)
            results.append(result)
            if result.ok:
                crops.append(result.image)
                indices.append(index)
        return crops, indices, results


class PersonCropPreprocessor(CropPreprocessor):
    """Whole-body appearance crops.

    Person ReID encoders are trained on tall, aspect-distorted crops (the
    classic 256x128 convention), so a plain resize is used by default. Set
    ``preserve_aspect`` to letterbox instead when the encoder expects that.
    """

    def __init__(
        self,
        *,
        padding: float = 0.0,
        min_crop_size: int = 16,
        preserve_aspect: bool = False,
    ) -> None:
        self._padding = padding
        self._min_crop_size = min_crop_size
        self._preserve_aspect = preserve_aspect

    @property
    def mode(self) -> RecognitionMode:
        return RecognitionMode.PERSON_REID

    def extract(self, image: np.ndarray, bbox: BBox) -> CropResult:
        crop = crop_bbox(image, bbox, padding=self._padding, min_size=self._min_crop_size)
        if crop is None:
            return CropResult(
                None,
                CropStatus.DEGENERATE_BOX,
                detail=f"person region {bbox.to_int()} is too small to crop",
            )
        return CropResult(crop, CropStatus.OK)

    def prepare(self, crop: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        if self._preserve_aspect:
            return letterbox(crop, size_hw)
        return resize_crop(crop, size_hw)


def build_preprocessor(config: AppConfig, face_detector=None) -> CropPreprocessor:
    """Create the preprocessor for the configured recognition mode.

    Args:
        config: Validated application configuration.
        face_detector: Required in ``face`` mode; supplied by the composition
            root so this module stays free of backend imports.
    """
    mode = config.recognition.mode
    if mode is RecognitionMode.PERSON_REID:
        return PersonCropPreprocessor(
            padding=config.reid.crop_padding,
            min_crop_size=config.reid.min_crop_size,
        )
    if mode is RecognitionMode.FACE:
        if face_detector is None:
            raise ConfigurationError(
                "recognition.mode='face' requires a face detector; "
                "build the engine through src.pipeline.engine.build_engine()"
            )
        from src.face.preprocess import FaceCropPreprocessor  # noqa: PLC0415
        from src.face.quality_gate import FaceQualityGate  # noqa: PLC0415

        face = config.face
        gate = FaceQualityGate(face) if face.runtime_quality.enabled else None
        return FaceCropPreprocessor(
            face_detector,
            chip_size=face.chip_size,
            min_face_size=face.min_face_size,
            min_eye_distance=face.min_eye_distance,
            search_region=face.search_region,
            head_fraction=face.head_fraction,
            margin=face.search_margin,
            require_landmarks=face.require_landmarks,
            min_blur_variance=face.min_blur_variance,
            quality_gate=gate,
            adaptive_search=face.adaptive_search,
        )
    raise ConfigurationError(  # pragma: no cover - enum is exhaustive
        f"recognition.mode '{mode.value}' has no preprocessor"
    )
