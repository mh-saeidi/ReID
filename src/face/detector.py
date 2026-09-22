"""Face detector interface and the YuNet implementation.

The person detector (YOLO26) stays exactly where it was: it finds people and
drives tracking. The face detector runs *inside* each person region and
supplies the landmarks that alignment needs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import ModelLoadError
from src.core.types import BBox
from src.face.types import FaceDetection, FaceDetectorInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FaceDetector(ABC):
    """Locates faces and their five alignment landmarks."""

    @abstractmethod
    def load(self) -> FaceDetectorInfo:
        """Load the model. Idempotent."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR image, in that image's coordinates."""

    @property
    @abstractmethod
    def info(self) -> FaceDetectorInfo: ...

    def close(self) -> None:
        """Release resources. Safe to call multiple times."""

    def detect_best(self, image: np.ndarray) -> FaceDetection | None:
        """Highest-scoring face, or ``None``."""
        faces = self.detect(image)
        return max(faces, key=lambda f: f.score) if faces else None


class YuNetFaceDetector(FaceDetector):
    """OpenCV YuNet -- a small, fast, landmark-producing face detector.

    Ships with OpenCV itself (``cv2.FaceDetectorYN``), so it adds one 230 KB
    model file and no new Python dependency.

    Args:
        model_path: Path to ``face_detection_yunet_*.onnx``.
        confidence: Minimum detection score.
        nms_threshold: NMS IoU threshold.
        top_k: Candidates kept before NMS.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 500,
    ) -> None:
        self._model_path = str(model_path)
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._detector = None
        self._info: FaceDetectorInfo | None = None
        self._input_size: tuple[int, int] = (0, 0)

    def load(self) -> FaceDetectorInfo:
        if self._info is not None:
            return self._info
        path = Path(self._model_path)
        if not path.exists():
            raise ModelLoadError(
                f"YuNet face detector model not found: {path}. "
                "Download it with: python scripts/fetch_face_models.py"
            )
        started = time.perf_counter()
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), self._confidence, self._nms_threshold, self._top_k
            )
        except cv2.error as exc:
            raise ModelLoadError(
                f"cannot load the YuNet face detector '{path}': {exc}"
            ) from exc
        self._input_size = (320, 320)
        elapsed = time.perf_counter() - started
        self._info = FaceDetectorInfo(
            name=path.stem,
            model_path=str(path),
            backend="opencv.yunet",
            device="cpu",
            input_size=(320, 320),
            load_time_s=elapsed,
            metadata={"confidence": self._confidence, "nms": self._nms_threshold},
        )
        logger.info(
            "Face detector loaded",
            extra={
                "model": path.stem,
                "backend": "opencv.yunet",
                "conf": self._confidence,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        if self._detector is None:
            self.load()
        height, width = image.shape[:2]
        if height < 8 or width < 8:
            return []
        if (width, height) != self._input_size:
            # YuNet is fully convolutional; matching the input size to the image
            # avoids a resize and keeps small faces at their native resolution.
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        try:
            _, raw = self._detector.detect(image)
        except cv2.error as exc:  # pragma: no cover - malformed input
            logger.debug("YuNet detection failed: %s", exc)
            return []
        if raw is None:
            return []

        faces: list[FaceDetection] = []
        for row in raw:
            x, y, w, h = (float(v) for v in row[:4])
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            faces.append(
                FaceDetection(
                    bbox=BBox(x, y, x + w, y + h),
                    score=float(row[-1]),
                    landmarks=landmarks,
                )
            )
        return faces

    @property
    def info(self) -> FaceDetectorInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        self._detector = None
