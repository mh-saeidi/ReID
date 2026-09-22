"""SCRFD face detector.

SCRFD is an anchor-based single-stage detector trained on WIDER FACE. It is
used here in preference to YuNet for two reasons that matter to this system:

* it holds up far better on the cases this deployment actually sees --
  non-frontal heads, small faces at a distance, and partially covered faces --
  where YuNet's recall collapses;
* it emits the same five landmarks (eyes, nose, mouth corners), so the
  alignment stage downstream is unchanged.

The decode is written out rather than pulled from a framework because the
output layout is unusual: three FPN strides, each producing a score map, a
distance-encoded box map and a distance-encoded landmark map, all flattened.
Getting the anchor ordering wrong produces detections that look plausible and
are subtly misplaced, which then quietly degrades alignment and every embedding
after it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.exceptions import BackendUnavailableError, ModelLoadError
from src.core.types import BBox
from src.face.detector import FaceDetector
from src.face.types import FaceDetection, FaceDetectorInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

# SCRFD's three FPN levels. Each stride has two anchors per location.
_STRIDES = (8, 16, 32)
_ANCHORS_PER_LOCATION = 2
_INPUT_MEAN = 127.5
_INPUT_STD = 128.0


def _distance_to_boxes(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + (left, top, right, bottom) distances -> xyxy."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_landmarks(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + per-point offsets -> (N, 5, 2) landmarks."""
    count = distance.shape[1] // 2
    xs = points[:, 0:1] + distance[:, 0::2]
    ys = points[:, 1:2] + distance[:, 1::2]
    return np.stack([xs, ys], axis=-1).reshape(-1, count, 2)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Standard greedy IoU suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        xx1 = np.maximum(x1[best], x1[order[1:]])
        yy1 = np.maximum(y1[best], y1[order[1:]])
        xx2 = np.minimum(x2[best], x2[order[1:]])
        yy2 = np.minimum(y2[best], y2[order[1:]])
        overlap = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = overlap / (areas[best] + areas[order[1:]] - overlap)
        order = order[1:][iou <= threshold]
    return keep


class SCRFDFaceDetector(FaceDetector):
    """SCRFD through ONNX Runtime.

    Args:
        model_path: ``scrfd_*.onnx`` (InsightFace ``det_10g`` and friends).
        confidence: Minimum detection score.
        nms_threshold: IoU threshold for suppression.
        input_size: Network input as ``(width, height)``. Larger finds smaller
            faces at proportionally higher cost.
        device: Resolved compute device, used to pick the execution provider.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        device: Any = None,
        max_faces: int = 100,
    ) -> None:
        self._model_path = str(model_path)
        self._confidence = confidence
        self._nms_threshold = nms_threshold
        self._input_size = input_size
        self._device = device
        self._max_faces = max_faces
        self._session: Any = None
        self._input_name = ""
        self._output_names: list[str] = []
        self._info: FaceDetectorInfo | None = None
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    # ---------------------------------------------------------------- loading
    def load(self) -> FaceDetectorInfo:
        if self._info is not None:
            return self._info
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                "onnxruntime is required for the SCRFD face detector"
            ) from exc

        path = Path(self._model_path)
        if not path.exists():
            raise ModelLoadError(
                f"SCRFD model not found: {path}. "
                "Fetch it with: python scripts/fetch_face_models.py"
            )

        providers = ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if getattr(self._device, "is_cuda", False) and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise ModelLoadError(f"cannot load SCRFD model '{path}': {exc}") from exc

        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]
        if len(self._output_names) != 9:
            raise ModelLoadError(
                f"{path.name} exposes {len(self._output_names)} outputs; this "
                "decoder expects the 9-output SCRFD layout (3 strides x "
                "score/bbox/landmark)"
            )

        elapsed = time.perf_counter() - started
        self._info = FaceDetectorInfo(
            name=path.stem,
            model_path=str(path),
            backend="onnxruntime.scrfd",
            device=session.get_providers()[0],
            input_size=self._input_size,
            load_time_s=elapsed,
            metadata={"confidence": self._confidence, "nms": self._nms_threshold},
        )
        logger.info(
            "Face detector loaded",
            extra={
                "model": path.stem,
                "backend": "onnxruntime.scrfd",
                "input": f"{self._input_size[0]}x{self._input_size[1]}",
                "conf": self._confidence,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    # -------------------------------------------------------------- anchors
    def _anchor_centres(self, height: int, width: int, stride: int) -> np.ndarray:
        """Anchor centres for one FPN level, cached per geometry.

        The ordering has to match the network's flattening exactly: rows vary
        slowest, and the two anchors at a location are adjacent.
        """
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        ys, xs = np.mgrid[:height, :width]
        centres = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
        centres = centres.reshape(-1, 2)
        if _ANCHORS_PER_LOCATION > 1:
            centres = np.repeat(centres, _ANCHORS_PER_LOCATION, axis=0)
        self._anchor_cache[key] = centres
        return centres

    # ------------------------------------------------------------- inference
    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        if self._session is None:
            self.load()
        height, width = image.shape[:2]
        if height < 16 or width < 16:
            return []

        target_w, target_h = self._input_size
        # Letterbox: preserve aspect so landmark geometry stays undistorted.
        scale = min(target_w / width, target_h / height)
        resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / _INPUT_STD, (target_w, target_h),
            (_INPUT_MEAN, _INPUT_MEAN, _INPUT_MEAN), swapRB=True,
        )
        outputs = self._session.run(self._output_names, {self._input_name: blob})

        boxes, scores, landmarks = self._decode(outputs, target_h, target_w)
        if boxes.size == 0:
            return []

        keep = _nms(boxes, scores, self._nms_threshold)[: self._max_faces]
        faces: list[FaceDetection] = []
        for index in keep:
            box = boxes[index] / scale
            points = landmarks[index] / scale
            faces.append(
                FaceDetection(
                    bbox=BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(scores[index]),
                    landmarks=points.astype(np.float32),
                )
            )
        return faces

    def _decode(
        self, outputs: Sequence[np.ndarray], height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Turn the nine flattened maps into boxes, scores and landmarks."""
        all_boxes, all_scores, all_landmarks = [], [], []
        levels = len(_STRIDES)
        for level, stride in enumerate(_STRIDES):
            score = outputs[level].reshape(-1)
            bbox = outputs[levels + level].reshape(-1, 4) * stride
            landmark = outputs[levels * 2 + level].reshape(-1, 10) * stride

            positive = np.nonzero(score >= self._confidence)[0]
            if positive.size == 0:
                continue
            centres = self._anchor_centres(height // stride, width // stride, stride)
            all_boxes.append(_distance_to_boxes(centres[positive], bbox[positive]))
            all_landmarks.append(_distance_to_landmarks(centres[positive], landmark[positive]))
            all_scores.append(score[positive])

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 5, 2))
        return (
            np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_landmarks, axis=0),
        )

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[FaceDetection]]:
        """Detect on several images. SCRFD's graph is fixed-batch, so this
        loops; the method exists so callers need not special-case it."""
        return [self.detect(image) for image in images]

    @property
    def info(self) -> FaceDetectorInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        self._session = None
