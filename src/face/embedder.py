"""Face embedding backends.

All of them implement :class:`~src.reid.encoder.ReIDEncoder`, so the gallery,
matcher, pipeline and tracker are completely unaware that identity is now
coming from a face rather than from whole-body appearance. Swapping the mode
swaps one object in the composition root.

Inputs are aligned 112x112 BGR chips produced by :mod:`src.face.align`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import BackendUnavailableError, ModelLoadError
from src.reid.encoder import EncoderInfo, ReIDEncoder, l2_normalize
from src.utils.device import DeviceInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

CHIP_SIZE = 112


class SFaceEmbedder(ReIDEncoder):
    """OpenCV SFace (``cv2.FaceRecognizerSF``).

    Needs no dependency beyond OpenCV itself, which makes it the default. It
    produces 128-D features; cosine similarity between two faces of the same
    person is typically well above 0.4, and below 0.2 for different people --
    a far wider margin than whole-body appearance gives.

    Args:
        model_path: Path to ``face_recognition_sface_*.onnx``.
        device: Resolved compute device (SFace runs on CPU through OpenCV DNN).
        normalize: L2-normalise the output so cosine reduces to a dot product.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceInfo,
        *,
        normalize: bool = True,
        batch_size: int = 16,
    ) -> None:
        self._model_path = str(model_path)
        self._device = device
        self._normalize = normalize
        self._batch_size = max(1, batch_size)
        self._recognizer: Any | None = None
        self._info: EncoderInfo | None = None

    def load(self) -> EncoderInfo:
        if self._info is not None:
            return self._info
        import cv2  # noqa: PLC0415

        path = Path(self._model_path)
        if not path.exists():
            raise ModelLoadError(
                f"SFace recognition model not found: {path}. "
                "Download it with: python scripts/fetch_face_models.py"
            )
        started = time.perf_counter()
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(str(path), "")
        except cv2.error as exc:
            raise ModelLoadError(
                f"cannot load the SFace recognition model '{path}': {exc}"
            ) from exc

        probe = self._recognizer.feature(np.zeros((CHIP_SIZE, CHIP_SIZE, 3), dtype=np.uint8))
        dimension = int(np.asarray(probe).reshape(-1).shape[0])
        elapsed = time.perf_counter() - started

        self._info = EncoderInfo(
            name=path.stem,
            model_path=str(path),
            backend="opencv.sface",
            device="cpu",
            fp16=False,
            input_size=(CHIP_SIZE, CHIP_SIZE),
            embedding_dimension=dimension,
            load_time_s=elapsed,
            # cv2.FaceRecognizerSF.feature() takes one image at a time; there is
            # no batched entry point to call.
            max_batch_size=1,
            metadata={"aligned_input": True, "modality": "face"},
        )
        logger.info(
            "Face encoder loaded",
            extra={
                "model": path.stem,
                "backend": "opencv.sface",
                "dim": dimension,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if self._info is None:
            self.load()
        assert self._info is not None  # noqa: S101
        dimension = self._info.embedding_dimension
        out = np.zeros((len(crops), dimension), dtype=np.float32)

        for index, chip in enumerate(crops):
            if chip is None or not getattr(chip, "size", 0):
                continue
            prepared = self._ensure_chip(chip)
            feature = np.asarray(self._recognizer.feature(prepared), dtype=np.float32).reshape(-1)
            out[index] = feature
        return l2_normalize(out, axis=1) if self._normalize else out

    @staticmethod
    def _ensure_chip(chip: np.ndarray) -> np.ndarray:
        if chip.shape[0] == CHIP_SIZE and chip.shape[1] == CHIP_SIZE:
            return np.ascontiguousarray(chip)
        import cv2  # noqa: PLC0415

        return cv2.resize(chip, (CHIP_SIZE, CHIP_SIZE), interpolation=cv2.INTER_LINEAR)

    @property
    def info(self) -> EncoderInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        self._recognizer = None


class ArcFaceOnnxEmbedder(ReIDEncoder):
    """Any ArcFace-format ONNX recognition model, run through ONNX Runtime.

    Covers the InsightFace exports (``w600k_r50.onnx``, ``glintr100.onnx``, ...)
    and compatible third-party models. They take ``(N, 3, 112, 112)`` RGB input
    scaled to roughly ``[-1, 1]`` and return an unnormalised embedding.

    Args:
        model_path: Path to the ``.onnx`` recognition model.
        device: Resolved compute device (selects the execution provider).
        input_scale/input_mean: Preprocessing, matching InsightFace's convention
            of ``(pixel - 127.5) / 127.5``.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceInfo,
        *,
        normalize: bool = True,
        batch_size: int = 16,
        input_mean: float = 127.5,
        input_scale: float = 127.5,
    ) -> None:
        self._model_path = str(model_path)
        self._device = device
        self._normalize = normalize
        self._batch_size = max(1, batch_size)
        self._input_mean = input_mean
        self._input_scale = input_scale
        self._session: Any | None = None
        self._input_name = ""
        self._output_name = ""
        self._chip_size = CHIP_SIZE
        self._static_batch: int | None = None
        self._info: EncoderInfo | None = None

    def load(self) -> EncoderInfo:
        if self._info is not None:
            return self._info
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                "onnxruntime is required for the ArcFace ONNX face backend"
            ) from exc

        path = Path(self._model_path)
        if not path.exists():
            raise ModelLoadError(f"ArcFace ONNX model not found: {path}")

        providers = ["CPUExecutionProvider"]
        if self._device.is_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise ModelLoadError(f"cannot load the ArcFace model '{path}': {exc}") from exc

        self._session = session
        spec = session.get_inputs()[0]
        self._input_name = spec.name
        self._output_name = session.get_outputs()[0].name
        if len(spec.shape) == 4 and isinstance(spec.shape[2], int) and spec.shape[2] > 0:
            self._chip_size = int(spec.shape[2])

        # A graph exported with a fixed batch dimension cannot be batched, no
        # matter what the configuration requests. Feeding it a larger batch
        # makes ONNX Runtime fall back off its accelerated partition, which is
        # measurably *slower* than the equivalent sequential calls -- so this is
        # detected once and honoured rather than discovered per frame.
        self._static_batch = _static_batch_dim(spec.shape, session.get_outputs()[0].shape)
        if self._static_batch == 1:
            if self._batch_size > 1:
                logger.info(
                    "The face encoder has a fixed batch dimension of 1, so "
                    "recognition.batch has no effect for this model. Re-export "
                    "it with a dynamic batch axis to enable batching.",
                    extra={"model": path.stem},
                )
            self._batch_size = 1

        dummy = np.zeros((1, 3, self._chip_size, self._chip_size), dtype=np.float32)
        try:
            probe = session.run([self._output_name], {self._input_name: dummy})[0]
        except Exception as exc:
            raise ModelLoadError(
                f"the ArcFace model '{path}' failed its warm-up pass: {exc}"
            ) from exc
        dimension = int(np.asarray(probe).reshape(1, -1).shape[1])
        elapsed = time.perf_counter() - started

        self._info = EncoderInfo(
            name=path.stem,
            model_path=str(path),
            backend="onnxruntime.arcface",
            device=session.get_providers()[0],
            fp16=False,
            input_size=(self._chip_size, self._chip_size),
            embedding_dimension=dimension,
            load_time_s=elapsed,
            max_batch_size=self._static_batch or 0,
            metadata={
                "aligned_input": True,
                "modality": "face",
                "static_batch": self._static_batch,
            },
        )
        logger.info(
            "Face encoder loaded",
            extra={
                "model": path.stem,
                "backend": "onnxruntime.arcface",
                "dim": dimension,
                "batch": "fixed-1" if self._static_batch == 1 else "dynamic",
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    def _preprocess(self, chips: Sequence[np.ndarray]) -> np.ndarray:
        import cv2  # noqa: PLC0415

        size = self._chip_size
        batch = np.empty((len(chips), size, size, 3), dtype=np.float32)
        for index, chip in enumerate(chips):
            if chip.shape[0] != size or chip.shape[1] != size:
                chip = cv2.resize(chip, (size, size), interpolation=cv2.INTER_LINEAR)
            batch[index] = chip[:, :, ::-1]  # BGR -> RGB
        batch = (batch - self._input_mean) / self._input_scale
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if self._info is None:
            self.load()
        assert self._info is not None  # noqa: S101
        dimension = self._info.embedding_dimension
        out = np.zeros((len(crops), dimension), dtype=np.float32)

        valid = [i for i, c in enumerate(crops) if c is not None and getattr(c, "size", 0)]
        if not valid:
            return out

        chips = [crops[i] for i in valid]
        features: list[np.ndarray] = []
        for start in range(0, len(chips), self._batch_size):
            tensor = self._preprocess(chips[start : start + self._batch_size])
            result = self._session.run([self._output_name], {self._input_name: tensor})[0]
            features.append(np.asarray(result, dtype=np.float32).reshape(tensor.shape[0], -1))
        stacked = np.concatenate(features, axis=0) if len(features) > 1 else features[0]
        out[valid] = stacked
        return l2_normalize(out, axis=1) if self._normalize else out

    @property
    def info(self) -> EncoderInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        self._session = None


def _static_batch_dim(input_shape: Sequence[Any], output_shape: Sequence[Any]) -> int | None:
    """Return the fixed batch size of a graph, or ``None`` when it is dynamic.

    ONNX reports a dynamic axis as a string (a symbolic name) and a fixed one as
    an int, so a leading int is exactly the case that cannot be batched.
    """
    for shape in (input_shape, output_shape):
        if shape and isinstance(shape[0], int) and shape[0] > 0:
            return int(shape[0])
    return None
