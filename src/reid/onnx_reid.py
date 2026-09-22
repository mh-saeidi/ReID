"""Pure ONNX Runtime ReID encoder.

Runs any ReID ONNX graph that maps ``(N, 3, H, W)`` to ``(N, D)`` without
pulling in PyTorch, which keeps CPU-only and containerised deployments small.
Mean/std normalisation is configurable because third-party encoders (OSNet,
TransReID, ...) use ImageNet statistics while the official YOLO26 ReID exports
expect plain ``/255``.
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
from src.utils.image import resize_crop
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OnnxReIDEncoder(ReIDEncoder):
    """ONNX Runtime appearance encoder.

    Args:
        model_path: Path to the ``.onnx`` graph.
        device: Resolved compute device (selects the execution provider).
        input_size: ``(height, width)``; overridden by a static model shape.
        batch_size: Maximum crops per forward pass.
        normalize: L2-normalise the output embeddings.
        mean/std: Optional per-channel normalisation applied after ``/255``.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceInfo,
        *,
        input_size: tuple[int, int] = (256, 128),
        batch_size: int = 16,
        normalize: bool = True,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        self._model_path = str(model_path)
        self._device = device
        self._input_size = input_size
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1) if mean else None
        self._std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1) if std else None

        self._session: Any | None = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._static_batch: int | None = None
        self._info: EncoderInfo | None = None

    def _providers(self) -> list[str]:
        import onnxruntime as ort  # noqa: PLC0415

        available = set(ort.get_available_providers())
        preferred: list[str] = []
        if self._device.is_cuda and "CUDAExecutionProvider" in available:
            preferred.append("CUDAExecutionProvider")
        if "CoreMLExecutionProvider" in available and self._device.device == "mps":
            preferred.append("CoreMLExecutionProvider")
        preferred.append("CPUExecutionProvider")
        return preferred

    def load(self) -> EncoderInfo:
        if self._info is not None:
            return self._info
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                "onnxruntime is required for the 'onnx' ReID backend: "
                "pip install onnxruntime (or onnxruntime-gpu)"
            ) from exc

        path = Path(self._model_path)
        if not path.exists():
            raise ModelLoadError(f"ONNX ReID model not found: {path}")

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(str(path), providers=self._providers())
        except Exception as exc:
            raise ModelLoadError(f"cannot load the ONNX ReID model '{path}': {exc}") from exc

        self._session = session
        spec = session.get_inputs()[0]
        self._input_name = spec.name
        self._output_name = session.get_outputs()[0].name
        self._adopt_static_shape(spec.shape)

        dimension = self._probe_dimension(session)
        elapsed = time.perf_counter() - started
        self._info = EncoderInfo(
            name=path.stem,
            model_path=str(path),
            backend="onnxruntime",
            device=session.get_providers()[0],
            fp16=False,
            input_size=self._input_size,
            embedding_dimension=dimension,
            load_time_s=elapsed,
            metadata={"providers": session.get_providers(), "static_batch": self._static_batch},
        )
        logger.info(
            "ReID encoder loaded",
            extra={
                "model": path.stem,
                "backend": "onnxruntime",
                "provider": session.get_providers()[0],
                "dim": dimension,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    def _adopt_static_shape(self, shape: Sequence[Any]) -> None:
        if len(shape) != 4:
            return
        if isinstance(shape[0], int) and shape[0] > 0:
            self._static_batch = int(shape[0])
        height, width = shape[2], shape[3]
        if isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0:
            self._input_size = (int(height), int(width))

    def _probe_dimension(self, session: Any) -> int:
        height, width = self._input_size
        dummy = np.zeros((self._static_batch or 1, 3, height, width), dtype=np.float32)
        try:
            output = session.run([self._output_name], {self._input_name: dummy})[0]
        except Exception as exc:
            raise ModelLoadError(
                f"the ONNX ReID model '{self._model_path}' failed its warm-up pass: {exc}"
            ) from exc
        array = np.asarray(output)
        if array.ndim < 2:
            raise ModelLoadError(
                f"the ONNX ReID model '{self._model_path}' must output a "
                f"(batch, dimension) tensor, got shape {array.shape}"
            )
        return int(np.prod(array.shape[1:]))

    def _preprocess(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        height, width = self._input_size
        batch = np.empty((len(crops), height, width, 3), dtype=np.float32)
        for index, crop in enumerate(crops):
            batch[index] = resize_crop(crop, (height, width))[:, :, ::-1]
        batch /= 255.0
        tensor = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        if self._mean is not None:
            tensor = tensor - self._mean
        if self._std is not None:
            tensor = tensor / self._std
        return tensor

    def _run(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        assert self._session is not None  # noqa: S101
        tensor = self._preprocess(crops)
        count = tensor.shape[0]
        static = self._static_batch
        if static is not None and count != static:
            pad = np.repeat(tensor[-1:], static - count % static, axis=0) if count % static else None
            chunks = []
            for start in range(0, count, static):
                chunk = tensor[start : start + static]
                if chunk.shape[0] < static:
                    chunk = np.concatenate([chunk, pad], axis=0)
                chunks.append(self._session.run([self._output_name], {self._input_name: chunk})[0])
            output = np.concatenate(chunks, axis=0)[:count]
        else:
            output = self._session.run([self._output_name], {self._input_name: tensor})[0]
        return np.asarray(output, dtype=np.float32).reshape(count, -1)

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if self._info is None:
            self.load()
        assert self._info is not None  # noqa: S101
        dimension = self._info.embedding_dimension

        valid_indices = [
            i for i, crop in enumerate(crops) if crop is not None and getattr(crop, "size", 0)
        ]
        embeddings = np.zeros((len(crops), dimension), dtype=np.float32)
        if not valid_indices:
            return embeddings

        valid = [crops[i] for i in valid_indices]
        outputs = [
            self._run(valid[start : start + self._batch_size])
            for start in range(0, len(valid), self._batch_size)
        ]
        features = np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]
        if self._normalize:
            features = l2_normalize(features, axis=1)
        embeddings[valid_indices] = features
        return embeddings

    @property
    def info(self) -> EncoderInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        self._session = None
