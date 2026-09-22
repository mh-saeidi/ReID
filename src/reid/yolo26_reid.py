"""YOLO26 ReID encoder backed by the Ultralytics ecosystem.

Two model families are supported through one class:

* The official exported ReID encoders (``yolo26{n,s,m,l,x}-reid.onnx``), which
  output the embedding tensor directly and are auto-downloaded when missing.
  These are run through :class:`ultralytics.nn.autobackend.AutoBackend`, so
  ONNX Runtime, OpenVINO and TensorRT engines all work unchanged.
* Any ``.pt`` Ultralytics checkpoint (for example ``yolo26n-cls.pt`` or a
  classification model fine-tuned on identity-labelled crops), whose embeddings
  are pulled from the second-to-last layer.

Preprocessing is done here rather than delegated, so the geometry the crops are
resized to is the configured one and is recorded in the encoder fingerprint.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import ModelLoadError
from src.reid.encoder import EncoderInfo, ReIDEncoder, l2_normalize
from src.utils.device import DeviceInfo
from src.utils.image import resize_crop
from src.utils.logging import get_logger

logger = get_logger(__name__)

OFFICIAL_REID_ASSETS: frozenset[str] = frozenset(
    f"yolo26{scale}-reid.onnx" for scale in "nsmlx"
)
_EXPORTED_SUFFIXES = (".onnx", ".torchscript", ".engine", ".mlpackage", ".tflite", ".xml")


class UltralyticsReIDEncoder(ReIDEncoder):
    """Appearance encoder for person crops.

    Args:
        model_path: Weights path or official asset name (``yolo26n-reid.onnx``).
        device: Resolved compute device.
        input_size: ``(height, width)`` the crops are resized to.
        fp16: Request half precision when the backend supports it.
        batch_size: Maximum crops per forward pass.
        normalize: L2-normalise the returned embeddings (required for cosine).
        embed_layer: ``.pt`` path only -- layer index to read features from.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceInfo,
        *,
        input_size: tuple[int, int] = (256, 128),
        fp16: bool = False,
        batch_size: int = 16,
        normalize: bool = True,
        embed_layer: int | None = None,
    ) -> None:
        self._model_path = str(model_path)
        self._device = device
        self._input_size = input_size
        self._fp16 = bool(fp16 and device.fp16_available)
        self._batch_size = max(1, batch_size)
        self._normalize = normalize
        self._embed_layer = embed_layer

        self._backend: Any | None = None
        self._is_pt = Path(self._model_path).suffix.lower() == ".pt"
        self._static_batch: int | None = None
        self._info: EncoderInfo | None = None

    # ---------------------------------------------------------------- loading
    def load(self) -> EncoderInfo:
        if self._info is not None:
            return self._info

        started = time.perf_counter()
        resolved = self._resolve_model_path()
        if self._is_pt:
            backend_name = self._load_pytorch(resolved)
        else:
            backend_name = self._load_exported(resolved)

        dimension = self._probe_embedding_dimension()
        elapsed = time.perf_counter() - started

        self._info = EncoderInfo(
            name=Path(resolved).stem,
            model_path=str(resolved),
            backend=backend_name,
            device=self._device.device,
            fp16=self._fp16,
            input_size=self._input_size,
            embedding_dimension=dimension,
            load_time_s=elapsed,
            metadata={"static_batch": self._static_batch, "normalize": self._normalize},
        )
        logger.info(
            "ReID encoder loaded",
            extra={
                "model": self._info.name,
                "backend": backend_name,
                "device": self._info.device,
                "input": f"{self._input_size[0]}x{self._input_size[1]}",
                "dim": dimension,
                "fp16": self._fp16,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    def _resolve_model_path(self) -> str:
        path = Path(self._model_path)
        if path.exists():
            return str(path)
        if path.name in OFFICIAL_REID_ASSETS:
            try:
                from ultralytics.utils.downloads import attempt_download_asset  # noqa: PLC0415

                logger.info("Downloading official ReID asset", extra={"asset": path.name})
                return str(attempt_download_asset(str(path)))
            except Exception as exc:
                raise ModelLoadError(
                    f"cannot download the official ReID asset '{path.name}': {exc}. "
                    "Download it manually into models/ or set models.reid to a local file."
                ) from exc
        if self._is_pt:
            # Ultralytics resolves known .pt asset names (e.g. yolo26n-cls.pt) itself.
            return self._model_path
        raise ModelLoadError(
            f"ReID model not found: '{self._model_path}'. Set models.reid to an "
            f"existing file, or to one of the official assets "
            f"{sorted(OFFICIAL_REID_ASSETS)}."
        )

    def _load_exported(self, resolved: str) -> str:
        suffix = Path(resolved).suffix.lower()
        if suffix not in _EXPORTED_SUFFIXES and not Path(resolved).is_dir():
            raise ModelLoadError(
                f"unsupported ReID model format '{suffix}' for '{resolved}'. "
                f"Supported: .pt or an exported format {list(_EXPORTED_SUFFIXES)}."
            )
        try:
            import torch  # noqa: PLC0415
            from ultralytics.nn.autobackend import AutoBackend  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ModelLoadError(
                "torch and ultralytics are required to run exported ReID models"
            ) from exc

        try:
            backend = AutoBackend(
                resolved,
                device=torch.device(self._device.device),
                fp16=self._fp16,
                verbose=False,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"cannot load the ReID model '{resolved}': {exc}. "
                "Verify the file is a valid exported model and that the matching "
                "runtime (e.g. onnxruntime) is installed."
            ) from exc

        self._backend = backend
        self._fp16 = bool(getattr(backend, "fp16", self._fp16))
        self._adopt_static_shapes(backend)
        return f"autobackend{Path(resolved).suffix}"

    def _adopt_static_shapes(self, backend: Any) -> None:
        """Respect fixed input shapes baked into an exported model."""
        session = getattr(backend, "session", None)
        if session is None:
            return
        try:
            shape = session.get_inputs()[0].shape
        except Exception:  # pragma: no cover - backend specific
            return
        if len(shape) != 4:
            return
        if isinstance(shape[0], int) and shape[0] > 0:
            self._static_batch = int(shape[0])
        height, width = shape[2], shape[3]
        if isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0:
            if (height, width) != self._input_size:
                logger.info(
                    "ReID model has a fixed input size; overriding reid.input_size",
                    extra={
                        "configured": f"{self._input_size[0]}x{self._input_size[1]}",
                        "model": f"{height}x{width}",
                    },
                )
            self._input_size = (int(height), int(width))

    def _load_pytorch(self, resolved: str) -> str:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ModelLoadError("ultralytics is required to run .pt ReID models") from exc
        try:
            model = YOLO(resolved)
            layer = self._embed_layer
            if layer is None:
                layer = len(model.model.model) - 2  # second-to-last layer features
            # Priming the predictor with embed=[...] makes later calls return features.
            model(embed=[layer], device=self._device.device, verbose=False, save=False)
        except Exception as exc:
            raise ModelLoadError(
                f"cannot initialise the .pt ReID model '{resolved}': {exc}. "
                "models.reid must be an Ultralytics checkpoint (e.g. yolo26n-cls.pt)."
            ) from exc
        self._backend = model
        self._fp16 = False  # the predictor path manages precision itself
        return "ultralytics.pt"

    def _probe_embedding_dimension(self) -> int:
        """Run one dummy crop to read the embedding width from the model."""
        height, width = self._input_size
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            features = self._forward([dummy])
        except Exception as exc:
            raise ModelLoadError(
                f"the ReID model '{self._model_path}' failed its warm-up pass: {exc}. "
                "It may be incompatible with reid.input_size or the selected device."
            ) from exc
        if features.ndim != 2 or features.shape[0] != 1:
            raise ModelLoadError(
                f"the ReID model '{self._model_path}' returned an unexpected output "
                f"shape {features.shape}; a (batch, dimension) embedding tensor is required."
            )
        return int(features.shape[1])

    # -------------------------------------------------------------- inference
    def _preprocess(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """BGR uint8 crops -> normalised float32 NCHW RGB batch."""
        height, width = self._input_size
        batch = np.empty((len(crops), height, width, 3), dtype=np.float32)
        for index, crop in enumerate(crops):
            resized = resize_crop(crop, (height, width))
            batch[index] = resized[:, :, ::-1]  # BGR -> RGB
        batch /= 255.0
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))

    def _forward(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if self._backend is None:
            self.load()
        if self._is_pt:
            return self._forward_pytorch(crops)
        return self._forward_exported(crops)

    def _forward_exported(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import torch  # noqa: PLC0415

        tensor = torch.from_numpy(self._preprocess(crops)).to(self._device.device)
        if self._fp16:
            tensor = tensor.half()

        count = tensor.shape[0]
        static = self._static_batch
        with torch.no_grad():
            if static is None or count == static:
                output = self._backend(tensor)
            else:
                # Fixed-batch exports: pad the final chunk and discard the padding.
                chunks = []
                for start in range(0, count, static):
                    chunk = tensor[start : start + static]
                    if chunk.shape[0] < static:
                        pad = chunk[-1:].expand(static - chunk.shape[0], *chunk.shape[1:])
                        chunk = torch.cat([chunk, pad], dim=0)
                    chunks.append(self._backend(chunk))
                output = torch.cat(chunks, dim=0)[:count]
        return self._to_2d(output, count)

    def _forward_pytorch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        prepared = [resize_crop(crop, self._input_size) for crop in crops]
        features = self._backend.predictor(prepared)
        if len(features) != len(prepared) and getattr(features[0], "shape", (0,))[0] == len(
            prepared
        ):
            features = features[0]  # batched output from a non-PyTorch backend
        stacked = np.stack([self._as_numpy(f).reshape(-1) for f in features], axis=0)
        return stacked.astype(np.float32, copy=False)

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            return value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _to_2d(self, output: Any, count: int) -> np.ndarray:
        if isinstance(output, (list, tuple)):
            output = output[0]
        array = self._as_numpy(output)
        if array.ndim > 2:
            array = array.reshape(count, -1)
        return array.astype(np.float32, copy=False)

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Embed BGR crops, returning zero rows for unusable ones."""
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
        outputs: list[np.ndarray] = []
        for start in range(0, len(valid), self._batch_size):
            outputs.append(self._forward(valid[start : start + self._batch_size]))
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
        self._backend = None
