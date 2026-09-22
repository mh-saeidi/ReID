"""TensorRT face/body encoder implementing the existing ReIDEncoder interface.

Nothing downstream changes: the gallery, matcher, scheduler and pipeline still
see a :class:`~src.reid.encoder.ReIDEncoder`. The only difference is where the
matrix multiplies happen.

Falling back is a first-class path. If the engine cannot be built or loaded,
this class raises :class:`TensorRTUnavailable` and the factory constructs the
ONNX encoder instead -- a slower but correct system beats a stopped one.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.backends.tensorrt_builder import BuildRequest, TensorRTBuilder, TensorRTUnavailable
from src.backends.tensorrt_runtime import TensorRTSession
from src.reid.encoder import EncoderInfo, ReIDEncoder, l2_normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TensorRTEncoder(ReIDEncoder):
    """Runs an ArcFace-style encoder through a TensorRT engine.

    Preprocessing is identical to :class:`~src.face.embedder.ArcFaceOnnxEmbedder`
    -- same resize, same BGR->RGB, same ``(x - mean) / scale`` -- because a
    difference there would change the embeddings and therefore every calibrated
    threshold. The engine swap must be numerically neutral apart from FP16
    rounding.
    """

    def __init__(
        self,
        source_model: Path,
        builder: TensorRTBuilder,
        *,
        role: str = "face_encoder",
        chip_size: int = 112,
        normalize: bool = True,
        batch_size: int = 8,
        input_mean: float = 127.5,
        input_scale: float = 127.5,
        input_name: str | None = "input",
    ) -> None:
        self._source = Path(source_model)
        self._builder = builder
        self._role = role
        self._chip_size = chip_size
        self._normalize = normalize
        self._batch_size = max(1, batch_size)
        self._input_mean = input_mean
        self._input_scale = input_scale
        self._input_name = input_name
        self._session: TensorRTSession | None = None
        self._info: EncoderInfo | None = None
        self._stage_buffer: np.ndarray | None = None

    # --------------------------------------------------------------- loading
    def load(self) -> EncoderInfo:
        if self._info is not None:
            return self._info

        started = time.perf_counter()
        request = BuildRequest(
            role=self._role,
            source=self._source,
            input_shape=(3, self._chip_size, self._chip_size),
            input_name=self._input_name,
            dynamic_batch=True,
        )
        result = self._builder.ensure_engine(request)

        session = TensorRTSession(
            result.engine_path,
            max_batch_size=result.metadata.max_batch_size,
        )
        session.load()
        self._session = session
        self._batch_size = min(self._batch_size, session.max_batch_size)

        channels, height, width = session.input_chw
        if (height, width) != (self._chip_size, self._chip_size):
            logger.info(
                "Engine input geometry overrides the configured chip size",
                extra={"configured": self._chip_size, "engine": f"{height}x{width}"},
            )
            self._chip_size = height

        elapsed = time.perf_counter() - started
        self._info = EncoderInfo(
            name=self._source.stem,
            model_path=str(self._source),
            backend="tensorrt",
            device=f"cuda:{result.metadata.gpu_name or 0}",
            fp16=result.metadata.precision in ("fp16", "int8"),
            input_size=(self._chip_size, self._chip_size),
            embedding_dimension=session.output_dimension,
            load_time_s=elapsed,
            metadata={
                "aligned_input": True,
                "modality": "face" if self._role == "face_encoder" else "body",
                "engine": str(result.engine_path),
                "precision": result.metadata.precision,
                "max_batch": session.max_batch_size,
                "rebuilt": result.rebuilt,
            },
        )
        logger.info(
            "Face encoder loaded",
            extra={
                "model": self._source.stem,
                "backend": "tensorrt",
                "precision": result.metadata.precision,
                "dim": session.output_dimension,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    # ------------------------------------------------------------- inference
    def _preprocess(self, chips: Sequence[np.ndarray]) -> np.ndarray:
        """Chips -> NCHW float32, reusing one staging buffer across calls."""
        import cv2  # noqa: PLC0415

        size = self._chip_size
        count = len(chips)
        if self._stage_buffer is None or self._stage_buffer.shape[0] < count:
            self._stage_buffer = np.empty((max(count, self._batch_size), size, size, 3),
                                          dtype=np.float32)
        staging = self._stage_buffer[:count]
        for index, chip in enumerate(chips):
            if chip.shape[0] != size or chip.shape[1] != size:
                chip = cv2.resize(chip, (size, size), interpolation=cv2.INTER_LINEAR)
            staging[index] = chip[:, :, ::-1]
        staging -= self._input_mean
        staging /= self._input_scale
        return np.ascontiguousarray(staging.transpose(0, 3, 1, 2))

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if self._info is None:
            self.load()
        assert self._info is not None and self._session is not None  # noqa: S101
        dimension = self._info.embedding_dimension
        out = np.zeros((len(crops), dimension), dtype=np.float32)

        valid = [i for i, c in enumerate(crops) if c is not None and getattr(c, "size", 0)]
        if not valid:
            return out

        chips = [crops[i] for i in valid]
        features: list[np.ndarray] = []
        for start in range(0, len(chips), self._batch_size):
            window = chips[start : start + self._batch_size]
            features.append(self._session.infer(self._preprocess(window)))
        stacked = np.concatenate(features, axis=0) if len(features) > 1 else features[0]
        out[valid] = stacked.reshape(len(chips), -1)
        return l2_normalize(out, axis=1) if self._normalize else out

    @property
    def info(self) -> EncoderInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


def try_build_tensorrt_encoder(
    source_model: Path,
    config,
    paths,
    *,
    role: str = "face_encoder",
    chip_size: int = 112,
    input_mean: float = 127.5,
    input_scale: float = 127.5,
    caps=None,
) -> TensorRTEncoder | None:
    """Construct a TensorRT encoder, or ``None`` when it is not usable here.

    Returning ``None`` rather than raising keeps the fallback decision in the
    factory, where the ONNX alternative lives.
    """
    from src.backends.engine_store import EngineStore  # noqa: PLC0415

    trt_config = config.backend.tensorrt
    store = EngineStore(
        paths.resolve(trt_config.engine_dir),
        strict_version_check=trt_config.strict_version_check,
    )
    builder = TensorRTBuilder(trt_config, store, caps)
    encoder = TensorRTEncoder(
        source_model,
        builder,
        role=role,
        chip_size=chip_size,
        normalize=config.reid.normalize,
        batch_size=config.recognition.batch.max_size,
        input_mean=input_mean,
        input_scale=input_scale,
    )
    try:
        encoder.load()
    except TensorRTUnavailable as exc:
        logger.warning(
            "TensorRT encoder unavailable; falling back to ONNX Runtime",
            extra={"role": role, "reason": str(exc)},
        )
        return None
    except Exception as exc:  # noqa: BLE001 - never let TRT break startup
        logger.warning(
            "TensorRT encoder failed to initialise; falling back to ONNX Runtime",
            extra={"role": role, "error": str(exc)},
        )
        return None
    return encoder
