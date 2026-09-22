"""Face detector and encoder construction from configuration."""

from __future__ import annotations

from pathlib import Path

from src.backends.selection import ModelRole
from src.backends.selection import select_backend as select_inference_backend
from src.backends.trt_encoder import try_build_tensorrt_encoder
from src.config.paths import ProjectPaths
from src.config.schema import AppConfig, FaceBackend, InferenceBackend
from src.core.exceptions import ConfigurationError, ModelLoadError
from src.face.detector import FaceDetector, YuNetFaceDetector
from src.face.embedder import ArcFaceOnnxEmbedder, SFaceEmbedder
from src.reid.encoder import ReIDEncoder
from src.utils.device import DeviceInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

SFACE_MARKERS = ("sface",)
ARCFACE_MARKERS = ("w600k", "glintr", "arcface", "r50", "r100", "webface")


def _resolve(raw: str, paths: ProjectPaths) -> Path:
    candidate = paths.resolve(raw)
    if candidate.exists():
        return candidate
    direct = Path(raw)
    if direct.exists():
        return direct.resolve()
    return candidate  # report the resolved path in the error message


def select_backend(configured: FaceBackend, recognition_model: Path) -> FaceBackend:
    """Choose a recogniser backend from the model filename when set to auto."""
    if configured is not FaceBackend.AUTO:
        return configured
    name = recognition_model.name.lower()
    if any(marker in name for marker in SFACE_MARKERS):
        return FaceBackend.OPENCV
    if any(marker in name for marker in ARCFACE_MARKERS):
        return FaceBackend.ARCFACE_ONNX
    # OpenCV's SFace loader is the safer default: it validates the graph and
    # fails loudly rather than silently producing meaningless features.
    return FaceBackend.OPENCV


def build_face_detector(config: AppConfig, paths: ProjectPaths) -> FaceDetector:
    """Create the configured face detector (not yet loaded)."""
    model = _resolve(config.face.detector_model, paths)
    if not model.exists():
        raise ModelLoadError(
            f"face detector model not found: {model}. "
            "Fetch the face models with: python scripts/fetch_face_models.py"
        )
    return YuNetFaceDetector(
        model,
        confidence=config.face.detection_confidence,
        nms_threshold=config.face.nms_threshold,
        top_k=config.face.top_k,
    )


def build_face_encoder(
    config: AppConfig, paths: ProjectPaths, device: DeviceInfo
) -> ReIDEncoder:
    """Create the configured face recogniser (not yet loaded)."""
    model = _resolve(config.face.recognition_model, paths)
    if not model.exists():
        raise ModelLoadError(
            f"face recognition model not found: {model}. "
            "Fetch the face models with: python scripts/fetch_face_models.py"
        )
    backend = select_backend(config.face.backend, model)
    logger.debug(
        "Building face encoder", extra={"model": str(model), "backend": backend.value}
    )

    # TensorRT is tried first where the platform supports it. It is a pure
    # acceleration swap: same preprocessing, same embedding space, same
    # thresholds. When it is unavailable the ONNX/OpenCV encoder below runs
    # instead -- a slower correct system beats a stopped one.
    decision = select_inference_backend(ModelRole.FACE_ENCODER, model, config)
    if decision.backend is InferenceBackend.TENSORRT:
        encoder = try_build_tensorrt_encoder(
            model,
            config,
            paths,
            role="face_encoder",
            chip_size=config.face.chip_size,
            input_mean=config.face.arcface_input_mean,
            input_scale=config.face.arcface_input_scale,
        )
        if encoder is not None:
            return encoder

    if backend is FaceBackend.OPENCV:
        return SFaceEmbedder(
            model,
            device,
            normalize=config.reid.normalize,
            batch_size=config.recognition.batch.max_size,
        )
    if backend is FaceBackend.ARCFACE_ONNX:
        return ArcFaceOnnxEmbedder(
            model,
            device,
            normalize=config.reid.normalize,
            batch_size=config.recognition.batch.max_size,
            input_mean=config.face.arcface_input_mean,
            input_scale=config.face.arcface_input_scale,
        )
    raise ConfigurationError(  # pragma: no cover - enum is exhaustive
        f"unsupported face.backend '{backend}'"
    )
