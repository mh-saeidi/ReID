"""ReID encoder construction from configuration."""

from __future__ import annotations

from pathlib import Path

from src.config.paths import ProjectPaths
from src.config.schema import AppConfig, ReIDBackend
from src.core.exceptions import ConfigurationError
from src.reid.encoder import ReIDEncoder
from src.reid.onnx_reid import OnnxReIDEncoder
from src.reid.yolo26_reid import OFFICIAL_REID_ASSETS, UltralyticsReIDEncoder
from src.utils.device import DeviceInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_model_path(raw: str, paths: ProjectPaths) -> str:
    """Resolve a configured model reference.

    A path that exists on disk (absolute, or relative to the configuration
    file) wins; otherwise the value is passed through so Ultralytics can treat
    it as a downloadable asset name.
    """
    candidate = paths.resolve(raw)
    if candidate.exists():
        return str(candidate)
    if Path(raw).exists():
        return str(Path(raw).resolve())
    return raw


def _select_backend(configured: ReIDBackend, model_path: str) -> ReIDBackend:
    if configured is not ReIDBackend.AUTO:
        return configured
    suffix = Path(model_path).suffix.lower()
    if suffix == ".pt":
        return ReIDBackend.ULTRALYTICS
    if suffix == ".onnx":
        # Official YOLO26 ReID exports go through Ultralytics so they can be
        # auto-downloaded; other ONNX graphs use the dependency-light runtime.
        if Path(model_path).name in OFFICIAL_REID_ASSETS and not Path(model_path).exists():
            return ReIDBackend.ULTRALYTICS
        return ReIDBackend.ONNX
    return ReIDBackend.ULTRALYTICS


def build_encoder(config: AppConfig, paths: ProjectPaths, device: DeviceInfo) -> ReIDEncoder:
    """Create the configured ReID encoder (not yet loaded)."""
    model_path = resolve_model_path(config.models.reid, paths)
    backend = _select_backend(config.reid.backend, model_path)
    size = config.reid.size_hw

    logger.debug(
        "Building ReID encoder",
        extra={"model": model_path, "backend": backend.value, "size": f"{size[0]}x{size[1]}"},
    )

    if backend is ReIDBackend.ONNX:
        return OnnxReIDEncoder(
            model_path,
            device,
            input_size=size,
            batch_size=config.reid.batch_size,
            normalize=config.reid.normalize,
        )
    if backend in (ReIDBackend.ULTRALYTICS, ReIDBackend.TORCHSCRIPT):
        return UltralyticsReIDEncoder(
            model_path,
            device,
            input_size=size,
            fp16=config.device.fp16,
            batch_size=config.reid.batch_size,
            normalize=config.reid.normalize,
            embed_layer=config.reid.embed_layer,
        )
    raise ConfigurationError(  # pragma: no cover - enum is exhaustive
        f"unsupported reid.backend '{backend}'"
    )
