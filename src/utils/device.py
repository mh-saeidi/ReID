"""Device selection and capability reporting.

``auto`` resolves to CUDA, then Apple MPS, then CPU. An unavailable accelerator
degrades to CPU with a warning instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.schema import DeviceConfig, DeviceKind
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Resolved compute device and what it supports."""

    device: str
    """Torch/Ultralytics device string: 'cpu', 'cuda:0', 'mps'."""
    kind: DeviceKind
    fp16_available: bool
    fp16_enabled: bool
    torch_available: bool
    description: str = ""

    @property
    def is_cuda(self) -> bool:
        return self.kind is DeviceKind.CUDA


def _torch():
    try:
        import torch  # noqa: PLC0415 - optional heavy import kept lazy
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return None
    return torch


def cuda_available() -> bool:
    torch = _torch()
    return bool(torch and torch.cuda.is_available())


def mps_available() -> bool:
    torch = _torch()
    return bool(
        torch
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )


def resolve_device(config: DeviceConfig) -> DeviceInfo:
    """Resolve the configured device, degrading gracefully when unavailable."""
    torch = _torch()
    torch_available = torch is not None
    requested = config.device

    if requested is DeviceKind.AUTO:
        if cuda_available():
            kind = DeviceKind.CUDA
        elif mps_available():
            kind = DeviceKind.MPS
        else:
            kind = DeviceKind.CPU
    else:
        kind = requested
        if kind is DeviceKind.CUDA and not cuda_available():
            logger.warning(
                "CUDA requested but unavailable (torch.cuda.is_available() is False); "
                "falling back to CPU. Install a CUDA build of torch to use the GPU."
            )
            kind = DeviceKind.CPU
        elif kind is DeviceKind.MPS and not mps_available():
            logger.warning("Apple MPS requested but unavailable; falling back to CPU.")
            kind = DeviceKind.CPU

    if kind is DeviceKind.CUDA:
        device = f"cuda:{config.cuda_index}"
        description = _cuda_description(torch, config.cuda_index)
        fp16_available = True
    elif kind is DeviceKind.MPS:
        device, description, fp16_available = "mps", "Apple Silicon GPU (Metal)", False
    else:
        device, description, fp16_available = "cpu", "CPU", False

    fp16_enabled = bool(config.fp16 and fp16_available)
    if config.fp16 and not fp16_available:
        logger.info("fp16 requested but not supported on %s; using fp32.", device)

    return DeviceInfo(
        device=device,
        kind=kind,
        fp16_available=fp16_available,
        fp16_enabled=fp16_enabled,
        torch_available=torch_available,
        description=description,
    )


def _cuda_description(torch, index: int) -> str:  # pragma: no cover - needs a GPU
    try:
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        return f"{name} ({total:.1f} GiB)"
    except Exception:  # noqa: BLE001 - description is cosmetic
        return f"CUDA device {index}"


def log_device_info(info: DeviceInfo) -> None:
    logger.info(
        "Compute device selected",
        extra={"device": info.device, "detail": info.description, "fp16": info.fp16_enabled},
    )
