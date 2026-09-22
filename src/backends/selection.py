"""Inference backend selection.

One place decides which runtime each model uses, so the rule is auditable and
the same on every platform:

    Jetson + TensorRT available       -> TensorRT
    NVIDIA CUDA, no TensorRT          -> CUDA (torch) / CUDAExecutionProvider
    Apple Silicon                     -> MPS / CoreML
    anything else                     -> CPU

An explicitly configured backend always wins. An unavailable one degrades with
a warning and a stated reason rather than crashing, because a deployment that
silently runs 5x slower is worse than one that says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.config.schema import InferenceBackend
from src.hardware.capabilities import SystemCapabilities, detect_capabilities
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelRole(str, Enum):
    """Which model a backend decision is about."""

    DETECTOR = "detector"
    FACE_DETECTOR = "face_detector"
    FACE_ENCODER = "face_encoder"
    BODY_ENCODER = "body_encoder"


# Formats each backend can consume. A TensorRT engine is only ever built from
# an ONNX source, so a .pt model must be exported first.
_TENSORRT_SOURCES = frozenset({".onnx", ".engine", ".plan", ".trt"})
_ONNX_SOURCES = frozenset({".onnx"})


@dataclass(frozen=True, slots=True)
class BackendDecision:
    """The chosen backend, and why."""

    role: ModelRole
    backend: InferenceBackend
    requested: InferenceBackend
    reason: str
    fallback_from: InferenceBackend | None = None

    @property
    def is_fallback(self) -> bool:
        return self.fallback_from is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "backend": self.backend.value,
            "requested": self.requested.value,
            "reason": self.reason,
            "fallback_from": self.fallback_from.value if self.fallback_from else None,
        }

    def log(self) -> None:
        if self.is_fallback:
            logger.warning(
                "Backend fallback",
                extra={
                    "role": self.role.value,
                    "requested": self.fallback_from.value if self.fallback_from else "-",
                    "using": self.backend.value,
                    "why": self.reason,
                },
            )
        else:
            logger.debug(
                "Backend selected",
                extra={"role": self.role.value, "backend": self.backend.value},
            )


def tensorrt_enabled(config, caps: SystemCapabilities) -> bool:
    """Resolve ``backend.tensorrt.enabled`` (which may be ``"auto"``)."""
    setting = config.backend.tensorrt.enabled
    if setting is False:
        return False
    if setting is True:
        return True
    # auto: only where the platform actually provides it.
    return caps.has_tensorrt and caps.has_cuda


def _native_backend(role: ModelRole) -> InferenceBackend:
    """The backend these models use when no accelerator applies.

    YuNet and SFace are OpenCV DNN models: OpenCV *is* their runtime, and there
    is no separate ONNX session to fall back to.
    """
    if role in (ModelRole.FACE_DETECTOR,):
        return InferenceBackend.OPENCV
    return InferenceBackend.ONNX


def select_backend(
    role: ModelRole,
    model_path: str | Path,
    config,
    caps: SystemCapabilities | None = None,
) -> BackendDecision:
    """Decide which runtime executes one model."""
    caps = caps or detect_capabilities()
    requested: InferenceBackend = getattr(config.backend, role.value)
    suffix = Path(str(model_path)).suffix.lower()

    def decide(backend: InferenceBackend, reason: str,
               fallback_from: InferenceBackend | None = None) -> BackendDecision:
        decision = BackendDecision(role, backend, requested, reason, fallback_from)
        decision.log()
        return decision

    # ----------------------------------------------------------- explicit ---
    if requested is InferenceBackend.TENSORRT:
        problem = _tensorrt_blocker(suffix, caps, config, role)
        if problem is None:
            return decide(InferenceBackend.TENSORRT, "explicitly configured")
        return decide(
            _fallback_for(role, suffix),
            f"TensorRT requested but unusable: {problem}",
            fallback_from=InferenceBackend.TENSORRT,
        )

    if requested is not InferenceBackend.AUTO:
        return decide(requested, "explicitly configured")

    # --------------------------------------------------------------- auto ---
    if tensorrt_enabled(config, caps):
        problem = _tensorrt_blocker(suffix, caps, config, role)
        if problem is None:
            where = "Jetson" if caps.is_jetson else "CUDA host"
            return decide(InferenceBackend.TENSORRT, f"TensorRT available on this {where}")
        logger.debug("TensorRT not used for %s: %s", role.value, problem)

    return decide(_fallback_for(role, suffix), _auto_reason(caps))


# Roles this implementation does not execute through TensorRT. YuNet is run by
# OpenCV's own DNN module, which owns its pre/postprocessing; there is no
# TensorRT path for it here, so reporting one would be misleading.
#
# SCRFD is an ONNX model and does benefit from TensorRT, but not through this
# route: the shared TensorRTSession returns a single output tensor, and SCRFD
# emits nine (three strides x score/bbox/landmark). It is accelerated instead
# by ONNX Runtime's TensorRT execution provider, which handles multi-output
# graphs and caches its own engines -- see src/face/scrfd.py.
_NO_TENSORRT_ROLES = frozenset({ModelRole.FACE_DETECTOR})


def _tensorrt_blocker(
    suffix: str, caps: SystemCapabilities, config, role: ModelRole | None = None
) -> str | None:
    """Why TensorRT cannot be used here, or ``None`` when it can."""
    if role in _NO_TENSORRT_ROLES:
        return (
            "this model runs on OpenCV DNN, which has no TensorRT path in this "
            "implementation"
        )
    if not caps.has_tensorrt:
        return "TensorRT is not installed on this system"
    if not caps.has_cuda:
        return "TensorRT needs a CUDA device and none was found"
    if suffix not in _TENSORRT_SOURCES:
        return (
            f"'{suffix or 'no extension'}' cannot be converted to a TensorRT "
            "engine; export the model to ONNX first"
        )
    if not caps.tensorrt.can_build and suffix not in (".engine", ".plan", ".trt"):
        return "no TensorRT builder (python bindings or trtexec) is available"
    if not config.backend.tensorrt.allow_build and suffix == ".onnx":
        return "backend.tensorrt.allow_build is false and no prebuilt engine was given"
    return None


def _fallback_for(role: ModelRole, suffix: str) -> InferenceBackend:
    if suffix == ".pt":
        return InferenceBackend.PYTORCH
    if suffix in _ONNX_SOURCES:
        return (
            InferenceBackend.OPENCV
            if role is ModelRole.FACE_DETECTOR
            else InferenceBackend.ONNX
        )
    return _native_backend(role)


def _auto_reason(caps: SystemCapabilities) -> str:
    if caps.has_cuda:
        return "CUDA available, TensorRT not in use"
    if caps.mps_available:
        return "Apple Silicon: MPS/CoreML"
    return "CPU"


def describe_selection(config, caps: SystemCapabilities | None = None) -> dict[str, object]:
    """Summarise every backend decision, for logs, `system info` and the API."""
    caps = caps or detect_capabilities()
    return {
        "tensorrt_enabled": tensorrt_enabled(config, caps),
        "platform": {
            "jetson": caps.is_jetson,
            "cuda": caps.has_cuda,
            "tensorrt": caps.has_tensorrt,
            "mps": caps.mps_available,
        },
    }
