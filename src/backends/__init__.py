"""Inference backend selection, TensorRT engine building and execution."""

from src.backends.engine_store import (
    EngineMetadata,
    EngineStore,
    build_expected_metadata,
    source_fingerprint,
)
from src.backends.selection import (
    BackendDecision,
    ModelRole,
    describe_selection,
    select_backend,
    tensorrt_enabled,
)
from src.backends.tensorrt_builder import (
    BuildRequest,
    BuildResult,
    TensorRTBuilder,
    TensorRTUnavailable,
    default_build_requests,
)

__all__ = [
    "ModelRole",
    "BackendDecision",
    "select_backend",
    "tensorrt_enabled",
    "describe_selection",
    "EngineStore",
    "EngineMetadata",
    "source_fingerprint",
    "build_expected_metadata",
    "TensorRTBuilder",
    "BuildRequest",
    "BuildResult",
    "TensorRTUnavailable",
    "default_build_requests",
]
