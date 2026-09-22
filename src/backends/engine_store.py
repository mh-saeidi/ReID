"""TensorRT engine cache: metadata, fingerprinting and invalidation.

A TensorRT engine is a *derived artefact*, never the source of truth. It is
tied to the exact GPU architecture, TensorRT version, CUDA version, input
geometry and precision it was built with, and silently using a stale or foreign
engine is one of the easier ways to ship wrong results at high speed.

Every engine therefore gets a JSON sidecar recording what produced it, and is
rebuilt whenever any of that changes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.exceptions import ModelLoadError
from src.hardware.capabilities import SystemCapabilities, detect_capabilities
from src.utils.logging import get_logger

logger = get_logger(__name__)

METADATA_SUFFIX = ".engine.json"
ENGINE_SUFFIX = ".engine"
SCHEMA_VERSION = 1

# Hashing a 174 MB ONNX file on every startup would cost more than it saves, so
# the fingerprint uses size + mtime + a bounded sample of the content. That
# detects a replaced or edited model without reading it end to end.
_HASH_SAMPLE_BYTES = 1 << 20  # 1 MiB from the head and the tail


def source_fingerprint(path: Path) -> str:
    """Cheap but change-sensitive fingerprint of a model file."""
    path = Path(path)
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(_HASH_SAMPLE_BYTES))
            if stat.st_size > _HASH_SAMPLE_BYTES * 2:
                handle.seek(-_HASH_SAMPLE_BYTES, 2)
                digest.update(handle.read(_HASH_SAMPLE_BYTES))
        return digest.hexdigest()
    except OSError as exc:
        raise ModelLoadError(f"cannot fingerprint model {path}: {exc}") from exc


@dataclass(slots=True)
class EngineMetadata:
    """Everything needed to decide whether an engine is still valid."""

    schema_version: int
    role: str
    source_model: str
    source_fingerprint: str
    engine_path: str
    precision: str
    tensorrt_version: str | None
    cuda_version: str | None
    jetpack_version: str | None
    gpu_name: str | None
    compute_capability: str | None
    input_shape: list[int]
    min_batch_size: int
    optimal_batch_size: int
    max_batch_size: int
    workspace_mb: int
    builder_optimization_level: int
    built_at: str
    build_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def build_fingerprint(self) -> str:
        """Identifies the *build configuration*, independent of the source file."""
        parts = [
            self.role,
            self.precision,
            str(self.tensorrt_version),
            str(self.cuda_version),
            str(self.compute_capability),
            "x".join(str(v) for v in self.input_shape),
            f"{self.min_batch_size}-{self.optimal_batch_size}-{self.max_batch_size}",
            str(self.workspace_mb),
            str(self.builder_optimization_level),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["build_fingerprint"] = self.build_fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineMetadata:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Why a cached engine was accepted or rejected."""

    valid: bool
    reason: str

    def __bool__(self) -> bool:
        return self.valid


class EngineStore:
    """Locates, validates and records TensorRT engines on disk."""

    def __init__(self, engine_dir: Path, *, strict_version_check: bool = True) -> None:
        self._dir = Path(engine_dir)
        self._strict = strict_version_check

    @property
    def directory(self) -> Path:
        return self._dir

    # ------------------------------------------------------------ locations
    def engine_path(self, source: Path, role: str, precision: str, batch: int) -> Path:
        """Deterministic engine filename derived from the build parameters.

        Encoding the parameters in the name means several precisions or batch
        limits can coexist rather than overwriting one another.
        """
        stem = Path(source).stem
        return self._dir / f"{stem}.{role}.{precision}.b{batch}{ENGINE_SUFFIX}"

    @staticmethod
    def metadata_path(engine: Path) -> Path:
        return engine.with_suffix("").with_suffix(METADATA_SUFFIX)

    # ------------------------------------------------------------- metadata
    def load_metadata(self, engine: Path) -> EngineMetadata | None:
        path = self.metadata_path(engine)
        if not path.exists():
            return None
        try:
            return EngineMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Ignoring unreadable engine metadata; the engine will be rebuilt",
                extra={"path": str(path), "error": str(exc)},
            )
            return None

    def save_metadata(self, metadata: EngineMetadata) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.metadata_path(Path(metadata.engine_path))
        path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        return path

    # ----------------------------------------------------------- validation
    def validate(
        self,
        engine: Path,
        source: Path,
        expected: EngineMetadata,
        caps: SystemCapabilities | None = None,
    ) -> ValidationResult:
        """Is this cached engine still usable for this source and this machine?"""
        caps = caps or detect_capabilities()

        if not engine.exists():
            return ValidationResult(False, "no engine file")
        if engine.stat().st_size == 0:
            return ValidationResult(False, "engine file is empty")

        metadata = self.load_metadata(engine)
        if metadata is None:
            return ValidationResult(False, "no engine metadata sidecar")
        if metadata.schema_version != SCHEMA_VERSION:
            return ValidationResult(
                False, f"metadata schema {metadata.schema_version} != {SCHEMA_VERSION}"
            )

        current = source_fingerprint(source)
        if metadata.source_fingerprint != current:
            return ValidationResult(False, "source model changed since the engine was built")

        if metadata.build_fingerprint != expected.build_fingerprint:
            return ValidationResult(False, "build configuration changed")

        if self._strict:
            running_trt = caps.tensorrt.version
            if running_trt and metadata.tensorrt_version != running_trt:
                return ValidationResult(
                    False,
                    f"engine built with TensorRT {metadata.tensorrt_version}, "
                    f"running {running_trt}",
                )
            running_cc = caps.cuda.compute_capability
            if running_cc and metadata.compute_capability != running_cc:
                return ValidationResult(
                    False,
                    f"engine built for compute capability "
                    f"{metadata.compute_capability}, this GPU is {running_cc}",
                )

        return ValidationResult(True, "engine is current")

    def describe(self, source: Path, role: str, precision: str, batch: int) -> dict[str, Any]:
        """Report an engine's state without building anything."""
        engine = self.engine_path(source, role, precision, batch)
        metadata = self.load_metadata(engine)
        return {
            "role": role,
            "source": str(source),
            "engine": str(engine),
            "exists": engine.exists(),
            "size_mb": round(engine.stat().st_size / 1e6, 1) if engine.exists() else None,
            "metadata": metadata.to_dict() if metadata else None,
        }

    def list_engines(self) -> list[dict[str, Any]]:
        if not self._dir.exists():
            return []
        rows: list[dict[str, Any]] = []
        for engine in sorted(self._dir.glob(f"*{ENGINE_SUFFIX}")):
            metadata = self.load_metadata(engine)
            rows.append(
                {
                    "engine": str(engine),
                    "size_mb": round(engine.stat().st_size / 1e6, 1),
                    "role": metadata.role if metadata else None,
                    "precision": metadata.precision if metadata else None,
                    "source": metadata.source_model if metadata else None,
                    "tensorrt": metadata.tensorrt_version if metadata else None,
                    "built_at": metadata.built_at if metadata else None,
                    "valid_metadata": metadata is not None,
                }
            )
        return rows

    def remove(self, engine: Path) -> bool:
        removed = False
        for path in (engine, self.metadata_path(engine)):
            if path.exists():
                try:
                    path.unlink()
                    removed = True
                except OSError as exc:  # pragma: no cover - permissions
                    logger.warning("Cannot delete %s: %s", path, exc)
        return removed


def build_expected_metadata(
    *,
    role: str,
    source: Path,
    engine: Path,
    precision: str,
    input_shape: list[int],
    trt_config,
    caps: SystemCapabilities,
    source_hash: str | None = None,
) -> EngineMetadata:
    """The metadata an engine built *now, here* would carry."""
    return EngineMetadata(
        schema_version=SCHEMA_VERSION,
        role=role,
        source_model=str(source),
        source_fingerprint=source_hash if source_hash is not None else source_fingerprint(source),
        engine_path=str(engine),
        precision=precision,
        tensorrt_version=caps.tensorrt.version,
        cuda_version=caps.cuda.runtime_version,
        jetpack_version=caps.jetson.jetpack_version,
        gpu_name=caps.cuda.device_name,
        compute_capability=caps.cuda.compute_capability,
        input_shape=list(input_shape),
        min_batch_size=trt_config.min_batch_size,
        optimal_batch_size=trt_config.optimal_batch_size,
        max_batch_size=trt_config.max_batch_size,
        workspace_mb=trt_config.workspace_mb,
        builder_optimization_level=trt_config.builder_optimization_level,
        built_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    )
