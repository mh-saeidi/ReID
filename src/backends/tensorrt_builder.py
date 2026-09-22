"""TensorRT engine construction from ONNX sources.

Two build paths are supported, in preference order:

1. the TensorRT Python API, which gives explicit control over optimisation
   profiles (dynamic batch) and precision flags;
2. ``trtexec``, the JetPack-bundled builder, used when the Python bindings are
   absent -- a common state on stock JetPack images.

Neither path is reachable off a CUDA machine, and this module is careful to say
so clearly instead of failing obscurely: every unsupported combination raises
:class:`TensorRTUnavailable` with the reason and the suggested fallback.

INT8 is deliberately awkward to enable. Quantising a face encoder shifts the
embedding distribution, which moves every similarity threshold that was
calibrated against it -- so it requires both an explicit precision setting and
``allow_int8_for_face``, and it still warns.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.backends.engine_store import (
    EngineMetadata,
    EngineStore,
    build_expected_metadata,
    source_fingerprint,
)
from src.core.exceptions import ModelLoadError
from src.hardware.capabilities import SystemCapabilities, detect_capabilities
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TensorRTUnavailable(ModelLoadError):
    """TensorRT cannot build or run this model here; the caller should fall back."""


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """One engine to build."""

    role: str
    source: Path
    input_shape: Sequence[int]
    """``(C, H, W)`` -- the batch dimension is supplied by the profile."""
    input_name: str | None = None
    dynamic_batch: bool = True

    @property
    def chw(self) -> tuple[int, int, int]:
        c, h, w = self.input_shape[-3:]
        return int(c), int(h), int(w)


@dataclass(frozen=True, slots=True)
class BuildResult:
    engine_path: Path
    metadata: EngineMetadata
    rebuilt: bool
    build_seconds: float
    backend: str
    """``python`` or ``trtexec``."""


def _precision_flags(precision: str, request: BuildRequest, trt_config) -> tuple[bool, bool]:
    """(fp16, int8), with the face-recognition guard applied."""
    fp16 = precision in ("fp16", "int8")
    int8 = precision == "int8"
    if int8 and request.role in ("face_encoder", "body_encoder"):
        if not trt_config.allow_int8_for_face:
            raise TensorRTUnavailable(
                f"INT8 was requested for '{request.role}', but "
                "backend.tensorrt.allow_int8_for_face is false. Quantising a "
                "recognition encoder changes the embedding distribution, which "
                "invalidates the calibrated similarity thresholds. Re-run "
                "'main.py evaluate' after enabling it, and re-calibrate "
                "matching.recognition_threshold before deploying."
            )
        logger.warning(
            "Building an INT8 engine for %s. Similarity thresholds calibrated "
            "against the FP16/FP32 model are NOT valid for it -- re-run "
            "'main.py evaluate' and re-calibrate before deploying.",
            request.role,
        )
    return fp16, int8


class TensorRTBuilder:
    """Builds and caches TensorRT engines."""

    def __init__(self, trt_config, store: EngineStore,
                 caps: SystemCapabilities | None = None) -> None:
        self._config = trt_config
        self._store = store
        self._caps = caps or detect_capabilities()

    @property
    def store(self) -> EngineStore:
        return self._store

    # ------------------------------------------------------------ interface
    def ensure_engine(self, request: BuildRequest, *, force: bool = False) -> BuildResult:
        """Return a valid engine for ``request``, building it when necessary."""
        source = Path(request.source)
        if not source.exists():
            raise TensorRTUnavailable(f"source model not found: {source}")
        if source.suffix.lower() in (".engine", ".plan", ".trt"):
            return self._adopt_prebuilt(request, source)
        if source.suffix.lower() != ".onnx":
            raise TensorRTUnavailable(
                f"cannot build a TensorRT engine from '{source.suffix}'. "
                "Export the model to ONNX first (Ultralytics: "
                "`yolo export model=... format=onnx dynamic=True`)."
            )

        precision = self._config.precision.value
        engine = self._store.engine_path(
            source, request.role, precision, self._config.max_batch_size
        )
        source_hash = source_fingerprint(source)
        expected = build_expected_metadata(
            role=request.role,
            source=source,
            engine=engine,
            precision=precision,
            input_shape=list(request.input_shape),
            trt_config=self._config,
            caps=self._caps,
            source_hash=source_hash,
        )

        if not force:
            verdict = self._store.validate(engine, source, expected, self._caps)
            if verdict:
                metadata = self._store.load_metadata(engine)
                logger.info(
                    "Reusing TensorRT engine",
                    extra={"role": request.role, "engine": engine.name},
                )
                return BuildResult(engine, metadata or expected, False, 0.0, "cache")
            logger.info(
                "Rebuilding TensorRT engine",
                extra={"role": request.role, "why": verdict.reason},
            )

        if not self._config.allow_build:
            raise TensorRTUnavailable(
                f"no valid engine for '{request.role}' and "
                "backend.tensorrt.allow_build is false. Build it ahead of time "
                "with: python main.py models build-tensorrt"
            )

        started = time.perf_counter()
        backend = self._build(request, source, engine, precision)
        elapsed = time.perf_counter() - started

        expected.build_seconds = round(elapsed, 2)
        expected.extra = {"builder": backend}
        self._store.save_metadata(expected)
        logger.info(
            "TensorRT engine built",
            extra={
                "role": request.role,
                "engine": engine.name,
                "precision": precision,
                "seconds": round(elapsed, 1),
                "builder": backend,
            },
        )
        return BuildResult(engine, expected, True, elapsed, backend)

    def _adopt_prebuilt(self, request: BuildRequest, source: Path) -> BuildResult:
        """Use an engine the operator supplied directly.

        No provenance is available, so this is reported plainly rather than
        pretending the engine was validated.
        """
        logger.warning(
            "Using a pre-built TensorRT engine supplied directly; its provenance "
            "cannot be verified and it will not be rebuilt automatically",
            extra={"role": request.role, "engine": source.name},
        )
        metadata = build_expected_metadata(
            role=request.role,
            source=source,
            engine=source,
            precision=self._config.precision.value,
            input_shape=list(request.input_shape),
            trt_config=self._config,
            caps=self._caps,
        )
        metadata.extra = {"builder": "external", "verified": False}
        return BuildResult(source, metadata, False, 0.0, "external")

    # -------------------------------------------------------------- builds
    def _build(self, request: BuildRequest, source: Path, engine: Path,
               precision: str) -> str:
        engine.parent.mkdir(parents=True, exist_ok=True)
        fp16, int8 = _precision_flags(precision, request, self._config)

        if self._caps.tensorrt.python_bindings:
            try:
                self._build_with_python(request, source, engine, fp16, int8)
                return "python"
            except TensorRTUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - fall through to trtexec
                logger.warning(
                    "TensorRT Python build failed (%s); trying trtexec", exc
                )

        if self._caps.tensorrt.trtexec:
            self._build_with_trtexec(request, source, engine, fp16, int8)
            return "trtexec"

        raise TensorRTUnavailable(
            "no usable TensorRT builder: install the tensorrt Python bindings "
            "or make trtexec available on PATH"
        )

    def _build_with_python(self, request: BuildRequest, source: Path, engine: Path,
                           fp16: bool, int8: bool) -> None:  # pragma: no cover - needs TRT
        import tensorrt as trt  # noqa: PLC0415

        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger_trt)

        with source.open("rb") as handle:
            if not parser.parse(handle.read()):
                errors = "; ".join(
                    str(parser.get_error(i)) for i in range(parser.num_errors)
                )
                raise TensorRTUnavailable(
                    f"ONNX parse failed for {source.name}: {errors}. "
                    "The model may use an operator this TensorRT version does "
                    "not implement; use the ONNX Runtime backend instead."
                )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, self._config.workspace_mb * 1024 * 1024
        )
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = self._config.builder_optimization_level
        if fp16:
            if not builder.platform_has_fast_fp16:
                logger.warning("This platform has no fast FP16; the engine may not speed up")
            config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            if not builder.platform_has_fast_int8:
                raise TensorRTUnavailable("this platform has no fast INT8 support")
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = self._make_calibrator(request)
            if calibrator is not None:
                config.int8_calibrator = calibrator

        if self._config.timing_cache:
            cache_path = engine.with_suffix(".timing_cache")
            buffer = cache_path.read_bytes() if cache_path.exists() else b""
            cache = config.create_timing_cache(buffer)
            config.set_timing_cache(cache, ignore_mismatch=False)

        input_tensor = network.get_input(0)
        channels, height, width = request.chw
        # A dynamic batch axis (reported as -1) *requires* an optimisation
        # profile; building without one fails. The declared intent is honoured
        # where it can be, but the graph's actual shape decides.
        graph_batch = int(input_tensor.shape[0]) if len(input_tensor.shape) else 1
        needs_profile = request.dynamic_batch or graph_batch < 0
        if needs_profile and not request.dynamic_batch:
            logger.info(
                "The ONNX graph has a dynamic batch axis, so an optimisation "
                "profile is added even though a fixed batch was requested",
                extra={"role": request.role},
            )
        if needs_profile:
            profile = builder.create_optimization_profile()
            # The detector is fed one frame at a time, so a wide profile would
            # cost build time and engine size for a range never exercised.
            if request.dynamic_batch:
                minimum = self._config.min_batch_size
                optimal = self._config.optimal_batch_size
                maximum = self._config.max_batch_size
            else:
                minimum = optimal = maximum = 1
            profile.set_shape(
                input_tensor.name,
                (minimum, channels, height, width),
                (optimal, channels, height, width),
                (maximum, channels, height, width),
            )
            config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise TensorRTUnavailable(
                f"TensorRT failed to build an engine for {source.name}; "
                "see the TensorRT log above. Falling back to ONNX Runtime."
            )
        engine.write_bytes(bytes(serialized))

        if self._config.timing_cache:
            try:
                cache = config.get_timing_cache()
                if cache is not None:
                    engine.with_suffix(".timing_cache").write_bytes(
                        bytes(cache.serialize())
                    )
            except Exception:  # noqa: BLE001 - the cache is an optimisation only
                pass

    def _make_calibrator(self, request: BuildRequest):  # pragma: no cover - needs TRT
        directory = self._config.int8_calibration_dir
        if not directory:
            raise TensorRTUnavailable(
                "INT8 requires backend.tensorrt.int8_calibration_dir"
            )
        from src.backends.int8_calibrator import ImageFolderCalibrator  # noqa: PLC0415

        return ImageFolderCalibrator(
            Path(directory), request.chw, batch_size=self._config.optimal_batch_size
        )

    def _build_with_trtexec(self, request: BuildRequest, source: Path, engine: Path,
                            fp16: bool, int8: bool) -> None:  # pragma: no cover - needs TRT
        channels, height, width = request.chw
        name = request.input_name or "images"
        command: list[str] = [
            str(self._caps.tensorrt.trtexec),
            f"--onnx={source}",
            f"--saveEngine={engine}",
            f"--memPoolSize=workspace:{self._config.workspace_mb}M",
        ]
        if request.dynamic_batch:
            command += [
                f"--minShapes={name}:{self._config.min_batch_size}x{channels}x{height}x{width}",
                f"--optShapes={name}:{self._config.optimal_batch_size}x{channels}x{height}x{width}",
                f"--maxShapes={name}:{self._config.max_batch_size}x{channels}x{height}x{width}",
            ]
        if fp16:
            command.append("--fp16")
        if int8:
            command.append("--int8")

        logger.info("Running trtexec", extra={"engine": engine.name})
        try:
            result = subprocess.run(  # noqa: S603 - arguments are built here, not user shell
                command, capture_output=True, text=True, check=False, timeout=3600
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TensorRTUnavailable(f"trtexec could not be executed: {exc}") from exc

        if result.returncode != 0 or not engine.exists():
            tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-15:])
            raise TensorRTUnavailable(
                f"trtexec failed for {source.name} (exit {result.returncode}).\n{tail}\n"
                "If this names an unsupported operator, use the ONNX Runtime "
                "backend for this model: backend.face_encoder: onnx"
            )


def default_build_requests(config, paths) -> list[BuildRequest]:
    """The engines a given configuration needs, derived from the config itself."""
    from src.config.schema import RecognitionMode  # noqa: PLC0415

    requests: list[BuildRequest] = []

    detector = paths.resolve(config.models.detector)
    if detector.suffix.lower() == ".onnx":
        size = config.detector.imgsz
        requests.append(
            BuildRequest("detector", detector, (3, size, size), input_name="images",
                         dynamic_batch=False)
        )

    if config.recognition.mode is RecognitionMode.FACE:
        # Both face paths may be present: the passport-photo identity engine
        # has its own encoder setting, and it is the one that runs once people
        # are enrolled into the face gallery. Build an engine for each distinct
        # model so whichever path is active is accelerated.
        seen: set[Path] = set()
        candidates = [
            (paths.resolve(config.face_identity.face_encoder_model),
             config.face_identity.chip_size),
            (paths.resolve(config.face.recognition_model), config.face.chip_size),
        ]
        for encoder, chip in candidates:
            if encoder in seen or encoder.suffix.lower() != ".onnx":
                continue
            seen.add(encoder)
            requests.append(
                BuildRequest("face_encoder", encoder, (3, chip, chip), input_name="input")
            )
    else:
        body = paths.resolve(config.models.reid)
        if body.suffix.lower() == ".onnx":
            height, width = config.reid.size_hw
            requests.append(
                BuildRequest("body_encoder", body, (3, height, width), input_name="images")
            )
    return requests
