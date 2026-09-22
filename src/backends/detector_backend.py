"""TensorRT preparation for the YOLO26 person detector.

Ultralytics already knows how to execute a ``.engine`` file through its own
AutoBackend, including the NMS-free YOLO26 decode. Re-implementing that here
would duplicate postprocessing that is easy to get subtly wrong, so this module
only does the part Ultralytics does not: decide whether TensorRT applies, make
sure a valid engine exists, and hand back the path to load.

The detector's batch dimension stays fixed at 1. Frames arrive one at a time
from a live camera, so a dynamic batch profile would cost build time and engine
size for a dimension that is never exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.backends.engine_store import EngineStore
from src.backends.selection import BackendDecision, ModelRole, select_backend
from src.backends.tensorrt_builder import BuildRequest, TensorRTBuilder, TensorRTUnavailable
from src.config.schema import InferenceBackend
from src.hardware.capabilities import SystemCapabilities, detect_capabilities
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DetectorModelPlan:
    """Which detector artefact to load, and how it was chosen."""

    model_path: str
    decision: BackendDecision
    engine_built: bool = False
    note: str = ""

    @property
    def uses_tensorrt(self) -> bool:
        return self.decision.backend is InferenceBackend.TENSORRT


def prepare_detector_model(
    config,
    paths,
    source_path: str,
    caps: SystemCapabilities | None = None,
) -> DetectorModelPlan:
    """Resolve the detector model, building a TensorRT engine when applicable."""
    caps = caps or detect_capabilities()
    decision = select_backend(ModelRole.DETECTOR, source_path, config, caps)

    if decision.backend is not InferenceBackend.TENSORRT:
        return DetectorModelPlan(source_path, decision)

    source = Path(source_path)
    if source.suffix.lower() in (".engine", ".plan", ".trt"):
        return DetectorModelPlan(source_path, decision, note="pre-built engine supplied")

    trt_config = config.backend.tensorrt
    store = EngineStore(
        paths.resolve(trt_config.engine_dir),
        strict_version_check=trt_config.strict_version_check,
    )
    builder = TensorRTBuilder(trt_config, store, caps)
    size = config.detector.imgsz
    request = BuildRequest(
        role="detector",
        source=source,
        input_shape=(3, size, size),
        input_name="images",
        dynamic_batch=False,   # one frame at a time from a live source
    )

    try:
        result = builder.ensure_engine(request)
    except TensorRTUnavailable as exc:
        fallback = select_backend(
            ModelRole.DETECTOR, source_path, config, caps
        )
        logger.warning(
            "Detector TensorRT engine unavailable; using the original model",
            extra={"reason": str(exc)},
        )
        return DetectorModelPlan(source_path, fallback, note=str(exc))
    except Exception as exc:  # noqa: BLE001 - never block startup on TRT
        logger.warning(
            "Detector TensorRT build failed; using the original model",
            extra={"error": str(exc)},
        )
        return DetectorModelPlan(source_path, decision, note=str(exc))

    logger.info(
        "Detector will run on TensorRT",
        extra={"engine": result.engine_path.name, "rebuilt": result.rebuilt},
    )
    return DetectorModelPlan(
        str(result.engine_path), decision, engine_built=result.rebuilt
    )
