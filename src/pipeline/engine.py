"""Composition root.

Builds the detector, encoder, gallery, matcher and pipeline from a validated
configuration and wires them together. Every component takes its collaborators
as constructor arguments (dependency injection), so tests and the API layer can
substitute fakes without monkey-patching.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.backends.detector_backend import prepare_detector_model
from src.config.paths import ProjectPaths
from src.config.schema import AppConfig, RecognitionMode
from src.core.exceptions import GalleryError
from src.detection.detector import Detector
from src.detection.yolo26_detector import YOLO26Detector, build_tracker_config
from src.events.manager import EventManager
from src.face.detector import FaceDetector
from src.face.factory import build_face_detector, build_face_encoder
from src.hardware.capabilities import detect_capabilities
from src.identity.enrollment import Enroller
from src.identity.factory import (
    FaceIdentitySystem,
    build_face_identity_system,
    face_encoder_path,
)
from src.identity.gallery import BuildReport, IdentityGallery
from src.identity.matcher import IdentityMatcher
from src.pipeline.face_identity_processor import FaceIdentityPipeline
from src.pipeline.metrics import MetricsCollector
from src.pipeline.processor import ReIDPipeline
from src.reid.encoder import ReIDEncoder
from src.reid.factory import build_encoder, resolve_model_path
from src.reid.preprocess import CropPreprocessor, build_preprocessor
from src.utils.device import DeviceInfo, log_device_info, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Engine:
    """A fully wired, ready-to-run system."""

    config: AppConfig
    paths: ProjectPaths
    device: DeviceInfo
    detector: Detector
    encoder: ReIDEncoder
    preprocessor: CropPreprocessor
    gallery: IdentityGallery
    matcher: IdentityMatcher
    events: EventManager
    metrics: MetricsCollector
    face_detector: FaceDetector | None = None
    """Present only in face mode; the person detector is always YOLO26."""
    face_identity: FaceIdentitySystem | None = None
    """The passport-photo identity stack: SCRFD, the face gallery, calibrated
    matching and temporal evidence. Present when it is configured *and* has
    people enrolled -- an empty face gallery would name nobody, so the older
    path stays in charge until ``identity build`` has been run."""
    backends: dict[str, Any] = field(default_factory=dict)
    """How each model's runtime was chosen, for logs and `system info`."""

    @property
    def uses_face_identity(self) -> bool:
        """Whether the passport-photo identity stack is the active path."""
        return self.face_identity is not None and not self.face_identity.gallery.is_empty

    def pipeline(self, *, use_tracking: bool = True):
        """Create a pipeline sharing this engine's models and gallery.

        Which pipeline depends on what is enrolled: with a populated face
        identity gallery the passport-photo stack runs, otherwise the original
        path does. The choice is reported by ``describe()`` and logged at
        startup, so it is never silent.
        """
        if self.uses_face_identity:
            return FaceIdentityPipeline(
                self.config,
                self.detector,
                self.face_identity,
                metrics=self.metrics,
                use_tracking=use_tracking,
            )
        return ReIDPipeline(
            self.config,
            self.detector,
            self.encoder,
            self.preprocessor,
            self.gallery,
            self.matcher,
            metrics=self.metrics,
            use_tracking=use_tracking,
        )

    def enroller(self) -> Enroller:
        return Enroller(
            self.config,
            self.detector,
            self.encoder,
            self.preprocessor,
            crop_dir=self.paths.enrollment_crop_dir,
        )

    def build_gallery(self, *, force: bool = False, only=None) -> BuildReport:
        """Ensure the gallery is current, enrolling whatever is missing."""
        return self.gallery.build(
            self.enroller(), self.encoder.info, force=force, only=only
        )

    def ensure_gallery(self) -> None:
        """Load the gallery for a processing run, building it when allowed."""
        if self.config.gallery.auto_build:
            report = self.build_gallery()
            if report.failed:
                for identity_id, message in report.failed.items():
                    logger.error(
                        "Identity unavailable", extra={"identity": identity_id, "error": message}
                    )
        else:
            self.gallery.load(self.encoder.info)

        if not self.gallery.active_identities:
            logger.warning(
                "The identity gallery is empty: every detected person will be "
                "reported as '%s'. Add people to the configuration and run "
                "'gallery build'.",
                self.config.matching.unknown_label,
            )

    def describe(self) -> dict[str, Any]:
        """Serialisable system description (startup log, API ``/health``)."""
        return {
            "application": self.config.application.name,
            "version": self.config.application.version,
            "device": {
                "device": self.device.device,
                "detail": self.device.description,
                "fp16": self.device.fp16_enabled,
            },
            "detector": {
                "model": self.detector.info.model_path,
                "backend": self.detector.info.backend,
                "imgsz": self.detector.info.imgsz,
                "load_time_s": round(self.detector.info.load_time_s, 3),
            },
            "backends": self.backends,
            "recognition": {
                "mode": self.config.recognition.mode.value,
                "clothing_invariant": (
                    self.config.recognition.mode is RecognitionMode.FACE
                ),
                "face_detector": (
                    self.face_detector.info.model_path if self.face_detector else None
                ),
            },
            "reid": {
                "model": self.encoder.info.model_path,
                "backend": self.encoder.info.backend,
                "input_size": list(self.encoder.info.input_size),
                "embedding_dimension": self.encoder.info.embedding_dimension,
                "load_time_s": round(self.encoder.info.load_time_s, 3),
            },
            "matching": {
                "metric": self.config.matching.metric.value,
                "recognition_threshold": self.config.matching.recognition_threshold,
                "high_confidence_threshold": self.config.matching.high_confidence_threshold,
            },
            "tracking": {
                "enabled": self.config.tracking.enabled,
                "tracker": self.config.tracking.tracker,
                "with_reid": self.config.tracking.with_reid,
                "stabilization": self.config.tracking.identity_stability.strategy.value,
            },
            "gallery": {
                "identities": len(self.gallery.active_identities),
                "directory": str(self.paths.gallery_dir),
            },
            "identity_engine": (
                {
                    "path": "face_identity",
                    **self.face_identity.describe(),
                }
                if self.uses_face_identity
                else {"path": "person_gallery"}
            ),
        }

    def close(self) -> None:
        """Release every native model handle. Idempotent.

        The detector, face detector and encoder each wrap native resources
        (torch, OpenCV DNN, ONNX Runtime) holding hundreds of megabytes. The
        collection makes that memory come back promptly rather than whenever
        the collector next runs, which matters for a process that opens and
        closes engines. close() runs once per engine, never on a hot path.
        """
        self.detector.close()
        self.encoder.close()
        if self.face_detector is not None:
            self.face_detector.close()
        if self.face_identity is not None:
            self.face_identity.close()
        self.events.close()
        gc.collect()


def build_engine(
    config: AppConfig,
    *,
    load_models: bool = True,
    events: EventManager | None = None,
) -> Engine:
    """Create an :class:`Engine` from a validated configuration."""
    paths = ProjectPaths.from_config(config)
    paths.ensure(paths.gallery_dir, paths.embeddings_dir, paths.metadata_dir)

    caps = detect_capabilities()
    device = resolve_device(config.device)
    log_device_info(device)
    if caps.is_jetson:
        logger.info(
            "Jetson platform detected",
            extra={
                "model": caps.jetson.model or "unknown",
                "jetpack": caps.jetson.jetpack_version or "unknown",
                "tensorrt": caps.tensorrt.version or "unavailable",
            },
        )

    detector_path = resolve_model_path(config.models.detector, paths)
    # TensorRT is a drop-in acceleration for the detector: Ultralytics executes
    # a .engine through its own AutoBackend, so the decode path is unchanged.
    detector_plan = prepare_detector_model(config, paths, detector_path, caps)
    detector_path = detector_plan.model_path
    tracker_path: Path | None = None
    if config.tracking.enabled:
        tracker_reid = config.tracking.reid_model
        if tracker_reid == "gallery":
            tracker_reid = resolve_model_path(config.models.reid, paths)
        tracker_path = build_tracker_config(config, paths.output_dir / ".trackers",
                                            reid_model=tracker_reid)

    detector = YOLO26Detector(config, device, detector_path, tracker_path)

    # Recognition modality. YOLO26 finds the people either way; what changes is
    # which pixels become the identity embedding -- the whole body, or the face
    # alone. Face mode is clothing-invariant, which is why it is the default.
    face_detector: FaceDetector | None = None
    if config.recognition.mode is RecognitionMode.FACE:
        face_detector = build_face_detector(config, paths)
        encoder = build_face_encoder(config, paths, device)
    else:
        encoder = build_encoder(config, paths, device)
    preprocessor = build_preprocessor(config, face_detector=face_detector)

    if load_models:
        detector.load()
        if face_detector is not None:
            face_detector.load()
        encoder.load()

    gallery = IdentityGallery(config, paths)
    matcher = IdentityMatcher(config.matching)

    # The passport-photo identity stack. It is built whenever it is configured
    # and people have been enrolled into it, and it then becomes the identity
    # path; an empty face gallery leaves the original path in charge rather
    # than producing a system that can only ever answer "unknown".
    face_identity: FaceIdentitySystem | None = None
    if config.face_identity.enabled and config.recognition.mode is RecognitionMode.FACE:
        try:
            # Share the face embedding model when both paths resolve to the
            # same weights, rather than loading a second copy of them.
            shared = (
                encoder
                if Path(encoder.info.model_path or "").resolve()
                == face_encoder_path(config, paths).resolve()
                else None
            )
            face_identity = build_face_identity_system(
                config, paths, device, load=False, encoder=shared
            )
            enrolled = face_identity.gallery.load()
            if enrolled == 0:
                logger.debug(
                    "The face identity gallery at %s is empty; using the "
                    "person gallery. Run: python main.py identity build",
                    face_identity.gallery.root,
                )
                face_identity.close()
                face_identity = None
            elif load_models:
                face_identity.detector.load()
                if face_identity.owns_encoder:
                    face_identity.encoder.load()
        except Exception as exc:  # noqa: BLE001 - a missing model must not abort startup
            logger.warning(
                "The face identity stack could not be built, so the person "
                "gallery path will be used: %s",
                exc,
            )
            face_identity = None

    event_manager = events or EventManager(
        enabled=config.events.enabled,
        log_path=(paths.events_dir / config.events.filename)
        if config.events.log_to_file
        else None,
        console=config.events.console,
    )

    engine = Engine(
        config=config,
        paths=paths,
        device=device,
        detector=detector,
        encoder=encoder,
        preprocessor=preprocessor,
        gallery=gallery,
        matcher=matcher,
        events=event_manager,
        metrics=MetricsCollector(),
        face_detector=face_detector,
        face_identity=face_identity,
        backends={
            "detector": detector_plan.decision.to_dict()
            | {"model": detector_plan.model_path, "note": detector_plan.note},
        },
    )
    if load_models:
        logger.info(
            "Engine ready",
            extra={
                "mode": config.recognition.mode.value,
                "identity": (
                    "face_identity" if engine.uses_face_identity else "person_gallery"
                ),
                "detector": Path(detector.info.model_path).name,
                "encoder": Path(encoder.info.model_path).name,
                "dim": encoder.info.embedding_dimension,
                "device": device.device,
            },
        )
    return engine


def require_identities(engine: Engine) -> None:
    """Fail loudly when a command needs a populated gallery."""
    if not engine.gallery.active_identities:
        raise GalleryError(
            "no usable identities in the gallery. Add entries under 'people:' in "
            "the configuration and run: python main.py gallery build"
        )
