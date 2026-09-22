"""Assembly of the face identity stack from configuration.

One place builds the detector, encoder, gallery, enroller, decision engine and
adapter, so the wiring is auditable and the same everywhere -- CLI, pipeline,
evaluation and tests all get an identical system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.paths import ProjectPaths
from src.config.schema import AppConfig
from src.core.exceptions import ModelLoadError
from src.face.detector import FaceDetector, YuNetFaceDetector
from src.face.scrfd import SCRFDFaceDetector
from src.identity.adaptation import AdaptationConfig, GalleryAdapter
from src.identity.calibration import CalibrationModel
from src.identity.decision import IdentityDecisionEngine, MatchingThresholds
from src.identity.face_enrollment import (
    FaceEnroller,
    FaceSelection,
    ReferenceQualityConfig,
)
from src.identity.face_gallery import FaceGallery
from src.identity.state_machine import StabilityConfig
from src.reid.encoder import ReIDEncoder
from src.utils.device import DeviceInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class FaceIdentitySystem:
    """Everything needed to enrol, match and decide identities."""

    detector: FaceDetector
    encoder: ReIDEncoder
    gallery: FaceGallery
    enroller: FaceEnroller
    engine: IdentityDecisionEngine
    adapter: GalleryAdapter
    calibration: CalibrationModel | None
    chip_size: int = 112
    owns_encoder: bool = True
    """False when the encoder is shared with the engine, which closes it."""

    def describe(self) -> dict[str, Any]:
        return {
            "face_detector": {
                "model": self.detector.info.model_path,
                "backend": self.detector.info.backend,
            },
            "face_encoder": {
                "model": self.encoder.info.model_path,
                "backend": self.encoder.info.backend,
                "dimension": self.encoder.info.embedding_dimension,
            },
            "gallery": {
                "root": str(self.gallery.root),
                "identities": len(self.gallery.active),
                "embeddings": self.gallery.embedding_count,
            },
            "calibration": {
                "present": self.calibration is not None,
                "usable": bool(self.calibration and self.calibration.is_usable),
                "conditions": (
                    sorted(self.calibration.conditions) if self.calibration else []
                ),
            },
            "adaptation_enabled": self.adapter.enabled,
        }

    def close(self) -> None:
        self.detector.close()
        if self.owns_encoder:
            self.encoder.close()


def build_face_detector(config: AppConfig, paths: ProjectPaths,
                        device: DeviceInfo) -> FaceDetector:
    """Construct the configured face detector.

    SCRFD is the default. Measured on this project's own degraded-condition
    set, its landmarks produce materially better alignment than YuNet's, and
    the difference shows up where it matters: genuine and impostor score
    distributions that cleanly separate rather than overlap. YuNet remains
    available as a faster option.
    """
    settings = config.face_identity
    if settings.face_detector_backend == "yunet":
        model = paths.resolve(config.face.detector_model)
        if not model.exists():
            raise ModelLoadError(
                f"YuNet model not found: {model}. "
                "Run: python scripts/fetch_face_models.py"
            )
        return YuNetFaceDetector(
            model,
            confidence=settings.face_detector_confidence,
            nms_threshold=settings.face_detector_nms,
        )

    model = paths.resolve(settings.face_detector_model)
    if not model.exists():
        raise ModelLoadError(
            f"SCRFD face detector not found: {model}. Fetch it with: "
            "python scripts/fetch_face_models.py --tier accurate, or set "
            "face_identity.face_detector_backend: yunet to use the smaller one."
        )
    size = settings.face_detector_input
    return SCRFDFaceDetector(
        model,
        confidence=settings.face_detector_confidence,
        nms_threshold=settings.face_detector_nms,
        input_size=(size, size),
        device=device,
    )


def build_face_encoder(config: AppConfig, paths: ProjectPaths,
                       device: DeviceInfo) -> ReIDEncoder:
    """Construct the face embedding model."""
    from src.face.embedder import ArcFaceOnnxEmbedder, SFaceEmbedder

    model = paths.resolve(config.face_identity.face_encoder_model)
    if not model.exists():
        raise ModelLoadError(
            f"face encoder not found: {model}. "
            "Run: python scripts/fetch_face_models.py"
        )
    if "sface" in model.name.lower():
        return SFaceEmbedder(model, device, batch_size=config.recognition.batch.max_size)
    return ArcFaceOnnxEmbedder(
        model,
        device,
        batch_size=config.recognition.batch.max_size,
        input_mean=config.face.arcface_input_mean,
        input_scale=config.face.arcface_input_scale,
    )



def face_encoder_path(config: AppConfig, paths: ProjectPaths) -> Path:
    """Where the face identity encoder's weights live."""
    return paths.resolve(config.face_identity.face_encoder_model)


def build_face_identity_system(
    config: AppConfig,
    paths: ProjectPaths,
    device: DeviceInfo,
    *,
    load: bool = True,
    encoder: ReIDEncoder | None = None,
) -> FaceIdentitySystem:
    """Assemble the whole face identity stack.

    ``encoder`` lets a caller that already holds the same face embedding model
    share it rather than loading a second copy of the weights. It is the
    caller's job to confirm the models match; see
    :func:`face_encoder_path`.
    """
    settings = config.face_identity

    detector = build_face_detector(config, paths, device)
    shared_encoder = encoder is not None
    encoder = encoder or build_face_encoder(config, paths, device)
    if load:
        detector.load()
        if not shared_encoder:
            encoder.load()

    gallery = FaceGallery(
        paths.resolve(settings.gallery_dir),
        max_live_per_identity=settings.max_live_embeddings,
    )

    reference_quality = ReferenceQualityConfig(
        enabled=config.reference_quality.enabled,
        min_interocular_px=config.reference_quality.min_interocular_px,
        min_face_px=config.reference_quality.min_face_px,
        min_sharpness=config.reference_quality.min_sharpness,
        min_exposure=config.reference_quality.min_exposure,
        max_yaw_deg=config.reference_quality.max_yaw_deg,
        max_pitch_deg=config.reference_quality.max_pitch_deg,
        min_landmark_score=config.reference_quality.min_landmark_score,
        min_overall_quality=config.reference_quality.min_overall_quality,
        require_full_face=config.reference_quality.require_full_face,
        fail_on_warnings=config.reference_quality.fail_on_warnings,
        max_faces=config.reference_quality.max_faces,
        selection=FaceSelection(config.reference_quality.selection),
    )
    enroller = FaceEnroller(
        detector, encoder, reference_quality, chip_size=settings.chip_size
    )

    calibration = CalibrationModel.load(paths.resolve(settings.calibration_file))
    if calibration is not None and not calibration.is_usable:
        logger.warning(
            "A calibration file exists but none of its conditions have enough "
            "samples to be usable; no calibrated confidence will be reported. "
            "Re-run: python main.py calibrate --dataset <dir>"
        )

    engine = IdentityDecisionEngine(
        gallery,
        MatchingThresholds(
            full=settings.threshold_full,
            partial=settings.threshold_partial,
            masked=settings.threshold_masked,
            ambiguity_margin=settings.ambiguity_margin,
            min_quality=settings.min_face_quality,
            min_identity_confidence=settings.min_identity_confidence,
        ),
        calibration=calibration,
        stability=StabilityConfig(
            history_size=config.temporal.history_size,
            min_confirmation_frames=config.temporal.min_confirmation_frames,
            switch_margin=config.temporal.switch_margin,
            lost_track_timeout=config.temporal.lost_track_timeout,
            occlusion_hold_frames=config.temporal.occlusion_hold_frames,
            unknown_frames_to_release=config.temporal.unknown_frames_to_release,
            min_evidence_weight=config.temporal.min_evidence_weight,
        ),
        fallback_threshold=settings.fallback_threshold,
    )

    adaptation = config.online_adaptation
    adapter = GalleryAdapter(
        gallery,
        AdaptationConfig(
            enabled=adaptation.enabled,
            min_similarity=adaptation.min_similarity,
            similarity_headroom=adaptation.similarity_headroom,
            min_quality=adaptation.min_quality,
            min_confirmation_frames=adaptation.min_confirmation_frames,
            min_margin=adaptation.min_margin,
            require_full_face=adaptation.require_full_face,
            min_track_stability=adaptation.min_track_stability,
            max_samples_per_identity=adaptation.max_samples_per_identity,
            min_novelty=adaptation.min_novelty,
            max_novelty=adaptation.max_novelty,
            cooldown_seconds=adaptation.cooldown_seconds,
            max_per_track=adaptation.max_per_track,
        ),
    )

    if load:
        gallery.load(encoder.info.fingerprint)
        if calibration is None:
            logger.info(
                "No calibration found, so identity confidence will not be "
                "reported and the fallback threshold %.2f is in use. Fit one "
                "with: python main.py calibrate --dataset <evaluation dir>",
                settings.fallback_threshold,
            )

    return FaceIdentitySystem(
        detector=detector,
        encoder=encoder,
        gallery=gallery,
        enroller=enroller,
        engine=engine,
        adapter=adapter,
        calibration=calibration,
        chip_size=settings.chip_size,
        owns_encoder=not shared_encoder,
    )
