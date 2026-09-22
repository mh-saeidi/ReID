"""Pydantic configuration schema.

The whole application is driven by this schema; there are no hard-coded model
paths, thresholds or filesystem locations in the business logic. Validation
happens once at startup and produces actionable error messages.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """Base model: unknown keys are rejected so typos surface immediately."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class DeviceKind(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


class SelectionStrategy(str, Enum):
    LARGEST_PERSON = "largest_person"
    HIGHEST_CONFIDENCE = "highest_confidence"
    CENTER_MOST = "center_most"


class RecognitionMode(str, Enum):
    """Explicit recognition mode -- the system never switches silently.

    ``face`` identifies people from facial appearance only, which makes the
    decision independent of clothing. ``person_reid`` uses whole-body
    appearance, which works from behind and at a distance but is dominated by
    what the person is wearing.
    """

    PERSON_REID = "person_reid"
    FACE = "face"


class FaceBackend(str, Enum):
    AUTO = "auto"
    OPENCV = "opencv"
    """YuNet detector + SFace recogniser, both built into OpenCV."""
    ARCFACE_ONNX = "arcface_onnx"
    """YuNet detector + an ArcFace-format ONNX recogniser (higher accuracy)."""


class FaceSearchRegion(str, Enum):
    UPPER_BODY = "upper_body"
    """Search the head region of each person box (default: most robust)."""
    FULL_BOX = "full_box"
    FRAME = "frame"
    """One detection pass per frame, faces assigned to person boxes (fastest)."""


class SnapshotMode(str, Enum):
    ALL = "all"
    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"
    EVENTS_ONLY = "events_only"
    DISABLED = "disabled"


class RecordingMode(str, Enum):
    CONTINUOUS = "continuous"
    EVENT = "event"
    DISABLED = "disabled"


class SourceKind(str, Enum):
    WEBCAM = "webcam"
    VIDEO = "video"
    IMAGE = "image"
    DIRECTORY = "directory"
    STREAM = "stream"
    JETSON_CAMERA = "jetson_camera"
    """CSI camera via nvarguscamerasrc, or a V4L2 device through NVMM."""


class ReIDBackend(str, Enum):
    AUTO = "auto"
    ULTRALYTICS = "ultralytics"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"


class StabilizationStrategy(str, Enum):
    WEIGHTED_VOTE = "weighted_vote"
    EMA = "ema"
    MAJORITY = "majority"


class InferenceBackend(str, Enum):
    """Inference runtime for a single model."""

    AUTO = "auto"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENCV = "opencv"
    """OpenCV DNN (YuNet / SFace) -- no separate runtime needed."""


class TensorRTPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class OverloadPolicy(str, Enum):
    """What to shed first when the output stage cannot keep up."""

    DROP_VISUALIZATION = "drop_visualization"
    """Drop preview/annotated frames; never drop events or event snapshots."""
    DROP_ALL_OUTPUT = "drop_all_output"
    """Also drop periodic metadata; still never drops recognition events."""
    BLOCK = "block"
    """Apply back-pressure to the pipeline instead of dropping anything."""


class RecordingBackend(str, Enum):
    AUTO = "auto"
    OPENCV = "opencv"
    GSTREAMER = "gstreamer"
    """NVIDIA hardware encoders via GStreamer (nvv4l2h264enc/h265enc)."""


class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    MP4V = "mp4v"


class SchedulerState(str, Enum):
    """Why the scheduler thinks a track does or does not need recognition."""

    NEW_TRACK = "new_track"
    UNKNOWN = "unknown"
    LOW_CONFIDENCE = "low_confidence"
    RECOGNIZED = "recognized"
    NO_FACE = "no_face"
    LOST = "lost"
    RECOVERING = "recovering"


class EmbeddingStoreKind(str, Enum):
    LOCAL_NUMPY = "local_numpy"
    # Future: faiss / qdrant / milvus / pgvector -- see src/identity/embedding_store.py


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


class TensorRTConfig(StrictModel):
    """TensorRT engine building and loading.

    Engines are *derived artefacts*: the ONNX/PyTorch model stays the source of
    truth, and an engine is rebuilt whenever the source or any build parameter
    changes. Engines are never assumed portable between machines -- a TensorRT
    engine is tied to the GPU architecture, TensorRT version and CUDA version it
    was built on, all of which are recorded in the engine's metadata sidecar.
    """

    enabled: bool | Literal["auto"] = "auto"
    """``auto`` uses TensorRT when the platform provides it; ``false`` never does."""
    precision: TensorRTPrecision = TensorRTPrecision.FP16
    """FP16 is the Jetson deployment default. INT8 is opt-in only."""
    allow_int8_for_face: bool = False
    """INT8 face recognition must be validated before use: quantisation shifts
    the embedding distribution, which moves every similarity threshold. This
    flag has to be set explicitly, and the engine build still warns."""
    int8_calibration_dir: str | None = None
    workspace_mb: int = Field(default=1024, ge=64, le=32768)
    max_batch_size: int = Field(default=8, ge=1, le=128)
    min_batch_size: int = Field(default=1, ge=1)
    optimal_batch_size: int = Field(default=4, ge=1)
    engine_dir: str = "models/engines"
    allow_build: bool = True
    """Build a missing engine on first use. Disable for immutable deployments."""
    builder_optimization_level: int = Field(default=3, ge=0, le=5)
    timing_cache: bool = True
    strict_version_check: bool = True
    """Refuse an engine built by a different TensorRT/CUDA version."""

    @model_validator(mode="after")
    def _check_batches(self) -> TensorRTConfig:
        if not (self.min_batch_size <= self.optimal_batch_size <= self.max_batch_size):
            raise ValueError(
                "tensorrt batch sizes must satisfy "
                f"min ({self.min_batch_size}) <= optimal ({self.optimal_batch_size}) "
                f"<= max ({self.max_batch_size})"
            )
        if self.precision is TensorRTPrecision.INT8 and not self.int8_calibration_dir:
            raise ValueError(
                "tensorrt.precision='int8' requires tensorrt.int8_calibration_dir "
                "pointing at representative calibration images"
            )
        return self


class BackendConfig(StrictModel):
    """Per-model inference backend selection.

    ``auto`` resolves from the detected platform: TensorRT on a Jetson that has
    it, otherwise the existing CUDA/ONNX/PyTorch/CoreML/CPU behaviour. Setting a
    backend explicitly always wins, and an unavailable one degrades with a
    warning rather than crashing.
    """

    detector: InferenceBackend = InferenceBackend.AUTO
    face_detector: InferenceBackend = InferenceBackend.AUTO
    face_encoder: InferenceBackend = InferenceBackend.AUTO
    body_encoder: InferenceBackend = InferenceBackend.AUTO
    tensorrt: TensorRTConfig = Field(default_factory=TensorRTConfig)


class PipelineConfig(StrictModel):
    """Asynchronous pipeline topology.

    Off by default so existing deployments keep byte-identical behaviour;
    ``async: true`` moves capture, inference and output onto separate threads
    with bounded queues.
    """

    async_enabled: bool = Field(default=False, alias="async")
    capture_queue_size: int = Field(default=2, ge=1, le=256)
    detection_queue_size: int = Field(default=2, ge=1, le=256)
    recognition_queue_size: int = Field(default=2, ge=1, le=256)
    output_queue_size: int = Field(default=4, ge=1, le=1024)
    drop_stale_frames: bool = True
    """Live sources prefer the newest frame: an overloaded pipeline must not
    accumulate seconds of latency behind a full queue."""
    overload_policy: OverloadPolicy = OverloadPolicy.DROP_VISUALIZATION
    output_workers: int = Field(default=1, ge=1, le=8)
    shutdown_timeout_s: float = Field(default=10.0, gt=0.0)

    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              populate_by_name=True)


class RecognitionSchedulerConfig(StrictModel):
    """When to spend a face-recognition pass on a track.

    Replaces a flat "every N frames for everyone" rule with a decision that
    depends on what the track's identity state actually is. The expensive case
    is an unidentified track: re-running recognition on it every single frame is
    what the backoff exists to bound.
    """

    enabled: bool = True
    stable_interval: int = Field(default=6, ge=1)
    """Frames between passes for a confidently recognised, stable track."""
    unknown_interval: int = Field(default=3, ge=1)
    low_confidence_interval: int = Field(default=3, ge=1)
    no_face_interval: int = Field(default=2, ge=1)
    """How often to re-check whether a face has become visible again. The face
    *encoder* never runs here -- only the (much cheaper) face detector."""
    force_on_new_track: bool = True
    force_on_recovery: bool = True
    force_when_identity_changes: bool = True

    unknown_backoff_enabled: bool = True
    unknown_backoff_factor: float = Field(default=1.7, ge=1.0, le=10.0)
    unknown_backoff_max_interval: int = Field(default=30, ge=1)
    """A genuinely unregistered person stays unregistered; retrying every third
    frame forever is pure waste. The interval grows geometrically and resets the
    moment anything changes (new face, quality change, gallery update)."""

    max_recognitions_per_frame: int = Field(default=0, ge=0)
    """0 = unlimited. A hard cap bounds worst-case frame latency in crowds."""

    @model_validator(mode="after")
    def _check_backoff(self) -> RecognitionSchedulerConfig:
        if self.unknown_backoff_max_interval < self.unknown_interval:
            raise ValueError(
                "recognition_scheduler.unknown_backoff_max_interval "
                f"({self.unknown_backoff_max_interval}) must be >= unknown_interval "
                f"({self.unknown_interval})"
            )
        return self


class RecognitionBatchConfig(StrictModel):
    """Batched face embedding."""

    enabled: bool = True
    max_size: int = Field(default=8, ge=1, le=128)
    """Upper bound on the batch the encoder is asked for.

    This expresses intent, not capability: a model exported with a fixed batch
    dimension (the shipped ArcFace w600k_r50.onnx is one) cannot be batched,
    and the pipeline detects that at load time and falls back to single-image
    calls rather than forcing a slower path. To benefit, re-export the encoder
    with a dynamic batch axis. 8 is a starting point for an 8 GB Orin Nano, not
    a measured optimum -- benchmark with `main.py benchmark-suite matrix`."""


class RuntimeQualityConfig(StrictModel):
    """Face quality gate applied at runtime, before the encoder runs.

    A face that is too small, too blurry or too badly lit produces an embedding
    that is not merely weak but actively misleading, because a poor embedding
    tends to sit near the middle of the gallery rather than far from all of it.
    Rejecting it yields the existing NO_FACE / FACE_TOO_SMALL semantics; it
    never yields an identity.

    Each threshold defaults to ``None``, meaning "inherit the existing
    ``face.*`` value" so enabling this section changes no behaviour on its own.
    """

    enabled: bool = True
    min_face_size: int | None = None
    min_eye_distance: float | None = None
    min_detector_confidence: float | None = None
    min_sharpness: float | None = None
    min_brightness: float | None = None
    max_brightness: float | None = None
    max_landmark_skew: float | None = None
    """Maximum |left-eye y - right-eye y| / inter-ocular distance. High values
    mean extreme roll or a bad landmark fit."""


class AdaptiveSearchConfig(StrictModel):
    """Switch face-search strategy based on how crowded the frame is.

    Per-person ROI search costs one detector call per person but each call is
    small; whole-frame search is a single larger call. Which wins depends on
    person count and frame size, so this is measured, not assumed -- it is off
    by default.
    """

    enabled: bool = False
    crowd_threshold: int = Field(default=4, ge=1)
    """At or above this many recognition candidates, use whole-frame search."""
    crowded_region: FaceSearchRegion = FaceSearchRegion.FRAME
    sparse_region: FaceSearchRegion = FaceSearchRegion.UPPER_BODY


class BenchmarkTargetConfig(StrictModel):
    """Per-profile performance targets for regression checking.

    Deliberately not global: a target that makes sense for an Orin Nano is
    meaningless on a laptop, so leaving these unset simply reports the measured
    numbers without a verdict.
    """

    enabled: bool = False
    minimum_fps: float | None = Field(default=None, gt=0.0)
    maximum_p95_latency_ms: float | None = Field(default=None, gt=0.0)
    maximum_p99_latency_ms: float | None = Field(default=None, gt=0.0)
    maximum_drop_rate_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    warn_margin_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    """How far inside a limit still counts as WARN rather than PASS."""


class BenchmarkConfig(StrictModel):
    warmup_frames: int = Field(default=10, ge=0)
    measure_frames: int = Field(default=200, ge=1)
    collect_system_telemetry: bool = True
    targets: BenchmarkTargetConfig = Field(default_factory=BenchmarkTargetConfig)


class ApplicationConfig(StrictModel):
    name: str = "YOLO26 Person ReID"
    version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str | None = None
    json_logs: bool = False


class ModelsConfig(StrictModel):
    """Model *locations*. Sizes are never hard-coded anywhere in the code."""

    detector: str = "yolo26n.pt"
    reid: str = "yolo26n-cls.pt"


class DeviceConfig(StrictModel):
    device: DeviceKind = DeviceKind.AUTO
    fp16: bool = True
    cuda_index: int = 0


class DetectorConfig(StrictModel):
    imgsz: int = Field(default=640, ge=64, le=4096)
    confidence: Probability = 0.35
    iou: Probability = 0.50
    person_class_id: int = 0
    classes: list[str] = Field(default_factory=lambda: ["person"])
    max_detections: int = Field(default=50, ge=1, le=1000)
    agnostic_nms: bool = False
    half: bool | None = None  # None -> inherit device.fp16

    @field_validator("classes")
    @classmethod
    def _only_person_supported(cls, value: list[str]) -> list[str]:
        if value != ["person"]:
            raise ValueError(
                "detector.classes currently supports exactly ['person']; "
                "person ReID is undefined for other classes."
            )
        return value


class ReIDConfig(StrictModel):
    backend: ReIDBackend = ReIDBackend.AUTO
    input_size: int | tuple[int, int] = 256
    """Square size, or an explicit (height, width) pair such as [256, 128]."""
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    normalize: bool = True
    batch_size: int = Field(default=16, ge=1, le=256)
    crop_padding: float = Field(default=0.0, ge=0.0, le=0.5)
    """Fractional padding added around a person box before cropping."""
    min_crop_size: int = Field(default=16, ge=1)
    embed_layer: int | None = None
    """Ultralytics backend: layer index to take embeddings from (None = auto)."""

    @field_validator("input_size")
    @classmethod
    def _validate_input_size(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("reid.input_size must be an int or a [height, width] pair")
            h, w = int(value[0]), int(value[1])
            if h < 16 or w < 16:
                raise ValueError("reid.input_size values must be >= 16")
            return (h, w)
        if int(value) < 16:
            raise ValueError("reid.input_size must be >= 16")
        return int(value)

    @property
    def size_hw(self) -> tuple[int, int]:
        if isinstance(self.input_size, tuple):
            return self.input_size
        return (int(self.input_size), int(self.input_size))


class MatchingConfig(StrictModel):
    metric: SimilarityMetric = SimilarityMetric.COSINE
    recognition_threshold: float = Field(default=0.70, ge=-1.0, le=1.0)
    """Acceptance threshold. The default was *measured* with
    ``python main.py evaluate`` on the bundled demo set using
    yolo26n-reid.onnx; it is a starting point for that encoder, not a
    universal constant. Re-measure it for your cameras and population."""
    high_confidence_threshold: float = Field(default=0.80, ge=-1.0, le=1.0)
    ambiguity_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    """If best - runner_up < margin the match is REJECTED instead of recognized."""
    unknown_label: str = "Unknown"
    no_face_label: str = "No face"
    """Shown in face mode when a person's face is not visible or is too small."""
    top_k: int = Field(default=3, ge=1, le=50)

    @model_validator(mode="after")
    def _check_threshold_order(self) -> MatchingConfig:
        if self.high_confidence_threshold < self.recognition_threshold:
            raise ValueError(
                "matching.high_confidence_threshold "
                f"({self.high_confidence_threshold}) must be >= "
                f"matching.recognition_threshold ({self.recognition_threshold})"
            )
        return self


class IdentityStabilityConfig(StrictModel):
    """Temporal smoothing of the identity decision for a track."""

    enabled: bool = True
    strategy: StabilizationStrategy = StabilizationStrategy.WEIGHTED_VOTE
    history_size: int = Field(default=10, ge=1, le=500)
    minimum_recognized_frames: int = Field(default=3, ge=1)
    switch_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    """A new identity must beat the incumbent's score by this much to take over."""
    decay: float = Field(default=0.9, gt=0.0, le=1.0)
    """Per-step weight decay for older observations (weighted_vote / ema)."""
    identity_persistence_frames: int = Field(default=15, ge=0)
    """How many consecutive unknown frames before a track drops its identity."""

    @model_validator(mode="after")
    def _check_history(self) -> IdentityStabilityConfig:
        if self.minimum_recognized_frames > self.history_size:
            raise ValueError(
                "identity_stability.minimum_recognized_frames "
                f"({self.minimum_recognized_frames}) cannot exceed history_size "
                f"({self.history_size})"
            )
        return self


class TrackingConfig(StrictModel):
    enabled: bool = True
    tracker: str = "botsort.yaml"
    persist: bool = True
    track_buffer: int = Field(default=30, ge=1)
    """Frames a lost track is kept alive by the tracker (occlusion tolerance)."""
    with_reid: bool = True
    """Enable appearance cues inside the tracker (BoT-SORT ReID)."""
    reid_model: str = "auto"
    """Tracker appearance model: 'auto' (detector features), 'gallery' (reuse the
    application ReID model) or an explicit path."""
    overrides: dict[str, Any] = Field(default_factory=dict)
    """Advanced passthrough merged into the generated tracker YAML."""
    lost_track_timeout: int = Field(default=30, ge=1)
    """Frames after which our own TrackManager forgets a vanished track."""
    identity_stability: IdentityStabilityConfig = Field(default_factory=IdentityStabilityConfig)

    @field_validator("tracker")
    @classmethod
    def _known_tracker(cls, value: str) -> str:
        allowed = {"botsort.yaml", "bytetrack.yaml"}
        if not value.endswith(".yaml"):
            raise ValueError("tracking.tracker must be a .yaml tracker configuration")
        if Path(value).name not in allowed and not Path(value).exists():
            raise ValueError(
                f"tracking.tracker '{value}' is neither a built-in tracker "
                f"({sorted(allowed)}) nor an existing file"
            )
        return value


class QualityConfig(StrictModel):
    """Reference-image quality gates used during enrollment (warn, not reject)."""

    enabled: bool = True
    min_bbox_height: int = Field(default=64, ge=1)
    min_bbox_width: int = Field(default=32, ge=1)
    min_crop_area_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    """Person area / image area below which the reference is considered small."""
    min_blur_variance: float = Field(default=25.0, ge=0.0)
    """Variance of Laplacian; lower means blurrier."""
    min_brightness: float = Field(default=25.0, ge=0.0, le=255.0)
    max_brightness: float = Field(default=235.0, ge=0.0, le=255.0)
    min_aspect_ratio: float = Field(default=1.2, ge=0.0)
    """Height/width of the person box; a full body is typically > 1.5."""
    truncation_border_tolerance: int = Field(default=2, ge=0)

    # Face-mode gates (used when recognition.mode is 'face').
    min_face_size: int = Field(default=60, ge=8)
    """Recommended face size in a *reference* photo -- stricter than the runtime
    minimum, because a weak reference degrades every future comparison."""
    min_eye_distance: float = Field(default=22.0, ge=0.0)
    min_face_score: Probability = 0.80

    fail_on_warnings: bool = False
    """When true, quality warnings become enrollment errors."""


class EnrollmentConfig(StrictModel):
    require_single_person: bool = True
    selection_strategy: SelectionStrategy = SelectionStrategy.LARGEST_PERSON
    regenerate_missing_embeddings: bool = True
    detector_confidence: Probability | None = None
    """Optional lower detector threshold used only for enrollment images."""
    quality: QualityConfig = Field(default_factory=QualityConfig)
    save_normalized_crop: bool = True
    """Store a copy of the crop used for enrollment (originals are never touched)."""
    crop_dir: str = "data/gallery/crops"


class GalleryConfig(StrictModel):
    store: EmbeddingStoreKind = EmbeddingStoreKind.LOCAL_NUMPY
    directory: str = "data/gallery"
    embeddings_subdir: str = "embeddings"
    metadata_subdir: str = "metadata"
    auto_build: bool = True
    """Build missing embeddings automatically when a pipeline starts."""
    revalidate_on_load: bool = True
    """Re-enroll when the reference image changed or the model changed."""


class RecognitionSectionConfig(StrictModel):
    mode: RecognitionMode = RecognitionMode.FACE
    """``face`` (default) is clothing-invariant; ``person_reid`` is not."""
    batch: RecognitionBatchConfig = Field(default_factory=RecognitionBatchConfig)


class FaceConfig(StrictModel):
    """Face detection, alignment and recognition settings."""

    backend: FaceBackend = FaceBackend.AUTO
    detector_model: str = "models/face_detection_yunet_2023mar.onnx"
    recognition_model: str = "models/w600k_r50.onnx"

    detection_confidence: Probability = 0.6
    nms_threshold: Probability = 0.3
    top_k: int = Field(default=500, ge=1)

    chip_size: int = Field(default=112, ge=32, le=512)
    """Aligned face chip size; 112 for every ArcFace-family encoder."""
    align: bool = True
    require_landmarks: bool = True
    """Refuse unaligned faces rather than embedding a materially worse chip."""

    min_face_size: int = Field(default=40, ge=8)
    """Minimum face box short side. Below this there is not enough detail to
    identify someone reliably, and guessing produces false accepts."""
    min_eye_distance: float = Field(default=14.0, ge=0.0)
    """Minimum inter-ocular distance -- catches strong profile views."""
    min_blur_variance: float = Field(default=0.0, ge=0.0)
    """Reject faces blurrier than this (0 disables the check)."""

    search_region: FaceSearchRegion = FaceSearchRegion.UPPER_BODY
    head_fraction: float = Field(default=0.55, gt=0.0, le=1.0)
    search_margin: float = Field(default=0.12, ge=0.0, le=1.0)

    identity_hold_frames: int = Field(default=45, ge=0)
    """How long a track keeps a face-derived identity while no face is visible
    (someone turning around, or walking away from the camera). Frames without a
    face are neutral evidence: they neither confirm nor refute the identity."""

    arcface_input_mean: float = 127.5
    arcface_input_scale: float = 127.5

    runtime_quality: RuntimeQualityConfig = Field(default_factory=RuntimeQualityConfig)
    adaptive_search: AdaptiveSearchConfig = Field(default_factory=AdaptiveSearchConfig)

    @model_validator(mode="after")
    def _check_chip(self) -> FaceConfig:
        if self.require_landmarks and not self.align:
            raise ValueError(
                "face.require_landmarks=true implies face.align=true; "
                "landmarks exist to drive alignment"
            )
        return self


class OutputConfig(StrictModel):
    directory: str = "data/output"
    save_snapshots: bool = True
    snapshot_mode: SnapshotMode = SnapshotMode.RECOGNIZED
    snapshot_dir: str = "snapshots"
    snapshot_cooldown_seconds: float = Field(default=5.0, ge=0.0)
    """Per identity/track rate limit so a stream does not flood the disk."""
    save_crops: bool = False
    crops_dir: str = "crops"
    save_metadata: bool = True
    metadata_dir: str = "metadata"
    save_video: bool = True
    videos_dir: str = "videos"
    annotated_image_dir: str = "images"
    codec: str = "mp4v"
    video_fps: float | None = None
    """Override the output FPS; None inherits the source FPS."""
    overwrite: bool = False
    """Never silently overwrite: when false a numeric suffix is appended."""
    jpeg_quality: int = Field(default=92, ge=1, le=100)


class RecordingConfig(StrictModel):
    mode: RecordingMode = RecordingMode.DISABLED
    save_when_identity_detected: bool = True
    save_unknown: bool = False
    pre_event_seconds: float = Field(default=3.0, ge=0.0, le=60.0)
    post_event_seconds: float = Field(default=5.0, ge=0.0, le=300.0)
    max_clip_seconds: float = Field(default=120.0, gt=0.0)
    filename_prefix: str = "event"

    backend: RecordingBackend = RecordingBackend.AUTO
    """``auto`` uses the NVIDIA hardware encoder when GStreamer exposes it,
    otherwise OpenCV's software writer. Falling back is always allowed."""
    codec: VideoCodec = VideoCodec.H264
    """Used by the GStreamer backend. The OpenCV backend keeps output.codec."""
    hardware_acceleration: bool = True
    bitrate_kbps: int = Field(default=4000, ge=100, le=100_000)
    async_write: bool = True
    """Encode on the output worker rather than the inference thread."""


class PerformanceConfig(StrictModel):
    reid_interval: int = Field(default=1, ge=1)
    """Run ReID for a stable track every N frames (1 = every frame)."""
    embedding_cache: bool = True
    """Reuse the last embedding of a track between ReID runs."""
    batch_size: int = Field(default=16, ge=1, le=256)
    """Deprecated: superseded by recognition.batch.max_size, which is the value
    the encoder actually uses. Kept so existing configurations still load."""
    max_detections: int = Field(default=50, ge=1, le=1000)
    force_reid_on_new_track: bool = True
    force_reid_when_unknown: bool = True
    """Unknown tracks are re-checked every frame so identities are not missed."""
    frame_stride: int = Field(default=1, ge=1)
    """Process every Nth frame of a video source."""


class DisplayConfig(StrictModel):
    show_window: bool = True
    window_name: str = "YOLO26 Person ReID"
    show_metrics: bool = True
    show_track_id: bool = True
    show_similarity: bool = True
    show_title: bool = True
    show_face_box: bool = True
    """Face mode: outline the face the identity decision was based on."""
    box_thickness: int = Field(default=2, ge=1, le=10)
    font_scale: float = Field(default=0.5, gt=0.0)
    colors: dict[str, tuple[int, int, int]] = Field(
        default_factory=lambda: {
            "recognized": (60, 200, 60),
            "low_confidence": (40, 190, 235),
            "unknown": (70, 70, 230),
            "rejected": (180, 120, 255),
            "pending": (170, 170, 170),
            "no_face": (150, 150, 150),
        }
    )


class EventsConfig(StrictModel):
    enabled: bool = True
    log_to_file: bool = True
    directory: str = "events"
    filename: str = "events.jsonl"
    console: bool = False


class RetentionConfig(StrictModel):
    enabled: bool = False
    snapshots_days: int = Field(default=30, ge=0)
    videos_days: int = Field(default=30, ge=0)
    events_days: int = Field(default=90, ge=0)
    crops_days: int = Field(default=30, ge=0)
    metadata_days: int = Field(default=90, ge=0)


class DebugConfig(StrictModel):
    enabled: bool = False
    directory: str = "debug"
    save_frames: bool = False
    save_crops: bool = True
    save_matches: bool = True
    save_track_state: bool = True
    max_frames: int = Field(default=200, ge=1)


class SourceConfig(StrictModel):
    type: SourceKind = SourceKind.WEBCAM
    device: int | str = 0
    path: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    buffer_size: int = Field(default=1, ge=1)
    loop: bool = False
    recursive: bool = False
    extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    )

    # --- jetson_camera ------------------------------------------------------
    sensor_id: int = Field(default=0, ge=0)
    """CSI sensor index for nvarguscamerasrc."""
    csi: bool | Literal["auto"] = "auto"
    """``auto`` uses the CSI pipeline when nvarguscamerasrc exists, else V4L2."""
    flip_method: int = Field(default=0, ge=0, le=7)
    capture_width: int | None = None
    capture_height: int | None = None
    """Sensor capture geometry, when it differs from the delivered frame size.
    Downscaling happens once inside GStreamer rather than repeatedly in Python."""
    gst_pipeline: str | None = None
    """Full manual GStreamer pipeline; overrides everything else when set."""
    latency_ms: int = Field(default=200, ge=0)
    """RTSP jitter buffer depth."""

    @field_validator("extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for ext in value:
            ext = ext.strip().lower()
            if not ext:
                continue
            out.append(ext if ext.startswith(".") else f".{ext}")
        if not out:
            raise ValueError("source.extensions must not be empty")
        return out


class FaceIdentityConfig(StrictModel):
    """The face-based identity engine.

    This is the identity pipeline for passport-photo enrollment. The body-ReID
    encoder is never the identity mechanism here: measured on this project's
    own evaluation set, passport-enrolled body appearance produces genuine and
    impostor score distributions that *overlap*, so no threshold can both
    accept a registered person and reject an unknown one. Open-set recognition
    is not possible from it. See docs/identity.md.
    """

    enabled: bool = True
    gallery_dir: str = "data/people"
    """One directory per person: reference photo, embeddings, metadata."""

    face_detector_model: str = "models/scrfd_10g.onnx"
    face_detector_backend: Literal["scrfd", "yunet"] = "scrfd"
    face_detector_confidence: Probability = 0.5
    face_detector_nms: Probability = 0.4
    face_detector_input: int = Field(default=640, ge=128, le=2048)

    face_encoder_model: str = "models/w600k_r50.onnx"
    chip_size: int = Field(default=112, ge=64, le=256)

    calibration_file: str = "data/people/calibration.json"
    """Fitted score-to-probability mapping. Absent means no calibrated
    confidence is reported -- never an invented one."""

    # Thresholds default to None, meaning "use the calibrated value". Setting
    # one pins it, which is only appropriate when it came from measurement.
    threshold_full: float | None = Field(default=None, ge=-1.0, le=1.0)
    threshold_partial: float | None = Field(default=None, ge=-1.0, le=1.0)
    threshold_masked: float | None = Field(default=None, ge=-1.0, le=1.0)
    fallback_threshold: float = Field(default=0.40, ge=-1.0, le=1.0)
    """Used only when nothing has been calibrated. Deliberately conservative."""
    ambiguity_margin: float = Field(default=0.06, ge=0.0, le=1.0)
    min_face_quality: float = Field(default=0.30, ge=0.0, le=1.0)
    min_identity_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    max_live_embeddings: int = Field(default=20, ge=1, le=200)

    recognize_orphan_faces: bool = True
    """Identify a face that lies inside no detected person box.

    The architecture runs person detection first, and normally every face
    belongs to a person the detector found. But a person who is seated behind
    a desk, or visible only through a doorway, is often not detected as a
    person at all while their face is perfectly clear -- measured on real
    footage, YOLO26 returned a box around such a person's legs, four hundred
    pixels below their head. Dropping those faces discards identity evidence
    for no reason: a visible face is a person.

    Such a detection reports a *derived* person region and a detector
    confidence of zero, so nothing downstream mistakes it for a person the
    detector actually found. It never changes how identity is decided.
    """

    orphan_scan_interval: int = Field(default=3, ge=1)
    """How often to look for orphan faces when no tracked person is due.

    Face detection is the expensive stage, and normally it runs only when the
    scheduler wants a recognition pass. Scanning for orphans on every frame
    regardless roughly halves throughput. The people this recovers are the
    ones the person detector missed *because* they are sitting still, so a
    few frames between scans costs little: measured on real footage, 1 gives
    90.4% and 11.4 FPS, 3 gives 89.6% and 16.3 FPS. Between scans an orphan
    keeps the conclusion of the last one rather than flickering out.
    """

    @model_validator(mode="after")
    def _warn_on_pinned_thresholds(self) -> FaceIdentityConfig:
        pinned = [
            name for name in ("threshold_full", "threshold_partial", "threshold_masked")
            if getattr(self, name) is not None
        ]
        if pinned and self.calibration_file:
            # Not an error: an operator may legitimately pin a measured value.
            pass
        return self


class TemporalIdentityConfig(StrictModel):
    """Temporal evidence accumulation and identity stability."""

    enabled: bool = True
    history_size: int = Field(default=15, ge=1, le=200)
    min_confirmation_frames: int = Field(default=4, ge=2)
    """Never 1: a single frame must never name anybody."""
    switch_margin: float = Field(default=0.10, ge=0.0, le=1.0)
    lost_track_timeout: int = Field(default=30, ge=1)
    occlusion_hold_frames: int = Field(default=45, ge=0)
    """How long a *confirmed* identity survives without a usable face. A track
    that never earned an identity can never acquire one this way."""
    unknown_frames_to_release: int = Field(default=12, ge=1)
    min_evidence_weight: float = Field(default=1.0, ge=0.0)
    fast_confirmation_frames: int = Field(default=2, ge=2)
    """Floor on observations before naming, when the evidence is strong.
    Never below 2: one frame must never name anybody."""
    strong_evidence_weight: float = Field(default=1.6, ge=0.0)
    """Accumulated quality-weighted evidence that confirms at
    ``fast_confirmation_frames`` instead of ``min_confirmation_frames``."""


class OnlineAdaptationConfig(StrictModel):
    """Conservative gallery adaptation. Off unless deliberately enabled."""

    enabled: bool = False
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    similarity_headroom: float = Field(default=0.15, ge=0.0, le=1.0)
    min_quality: float = Field(default=0.70, ge=0.0, le=1.0)
    min_confirmation_frames: int = Field(default=5, ge=1)
    min_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    min_track_stability: float = Field(default=0.60, ge=0.0, le=1.0)
    require_full_face: bool = True
    max_samples_per_identity: int = Field(default=20, ge=1, le=200)
    min_novelty: float = Field(default=0.02, ge=0.0, le=1.0)
    max_novelty: float = Field(default=0.45, ge=0.0, le=1.0)
    cooldown_seconds: float = Field(default=20.0, ge=0.0)
    max_per_track: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _check_novelty_band(self) -> OnlineAdaptationConfig:
        if self.min_novelty >= self.max_novelty:
            raise ValueError(
                "online_adaptation.min_novelty must be below max_novelty; the "
                "band accepts observations that are new but not implausible"
            )
        return self


class ReferenceQualityGateConfig(StrictModel):
    """Gates applied to a passport photograph at enrollment."""

    enabled: bool = True
    min_interocular_px: float = Field(default=28.0, ge=0.0)
    min_face_px: int = Field(default=70, ge=8)
    min_sharpness: float = Field(default=0.30, ge=0.0, le=1.0)
    min_exposure: float = Field(default=0.35, ge=0.0, le=1.0)
    max_yaw_deg: float = Field(default=25.0, ge=0.0, le=90.0)
    max_pitch_deg: float = Field(default=20.0, ge=0.0, le=90.0)
    min_landmark_score: float = Field(default=0.55, ge=0.0, le=1.0)
    min_overall_quality: float = Field(default=0.45, ge=0.0, le=1.0)
    require_full_face: bool = True
    fail_on_warnings: bool = False
    max_faces: int = Field(default=1, ge=1, le=20)
    selection: Literal["largest_face", "highest_confidence", "center_most"] = "largest_face"


class PersonConfig(StrictModel):
    """One enrolled identity. Person data lives in YAML, never in source code."""

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    title: str = ""
    image_path: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    """Reserved for future multi-image enrollment; one image is enough today."""
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("people[].id must not be empty")
        if any(c in value for c in '/\\:*?"<>| '):
            raise ValueError(
                f"people[].id '{value}' contains characters that are unsafe in "
                "filenames; use letters, digits, '-' or '_'"
            )
        return value

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("people[].name is required and must not be blank")
        return value

    @field_validator("title", mode="before")
    @classmethod
    def _blank_title(cls, value: Any) -> str:
        """Accept ``title:`` written with nothing after it.

        YAML reads a bare key as None, which is the most natural way to write
        "this person has no title". Rejecting it stops the whole configuration
        from loading over an optional field.
        """
        return "" if value is None else value

    @model_validator(mode="after")
    def _at_least_one_image(self) -> PersonConfig:
        if not self.image_path and not self.image_paths:
            raise ValueError(
                f"person '{self.id}' has no reference image: set image_path "
                "(one-shot enrollment) or image_paths"
            )
        return self

    @property
    def all_image_paths(self) -> list[str]:
        paths: list[str] = []
        if self.image_path:
            paths.append(self.image_path)
        paths.extend(p for p in self.image_paths if p not in paths)
        return paths


class ApiConfig(StrictModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    allow_upload: bool = True
    max_upload_mb: int = Field(default=25, ge=1)


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #


# Per-mode threshold defaults, each measured with `python main.py evaluate`
# rather than guessed. They are starting points for the shipped models, not
# universal constants -- re-measure them for your cameras and population.
_MODE_THRESHOLDS: dict[RecognitionMode, tuple[float, float]] = {
    RecognitionMode.FACE: (0.45, 0.60),
    RecognitionMode.PERSON_REID: (0.70, 0.80),
}


class AppConfig(StrictModel):
    """Root configuration object handed to every component."""

    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    recognition: RecognitionSectionConfig = Field(default_factory=RecognitionSectionConfig)
    recognition_scheduler: RecognitionSchedulerConfig = Field(
        default_factory=RecognitionSchedulerConfig
    )
    face: FaceConfig = Field(default_factory=FaceConfig)
    face_identity: FaceIdentityConfig = Field(default_factory=FaceIdentityConfig)
    temporal: TemporalIdentityConfig = Field(default_factory=TemporalIdentityConfig)
    online_adaptation: OnlineAdaptationConfig = Field(default_factory=OnlineAdaptationConfig)
    reference_quality: ReferenceQualityGateConfig = Field(
        default_factory=ReferenceQualityGateConfig
    )
    backend: BackendConfig = Field(default_factory=BackendConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    enrollment: EnrollmentConfig = Field(default_factory=EnrollmentConfig)
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    people: list[PersonConfig] = Field(default_factory=list)

    # Populated by the loader; never read from the YAML file itself.
    config_path: str | None = Field(default=None, exclude=True)
    base_dir: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _unique_person_ids(self) -> AppConfig:
        seen: dict[str, int] = {}
        duplicates: list[str] = []
        for index, person in enumerate(self.people):
            if person.id in seen:
                duplicates.append(
                    f"'{person.id}' (entries #{seen[person.id] + 1} and #{index + 1})"
                )
            else:
                seen[person.id] = index
        if duplicates:
            raise ValueError(
                "duplicate people[].id values: " + ", ".join(duplicates) +
                " -- every identity needs a unique id"
            )
        return self

    @model_validator(mode="after")
    def _reid_interval_requires_tracking(self) -> AppConfig:
        if self.performance.reid_interval > 1 and not self.tracking.enabled:
            raise ValueError(
                "performance.reid_interval > 1 requires tracking.enabled=true, "
                "because embeddings are cached per track id"
            )
        return self

    @model_validator(mode="after")
    def _apply_mode_threshold_defaults(self) -> AppConfig:
        """Fill unset thresholds with values appropriate to the recognition mode.

        Face and whole-body embeddings live in different similarity regimes: a
        cosine of 0.45 is a confident face match but near-noise for body
        appearance. Carrying one mode's threshold into the other is the easiest
        way to get silently wrong results, so an explicitly configured value is
        always respected and only an *unset* one is filled in here.
        """
        explicit = self.matching.model_fields_set
        low, high = _MODE_THRESHOLDS[self.recognition.mode]
        if "recognition_threshold" not in explicit:
            self.matching.recognition_threshold = low
        if "high_confidence_threshold" not in explicit:
            self.matching.high_confidence_threshold = high
        if self.matching.high_confidence_threshold < self.matching.recognition_threshold:
            raise ValueError(
                "matching.high_confidence_threshold "
                f"({self.matching.high_confidence_threshold}) must be >= "
                f"matching.recognition_threshold ({self.matching.recognition_threshold})"
            )
        return self

    @model_validator(mode="after")
    def _matching_metric_matches_reid(self) -> AppConfig:
        if self.matching.metric is not self.reid.similarity_metric:
            raise ValueError(
                f"matching.metric ('{self.matching.metric.value}') and "
                f"reid.similarity_metric ('{self.reid.similarity_metric.value}') "
                "must agree"
            )
        return self

    @property
    def enabled_people(self) -> list[PersonConfig]:
        return [p for p in self.people if p.enabled]
