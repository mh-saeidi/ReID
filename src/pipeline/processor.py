"""The ReID processing pipeline.

One frame in, one :class:`FrameResult` out:

    frame -> YOLO26 detection (+ tracking)
          -> person crops
          -> ReID embeddings
          -> gallery similarity search
          -> open-set identity decision
          -> temporal stabilization per track

The processor is headless and side-effect free apart from its own track state:
rendering, snapshots, recording and events are the runner's job. That is what
makes the same object reusable from the CLI, a test and a REST endpoint.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config.schema import AppConfig, RecognitionMode
from src.core.types import (
    BBox,
    Detection,
    DetectionResult,
    Frame,
    FrameResult,
    RecognitionResult,
    RecognitionStatus,
)
from src.detection.detector import Detector
from src.face.types import FaceDetection
from src.identity.gallery import IdentityGallery
from src.identity.matcher import IdentityMatcher
from src.pipeline.metrics import MetricsCollector, Stopwatch
from src.reid.encoder import ReIDEncoder, l2_normalize
from src.reid.preprocess import CropPreprocessor, CropResult, CropStatus
from src.tracking.scheduler import RecognitionScheduler
from src.tracking.track_manager import TrackManager, TrackTransition
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Crop failures that mean "there is nothing to recognise here", as opposed to
# "this detection was skipped this frame".
_NO_FACE_STATUSES = frozenset(
    {CropStatus.NO_FACE, CropStatus.FACE_TOO_SMALL, CropStatus.LOW_QUALITY}
)


@dataclass(slots=True)
class ProcessOutcome:
    """A frame result plus the track lifecycle changes it caused."""

    result: FrameResult
    transitions: list[TrackTransition]
    new_tracks: list[int]
    ended_tracks: list[int]


class ReIDPipeline:
    """Detection + ReID + identity decision for a single stream of frames."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        encoder: ReIDEncoder,
        preprocessor: CropPreprocessor,
        gallery: IdentityGallery,
        matcher: IdentityMatcher,
        *,
        metrics: MetricsCollector | None = None,
        use_tracking: bool = True,
    ) -> None:
        self._config = config
        self._detector = detector
        self._encoder = encoder
        self._preprocessor = preprocessor
        self._gallery = gallery
        self._matcher = matcher
        self._metrics = metrics or MetricsCollector()
        self._use_tracking = use_tracking and config.tracking.enabled
        self._scheduler = RecognitionScheduler(
            config.recognition_scheduler, config.matching
        )
        batch = config.recognition.batch
        # The configuration expresses intent; the model decides what is
        # possible. A graph with a fixed batch dimension cannot be batched, and
        # forcing it is measurably slower than sequential calls.
        encoder_max = encoder.info.max_batch_size if encoder is not None else 0
        self._max_batch = max(1, batch.max_size)
        if encoder_max == 1:
            self._max_batch = 1
            self._batch_enabled = False
        else:
            if encoder_max > 1:
                self._max_batch = min(self._max_batch, encoder_max)
            self._batch_enabled = batch.enabled and self._max_batch > 1
        self._tracks = TrackManager(
            config.tracking,
            config.matching,
            # Only face mode can lose sight of the identity signal while still
            # tracking the person, so the hold budget is zero otherwise.
            absent_hold_frames=(
                config.face.identity_hold_frames
                if config.recognition.mode is RecognitionMode.FACE
                else 0
            ),
        )

    # ------------------------------------------------------------- properties
    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def track_manager(self) -> TrackManager:
        return self._tracks

    @property
    def scheduler(self) -> RecognitionScheduler:
        return self._scheduler

    @property
    def gallery(self) -> IdentityGallery:
        return self._gallery

    @property
    def uses_tracking(self) -> bool:
        return self._use_tracking

    def reset(self) -> None:
        """Clear tracker and track state (call between independent sources)."""
        self._tracks.reset()
        self._scheduler.reset()
        if self._use_tracking:
            self._detector.reset_tracker()

    # ------------------------------------------------------------- processing
    def process(self, frame: Frame) -> ProcessOutcome:
        """Run the full pipeline on one frame."""
        started = time.perf_counter()

        with Stopwatch(self._metrics, "detector") as detect_timer:
            detections = (
                self._detector.track(frame.image, persist=self._config.tracking.persist)
                if self._use_tracking
                else self._detector.detect(frame.image)
            )
        detections = detections[: self._config.performance.max_detections]

        states, new_tracks = self._bind_tracks(detections, frame)
        with Stopwatch(self._metrics, "reid") as reid_timer:
            embeddings, crop_reasons = self._embed(
                frame.image, detections, states, frame
            )

        with Stopwatch(self._metrics, "matching"):
            recognitions = self._match(embeddings, crop_reasons)

        results, transitions = self._finalize(frame, detections, states, recognitions)
        # Tracks not observed this frame may come back with the same ID attached
        # to a different person, so mark them for a forced re-check.
        for state in self._tracks.tracks.values():
            if state.last_seen_frame != frame.index:
                state.recognition_cache.mark_lost()
        ended = [state.track_id for state in self._tracks.expire(frame.index)]

        total = time.perf_counter() - started
        self._metrics.record_stage("total", total)
        result = FrameResult(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            source_id=frame.source_id,
            width=frame.width,
            height=frame.height,
            detections=results,
            timings={
                "detector": detect_timer.elapsed,
                "reid": reid_timer.elapsed,
                "total": total,
            },
        )
        faces = sum(1 for d in results if d.face is not None)
        no_face = sum(
            1 for d in results if d.recognition_status is RecognitionStatus.NO_FACE
        )
        self._metrics.record_frame(
            detections=len(results),
            recognized=len(result.recognized),
            tracks=self._tracks.active_count,
            no_face=no_face,
            faces=faces,
        )
        self._metrics.scheduler_stats = self._scheduler.stats.to_dict()
        search = getattr(self._preprocessor, "search_stats", None)
        if search:
            self._metrics.search_stats = dict(search)
        return ProcessOutcome(result, transitions, new_tracks, ended)

    # ------------------------------------------------------------------ steps
    def _bind_tracks(self, detections: Sequence[Detection], frame: Frame):
        """Attach a :class:`TrackState` to every detection.

        Detections without a tracker id (still images, or a tracker that has not
        confirmed a track yet) get a negative synthetic id so per-detection
        bookkeeping stays uniform. Synthetic ids never collide with real ones.
        """
        states = []
        new_tracks: list[int] = []
        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                track_id = self._tracks.synthetic_id()
                detection.track_id = track_id
            state, is_new = self._tracks.touch(track_id, frame.index, frame.timestamp)
            state.last_bbox = detection.bbox
            if is_new:
                new_tracks.append(track_id)
            states.append(state)
        return states, new_tracks

    def _embed(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
        states: Sequence,
        frame: Frame,
    ) -> tuple[list[np.ndarray | None], dict[int, CropResult]]:
        """Run face detection, alignment and embedding for the eligible tracks.

        Three things decide whether a detection is embedded, in order:

        1. the recognition scheduler, which knows the track's identity state and
           bounds the cost of a person who will never match;
        2. face detection + the quality gate, which refuse a face that carries
           too little information to identify safely;
        3. batching, which sends the surviving chips through the encoder in one
           call rather than one at a time.

        A detection rejected at step 2 gets an explicit reason, which becomes a
        NO_FACE result downstream. It never becomes an identity, and in face
        mode no cached embedding is substituted for it.
        """
        performance = self._config.performance
        requires_face = self._preprocessor.requires_visible_face
        embeddings: list[np.ndarray | None] = [None] * len(detections)
        crop_reasons: dict[int, CropResult] = {}

        self._scheduler.begin_frame(frame.index)
        begin = getattr(self._preprocessor, "begin_frame", None)
        if callable(begin):
            begin(len(detections))

        pending_crops: list[np.ndarray] = []
        pending_slots: list[int] = []

        with Stopwatch(self._metrics, "face_detection"):
            for index, (detection, state) in enumerate(
                zip(detections, states, strict=True)
            ):
                cache = state.recognition_cache
                cache.track_id = state.track_id

                if not self._should_recognize(state, frame.index, performance):
                    cache.note_skip()
                    if not requires_face:
                        embeddings[index] = state.last_embedding
                    elif cache.last_face_bbox is not None:
                        # Keep showing where the face was last seen so the
                        # rendering stays stable between scheduled passes. This
                        # is presentation only: no identity is asserted here.
                        detection.face = _cached_face(cache)
                    continue

                cache.note_recognition(frame.index, frame.timestamp)
                result = self._preprocessor.extract(image, detection.bbox)
                if not result.ok:
                    crop_reasons[index] = result
                    if result.face is not None:
                        detection.face = result.face
                        cache.note_quality_rejection(result.detail)
                    logger.debug(
                        "No usable region for this detection",
                        extra={
                            "track_id": detection.track_id,
                            "reason": result.status.value,
                            "detail": result.detail,
                        },
                    )
                    # Critically: do NOT substitute the track's last embedding.
                    # In face mode that would keep asserting an identity from a
                    # face that is no longer visible, unbounded and unauditable.
                    # Carrying an identity through a hidden face is the
                    # stabilizer's job, where it is explicit and frame-limited.
                    if not requires_face:
                        embeddings[index] = state.last_embedding
                    continue

                detection.face = result.face
                pending_crops.append(result.image)
                pending_slots.append(index)

        self._metrics.face_detect_calls += 1
        if not pending_crops:
            return embeddings, crop_reasons

        with Stopwatch(self._metrics, "encoder"):
            batch = self._encode(pending_crops)

        for slot, vector in zip(pending_slots, batch, strict=True):
            if not np.any(vector):
                continue
            normalized = l2_normalize(vector)
            embeddings[slot] = normalized
            state = states[slot]
            state.last_embedding = normalized
            state.last_reid_frame = state.last_seen_frame
            state.reid_runs += 1
            state.recognition_cache.note_embedding(frame.index, int(normalized.shape[-1]))
        return embeddings, crop_reasons

    def _should_recognize(self, state, frame_index: int, performance) -> bool:
        """Scheduler decision, with the legacy interval rule as the fallback."""
        if self._scheduler.enabled:
            return self._scheduler.decide(state, frame_index).should_recognize
        return self._tracks.needs_reid(state, state.last_seen_frame, performance)

    def _encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed the eligible chips, in batches when batching is enabled.

        Only faces that passed detection and the quality gate reach this point,
        so the batch never contains a missing or rejected face.

        Batching is only attempted when the loaded model actually supports it
        (see EncoderInfo.max_batch_size). Padding batches up to fixed bucket
        sizes was tried and measured; it produced no improvement and wasted
        work on the padding rows, so batches are sent at their natural length.
        """
        self._metrics.reid_calls += 1
        self._metrics.reid_crops += len(crops)
        if not self._batch_enabled or len(crops) <= 1:
            self._metrics.record_batch(len(crops))
            return self._encoder.embed_batch(crops)

        outputs: list[np.ndarray] = []
        for start in range(0, len(crops), self._max_batch):
            window = crops[start : start + self._max_batch]
            self._metrics.record_batch(len(window))
            outputs.append(self._encoder.embed_batch(window))
        return np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]

    def _match(
        self,
        embeddings: Sequence[np.ndarray | None],
        crop_reasons: dict[int, CropResult] | None = None,
    ) -> list[RecognitionResult]:
        """Compare every available embedding against the gallery at once."""
        view = self._gallery.view
        crop_reasons = crop_reasons or {}
        results: list[RecognitionResult] = []
        for index in range(len(embeddings)):
            reason = crop_reasons.get(index)
            if reason is not None and reason.status in _NO_FACE_STATUSES:
                # Face mode with no usable face: report that, do not guess and
                # do not fall back to body appearance.
                results.append(
                    RecognitionResult.no_face(
                        self._config.matching.no_face_label, detail=reason.detail
                    )
                )
            else:
                results.append(
                    RecognitionResult(
                        status=RecognitionStatus.PENDING,
                        identity_name=self._config.matching.unknown_label,
                    )
                )
        # Only slots still awaiting a decision are matched: a NO_FACE outcome is
        # already final and must not be overwritten by a gallery comparison.
        slots = [
            i
            for i, embedding in enumerate(embeddings)
            if embedding is not None and results[i].status is RecognitionStatus.PENDING
        ]
        if not slots:
            return results
        if view.is_empty:
            for slot in slots:
                results[slot] = RecognitionResult.unknown(self._config.matching.unknown_label)
            return results

        queries = np.stack([embeddings[i] for i in slots], axis=0)  # type: ignore[index]
        for slot, recognition in zip(
            slots, self._matcher.match_batch(queries, view), strict=True
        ):
            results[slot] = recognition
        return results

    def _finalize(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        states: Sequence,
        recognitions: Sequence[RecognitionResult],
    ) -> tuple[list[DetectionResult], list[TrackTransition]]:
        """Stabilize identities and build the output records."""
        results: list[DetectionResult] = []
        transitions: list[TrackTransition] = []

        for detection, state, recognition in zip(detections, states, recognitions, strict=True):
            stabilized: RecognitionResult | None = None

            if (
                recognition.status is RecognitionStatus.PENDING
                and self._use_tracking
                and state.last_result is not None
            ):
                # The scheduler chose not to spend a pass on this track this
                # frame. That is a decision about *cost*, not about identity, so
                # the track keeps showing the conclusion of its last pass rather
                # than reverting to "pending" and flickering.
                #
                # This is not the same as reusing a stale embedding: no new
                # comparison is made and no new claim is asserted. The result is
                # still bounded, because the scheduler forces a fresh pass at
                # `stable_interval`, and a pass that then fails decays the
                # identity through the stabilizer exactly as before.
                stabilized = state.last_result

            elif self._use_tracking and recognition.status is not RecognitionStatus.PENDING:
                stabilized, transition = self._tracks.apply_recognition(
                    state, recognition, frame.index
                )
                if transition is not None:
                    transitions.append(transition)
            elif recognition.status is not RecognitionStatus.PENDING:
                state.last_result = recognition
                if recognition.is_recognized:
                    state.frames_recognized += 1
                else:
                    state.frames_unknown += 1

            effective = stabilized or recognition
            if recognition.status is not RecognitionStatus.PENDING:
                state.recognition_cache.note_result(
                    recognition.status,
                    recognition.identity_id,
                    recognition.similarity,
                    face=detection.face,
                )

            results.append(
                DetectionResult(
                    detection_id=detection.detection_id,
                    bbox=detection.bbox,
                    detector_confidence=detection.confidence,
                    source_id=frame.source_id,
                    timestamp=frame.timestamp,
                    frame_index=frame.index,
                    track_id=detection.track_id,
                    face=detection.face,
                    recognition=recognition,
                    stabilized=stabilized,
                )
            )

            if logger.isEnabledFor(10):  # DEBUG
                logger.debug(
                    "Recognition",
                    extra={
                        "track_id": detection.track_id,
                        "identity": effective.identity_id or "unknown",
                        "similarity": round(effective.similarity, 4),
                        "status": effective.status.value,
                        "scheduler": state.recognition_cache.scheduler_state.value,
                        "det_conf": round(detection.confidence, 3),
                    },
                )
        return results, transitions

    # ----------------------------------------------------------------- images
    def process_image(self, image: np.ndarray, *, source_id: str = "image") -> FrameResult:
        """Convenience entry point for a standalone image (no tracking)."""
        frame = Frame(image=image, index=0, timestamp=time.time(), source_id=source_id)
        was_tracking = self._use_tracking
        self._use_tracking = False
        try:
            return self.process(frame).result
        finally:
            self._use_tracking = was_tracking

    def embed_crop(self, crop: np.ndarray) -> np.ndarray:
        """Embed an already-cropped person image (used by calibration tools)."""
        return l2_normalize(self._encoder.embed(crop))

    def embed_bbox(self, image: np.ndarray, bbox: BBox) -> np.ndarray | None:
        """Embed one person box, or ``None`` when there is nothing to embed.

        In face mode "nothing to embed" means no usable face, which is why
        calibration tools must treat ``None`` as a skipped sample rather than
        as a failed match.
        """
        result = self._preprocessor.extract(image, bbox)
        return self.embed_crop(result.image) if result.ok else None

    def extract_region(self, image: np.ndarray, bbox: BBox) -> CropResult:
        """Expose the preprocessor's decision, including why it refused."""
        return self._preprocessor.extract(image, bbox)


def _cached_face(cache) -> FaceDetection | None:
    """Rebuild a display-only FaceDetection from the track's cached geometry.

    Used for rendering between scheduled recognition passes so the face box
    does not flicker. It carries no identity claim of its own -- the identity
    shown still comes from the stabilizer's bounded hold.
    """
    if cache.last_face_bbox is None:
        return None
    return FaceDetection(
        bbox=cache.last_face_bbox,
        score=cache.last_face_score,
        landmarks=None,
        detail="cached geometry from the last recognition pass",
    )
