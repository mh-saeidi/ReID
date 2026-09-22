"""The live pipeline for passport-photo identity.

One frame in, one :class:`ProcessOutcome` out, in the order the architecture
requires::

    frame -> YOLO26 person detection (+ tracking)
          -> face detection, once per frame, then assigned to person boxes
          -> face alignment
          -> face quality / visibility
          -> face embedding
          -> identity gallery search
          -> identity matching (visibility-specific, calibrated threshold)
          -> temporal evidence per track
          -> final identity decision

This is the runtime counterpart of the evaluation path: both drive the same
:class:`IdentityDecisionEngine`, so what is measured offline is what runs
online. It is a separate class from :class:`~src.pipeline.processor.ReIDPipeline`
rather than a mode inside it, because the two differ in what an identity *is*:
the older path stabilises a body-appearance match, this one accumulates
quality-weighted face evidence and can refuse to name anybody at all.

Face detection runs once on the whole frame and its results are assigned to
person boxes, rather than running per person crop. With several people in view
that is one detector call instead of N, and a face that belongs to nobody the
person detector found is dropped rather than silently promoted into a
detection of its own -- the architecture identifies *people*, and a face with
no person attached has no track to accumulate evidence on.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config.schema import AppConfig
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
from src.face.align import align_face
from src.face.quality import assess_face_quality
from src.face.types import FaceDetection
from src.face.visibility import FaceVisibility, VisibilityReport, classify_visibility
from src.identity.factory import FaceIdentitySystem
from src.identity.state_machine import TrackIdentityState
from src.identity.types import FailureReason, IdentityDecision, IdentityStatus
from src.pipeline.metrics import MetricsCollector, Stopwatch
from src.pipeline.processor import ProcessOutcome
from src.reid.encoder import l2_normalize
from src.tracking.scheduler import RecognitionScheduler
from src.tracking.track_manager import TrackManager
from src.utils.logging import get_logger

logger = get_logger(__name__)

# How an identity status is presented downstream.
#
# Two mappings are load-bearing. NO_FACE stays distinct from UNKNOWN, because
# "nothing to compare" is not "compared and matched nobody". And UNCERTAIN --
# a candidate leads but the evidence does not yet support naming them -- maps
# to UNKNOWN rather than LOW_CONFIDENCE: LOW_CONFIDENCE counts as recognized
# downstream, and nobody has been named. The distinction survives in the
# decision's reason and failure_reason.
_STATUS_MAP = {
    IdentityStatus.RECOGNIZED: RecognitionStatus.RECOGNIZED,
    IdentityStatus.TEMPORARILY_MAINTAINED: RecognitionStatus.RECOGNIZED,
    IdentityStatus.UNCERTAIN: RecognitionStatus.UNKNOWN,
    IdentityStatus.UNKNOWN: RecognitionStatus.UNKNOWN,
    IdentityStatus.NO_FACE: RecognitionStatus.NO_FACE,
    IdentityStatus.PENDING: RecognitionStatus.PENDING,
}


@dataclass(slots=True)
class FaceObservation:
    """Everything the face layers produced for one person box."""

    face: FaceDetection | None = None
    embedding: np.ndarray | None = None
    quality: object | None = None
    visibility: VisibilityReport | None = None
    failure: FailureReason | None = None
    detail: str = ""
    skipped: bool = False
    """The scheduler spent no pass on this track: a cost decision, not an
    identity one, so it must not be turned into a NO_FACE result."""

    @property
    def visibility_class(self) -> FaceVisibility:
        return (
            self.visibility.visibility if self.visibility is not None
            else FaceVisibility.NO_USABLE_FACE
        )


def assign_faces_to_people(
    faces: Sequence[FaceDetection], detections: Sequence[Detection]
) -> dict[int, FaceDetection]:
    """Map each person box to the best face inside it.

    A face belongs to the person box that contains its centre; when several
    boxes do (someone standing behind someone else), the smallest wins, since
    the nearer person's box is the tighter one. Each person keeps only their
    largest face, and each face is used once.
    """
    assignments: dict[int, FaceDetection] = {}
    for face in faces:
        cx, cy = face.bbox.center
        best_index, best_area = None, float("inf")
        for index, detection in enumerate(detections):
            box = detection.bbox
            if box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2 and box.area < best_area:
                best_index, best_area = index, box.area
        if best_index is None:
            continue
        current = assignments.get(best_index)
        if current is None or face.bbox.area > current.bbox.area:
            assignments[best_index] = face
    return assignments


class FaceIdentityPipeline:
    """Person detection + tracking + face identity for one stream of frames."""

    def __init__(
        self,
        config: AppConfig,
        detector: Detector,
        system: FaceIdentitySystem,
        *,
        metrics: MetricsCollector | None = None,
        use_tracking: bool = True,
    ) -> None:
        self._config = config
        self._detector = detector
        self._system = system
        self._metrics = metrics or MetricsCollector()
        self._use_tracking = use_tracking and config.tracking.enabled
        self._scheduler = RecognitionScheduler(
            config.recognition_scheduler, config.matching
        )
        self._tracks = TrackManager(
            config.tracking,
            config.matching,
            absent_hold_frames=config.temporal.occlusion_hold_frames,
        )
        # Identity state is keyed by track id and lives beside the track's own
        # bookkeeping, not inside it: a track is a trajectory, an identity is a
        # claim about a person, and the two have different lifetimes.
        self._identity_states: dict[int, TrackIdentityState] = {}

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
    def gallery(self):
        return self._system.gallery

    @property
    def uses_tracking(self) -> bool:
        return self._use_tracking

    def identity_state(self, track_id: int) -> TrackIdentityState | None:
        return self._identity_states.get(track_id)

    def reset(self) -> None:
        self._tracks.reset()
        self._scheduler.reset()
        self._identity_states.clear()
        if self._use_tracking:
            self._detector.reset_tracker()

    # ------------------------------------------------------------- processing
    def process(self, frame: Frame) -> ProcessOutcome:
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
            observations = self._observe_faces(frame, detections, states)

        with Stopwatch(self._metrics, "matching"):
            decisions = self._decide(frame, detections, observations)

        results = self._finalize(frame, detections, states, decisions)

        for state in self._tracks.tracks.values():
            if state.last_seen_frame != frame.index:
                state.recognition_cache.mark_lost()
        ended = [state.track_id for state in self._tracks.expire(frame.index)]
        for track_id in ended:
            self._identity_states.pop(track_id, None)

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
        self._metrics.record_frame(
            detections=len(results),
            recognized=len(result.recognized),
            tracks=self._tracks.active_count,
            no_face=sum(
                1 for d in results
                if d.recognition_status is RecognitionStatus.NO_FACE
            ),
            faces=sum(1 for d in results if d.face is not None),
        )
        self._metrics.scheduler_stats = self._scheduler.stats.to_dict()
        return ProcessOutcome(result, [], new_tracks, ended)

    # ------------------------------------------------------------------ steps
    def _bind_tracks(self, detections: Sequence[Detection], frame: Frame):
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
                self._identity_states[track_id] = TrackIdentityState(
                    track_id=track_id,
                    first_seen_frame=frame.index,
                    last_seen_frame=frame.index,
                )
            identity_state = self._identity_states.get(track_id)
            if identity_state is not None:
                identity_state.last_seen_frame = frame.index
            states.append(state)
        return states, new_tracks

    def _observe_faces(
        self, frame: Frame, detections: Sequence[Detection], states: Sequence
    ) -> list[FaceObservation]:
        """Detect, align, assess and embed one face per eligible person."""
        observations = [FaceObservation() for _ in detections]
        self._scheduler.begin_frame(frame.index)

        eligible = []
        for index, state in enumerate(states):
            cache = state.recognition_cache
            cache.track_id = state.track_id
            if self._should_recognize(state, frame.index):
                cache.note_recognition(frame.index, frame.timestamp)
                eligible.append(index)
            else:
                cache.note_skip()
                observations[index].skipped = True
                if cache.last_face_bbox is not None:
                    # Presentation only: show where the face last was, so the
                    # overlay does not flicker between scheduled passes. No
                    # identity is asserted from it.
                    detections[index].face = FaceDetection(
                        bbox=cache.last_face_bbox,
                        score=cache.last_face_score,
                        landmarks=None,
                        detail="cached geometry from the last recognition pass",
                    )

        if not eligible:
            return observations

        with Stopwatch(self._metrics, "face_detection"):
            faces = self._system.detector.detect(frame.image)
        self._metrics.face_detect_calls += 1
        assigned = assign_faces_to_people(faces, detections)

        chips: list[np.ndarray] = []
        slots: list[int] = []
        for index in eligible:
            observation = observations[index]
            face = assigned.get(index)
            if face is None:
                observation.failure = FailureReason.FACE_NOT_FOUND
                observation.detail = "no face was found inside this person's box"
                continue

            observation.face = face
            detections[index].face = face
            if not face.has_landmarks:
                observation.failure = FailureReason.FACE_ALIGNMENT_FAILED
                observation.detail = (
                    "the face detector returned no landmarks, so the face "
                    "cannot be aligned onto the template"
                )
                continue

            observation.visibility = classify_visibility(
                frame.image, face.bbox, face.landmarks
            )
            try:
                chip = align_face(frame.image, face.landmarks, self._system.chip_size)
            except Exception as exc:  # noqa: BLE001 - cv2 raises broadly
                observation.failure = FailureReason.FACE_ALIGNMENT_FAILED
                observation.detail = str(exc)
                logger.debug(
                    "Face alignment failed",
                    extra={"track_id": detections[index].track_id, "error": str(exc)},
                )
                continue

            observation.quality = assess_face_quality(chip, face, observation.visibility)
            chips.append(chip)
            slots.append(index)

        if not chips:
            return observations

        with Stopwatch(self._metrics, "encoder"):
            embedded = self._system.encoder.embed_batch(chips)
        self._metrics.reid_calls += 1
        self._metrics.reid_crops += len(chips)
        self._metrics.record_batch(len(chips))

        for slot, vector in zip(slots, embedded, strict=True):
            if not np.any(vector):
                observations[slot].failure = FailureReason.FACE_EMBEDDING_FAILED
                observations[slot].detail = "the encoder returned an empty vector"
                continue
            observations[slot].embedding = l2_normalize(vector)
            state = states[slot]
            state.last_reid_frame = state.last_seen_frame
            state.reid_runs += 1
            state.recognition_cache.note_embedding(
                frame.index, int(observations[slot].embedding.shape[-1])
            )
        return observations

    def _should_recognize(self, state, frame_index: int) -> bool:
        if self._scheduler.enabled:
            return self._scheduler.decide(state, frame_index).should_recognize
        return self._tracks.needs_reid(
            state, state.last_seen_frame, self._config.performance
        )

    def _decide(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        observations: Sequence[FaceObservation],
    ) -> list[IdentityDecision | None]:
        decisions: list[IdentityDecision | None] = []
        for detection, observation in zip(detections, observations, strict=True):
            if observation.skipped:
                # Skipped by the scheduler: a decision about cost, not identity.
                decisions.append(None)
                continue

            # Temporal evidence exists only where there is a sequence. On a
            # still image there is one frame, so demanding several confirming
            # observations would demand the impossible and answer "unknown" to
            # every photograph. Without tracking, each frame is decided on its
            # own merits.
            identity_state = (
                self._identity_states.get(detection.track_id or -1)
                if self._use_tracking else None
            )
            decision = self._system.engine.decide(
                detector_confidence=detection.confidence,
                face_detection_confidence=(
                    observation.face.score if observation.face else 0.0
                ),
                embedding=observation.embedding,
                quality=observation.quality,
                visibility=observation.visibility_class,
                track_state=identity_state,
                frame_index=frame.index,
                timestamp=frame.timestamp,
                face_failure=observation.failure,
                face_detail=observation.detail,
            )
            decisions.append(decision)

            if (
                decision.status is IdentityStatus.RECOGNIZED
                and observation.embedding is not None
                and decision.match is not None
                and identity_state is not None
            ):
                self._system.adapter.consider(
                    embedding=observation.embedding,
                    match=decision.match,
                    quality=observation.quality,
                    visibility=observation.visibility_class,
                    track_state=identity_state,
                    frame_index=frame.index,
                )
        return decisions

    def _finalize(
        self,
        frame: Frame,
        detections: Sequence[Detection],
        states: Sequence,
        decisions: Sequence[IdentityDecision | None],
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for detection, state, decision in zip(detections, states, decisions, strict=True):
            if decision is None:
                # No pass this frame: keep showing the track's last conclusion
                # rather than flickering to "pending". Nothing new is claimed.
                recognition = state.last_result or RecognitionResult(
                    status=RecognitionStatus.PENDING,
                    identity_name=self._config.matching.unknown_label,
                )
                stabilized = state.last_result
            else:
                recognition = self._to_recognition(decision)
                state.last_result = recognition
                stabilized = None
                if recognition.is_recognized:
                    state.frames_recognized += 1
                elif recognition.status is RecognitionStatus.NO_FACE:
                    state.frames_no_face += 1
                else:
                    state.frames_unknown += 1
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
        return results

    def _to_recognition(self, decision: IdentityDecision) -> RecognitionResult:
        """Present one identity decision, keeping every quantity separate."""
        match = decision.match
        identity = (
            self._system.gallery.get(decision.identity_id)
            if decision.identity_id else None
        )
        return RecognitionResult(
            status=_STATUS_MAP.get(decision.status, RecognitionStatus.UNKNOWN),
            identity_id=decision.identity_id,
            identity_name=(
                identity.name if identity is not None
                else decision.name or self._label_for(decision)
            ),
            identity_title=identity.title if identity is not None else decision.title,
            similarity=decision.face_similarity,
            runner_up_id=match.runner_up_id if match else None,
            runner_up_similarity=match.runner_up_similarity if match else 0.0,
            detail=decision.reason,
            identity_confidence=decision.identity_confidence,
            face_quality=decision.face_quality,
            face_detection_confidence=decision.face_detection_confidence,
            visibility=decision.visibility.value,
            failure_reason=decision.failure.value if decision.failure else "",
        )

    def _label_for(self, decision: IdentityDecision) -> str:
        if decision.status is IdentityStatus.NO_FACE:
            return self._config.matching.no_face_label
        return self._config.matching.unknown_label

    # ----------------------------------------------------------------- images
    def process_image(self, image: np.ndarray, *, source_id: str = "image") -> FrameResult:
        """Identify everybody in a standalone image (no tracking, no history)."""
        frame = Frame(image=image, index=0, timestamp=time.time(), source_id=source_id)
        was_tracking = self._use_tracking
        self._use_tracking = False
        try:
            return self.process(frame).result
        finally:
            self._use_tracking = was_tracking

    def embed_bbox(self, image: np.ndarray, bbox: BBox) -> np.ndarray | None:
        """Embed the face inside one person box, or None when there is none."""
        faces = self._system.detector.detect(image)
        inside = [
            face for face in faces
            if bbox.x1 <= face.bbox.center[0] <= bbox.x2
            and bbox.y1 <= face.bbox.center[1] <= bbox.y2
            and face.has_landmarks
        ]
        if not inside:
            return None
        face = max(inside, key=lambda f: f.bbox.area)
        try:
            chip = align_face(image, face.landmarks, self._system.chip_size)
        except Exception:  # noqa: BLE001 - cv2 raises broadly
            return None
        return l2_normalize(self._system.encoder.embed(chip))
