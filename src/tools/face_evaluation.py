"""Condition-split evaluation of the face identification pipeline.

Reports what was measured, per condition, with the open-set case treated as a
first-class outcome rather than an afterthought. Three things this deliberately
does *not* do:

* it never reports a single headline accuracy without the per-condition
  breakdown, because an average over easy and hard conditions hides both;
* it never fits a threshold on the same samples it reports against -- the
  calibration and test splits are disjoint, and mixing them is how systems come
  to report numbers they cannot reproduce;
* it never converts a similarity into a percentage.

Dataset layout (see ``scripts/build_evaluation_dataset.py``)::

    enrollment/<person_id>/reference.jpg
    query/<condition>/<person_id>/*.jpg
    unknown/<condition>/*.jpg
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.identity.calibration import (
    CalibrationModel,
    CalibrationSample,
)
from src.identity.types import FailureReason, IdentityStatus
from src.utils.image import DEFAULT_IMAGE_EXTENSIONS, imread, iter_images
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class QuerySample:
    """One labelled query image."""

    path: Path
    condition: str
    true_identity: str | None
    """``None`` means this person is deliberately not registered."""

    @property
    def is_known(self) -> bool:
        return self.true_identity is not None


@dataclass(slots=True)
class QueryOutcome:
    """What the system decided for one query."""

    sample: QuerySample
    status: IdentityStatus
    predicted_identity: str | None
    similarity: float
    quality: float
    visibility: str
    confidence: float | None
    failure: FailureReason | None
    reason: str = ""

    @property
    def correct(self) -> bool:
        """Was this the right answer?

        For a registered person, naming them. For an unknown person, *not*
        naming anybody -- rejection is the correct answer, not a failure.
        """
        if self.sample.is_known:
            return self.predicted_identity == self.sample.true_identity
        return self.predicted_identity is None

    @property
    def is_false_accept(self) -> bool:
        """An unregistered person given a registered name. The worst error."""
        return not self.sample.is_known and self.predicted_identity is not None

    @property
    def is_false_reject(self) -> bool:
        """A registered person not named."""
        return self.sample.is_known and self.predicted_identity is None

    @property
    def is_identity_error(self) -> bool:
        """A registered person named as a *different* registered person."""
        return (
            self.sample.is_known
            and self.predicted_identity is not None
            and self.predicted_identity != self.sample.true_identity
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.sample.path),
            "condition": self.sample.condition,
            "true_identity": self.sample.true_identity,
            "predicted_identity": self.predicted_identity,
            "status": self.status.value,
            "correct": self.correct,
            "similarity": round(self.similarity, 6),
            "quality": round(self.quality, 4),
            "visibility": self.visibility,
            "identity_confidence": (
                round(self.confidence, 4) if self.confidence is not None else None
            ),
            "failure": self.failure.value if self.failure else None,
            "failure_layer": self.failure.layer if self.failure else None,
        }


@dataclass(slots=True)
class ConditionMetrics:
    """Measured performance for one condition."""

    condition: str
    total: int = 0
    known_total: int = 0
    unknown_total: int = 0
    correct: int = 0
    rank1_hits: int = 0
    false_accepts: int = 0
    false_rejects: int = 0
    identity_errors: int = 0
    true_rejects: int = 0
    genuine_scores: list[float] = field(default_factory=list)
    impostor_scores: list[float] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def rank1_accuracy(self) -> float:
        """Did the correct identity rank first, ignoring the threshold?

        Separating this from accuracy distinguishes a *matching* failure (the
        wrong person ranks first) from a *threshold* failure (the right person
        ranks first but does not clear the bar). They need different fixes.
        """
        return self.rank1_hits / self.known_total if self.known_total else 0.0

    @property
    def tar(self) -> float:
        return (
            (self.known_total - self.false_rejects - self.identity_errors)
            / self.known_total
            if self.known_total
            else 0.0
        )

    @property
    def far(self) -> float:
        return self.false_accepts / self.unknown_total if self.unknown_total else 0.0

    @property
    def frr(self) -> float:
        return self.false_rejects / self.known_total if self.known_total else 0.0

    @property
    def unknown_rejection_rate(self) -> float:
        return self.true_rejects / self.unknown_total if self.unknown_total else 0.0

    @property
    def precision(self) -> float:
        named = self.correct_known + self.false_accepts + self.identity_errors
        return self.correct_known / named if named else 0.0

    @property
    def correct_known(self) -> int:
        return self.known_total - self.false_rejects - self.identity_errors

    @property
    def recall(self) -> float:
        return self.tar

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "samples": self.total,
            "known": self.known_total,
            "unknown": self.unknown_total,
            "accuracy": round(self.accuracy, 4),
            "rank1_accuracy": round(self.rank1_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tar": round(self.tar, 4),
            "far": round(self.far, 4),
            "frr": round(self.frr, 4),
            "unknown_rejection_rate": round(self.unknown_rejection_rate, 4),
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "identity_errors": self.identity_errors,
            "failures": dict(sorted(self.failures.items(), key=lambda kv: -kv[1])),
        }


@dataclass(slots=True)
class EvaluationReport:
    """The complete measured result."""

    dataset: str
    registered: list[str] = field(default_factory=list)
    conditions: dict[str, ConditionMetrics] = field(default_factory=dict)
    overall: ConditionMetrics = field(default_factory=lambda: ConditionMetrics("overall"))
    outcomes: list[QueryOutcome] = field(default_factory=list)
    calibration: CalibrationModel | None = None
    synthetic: bool = False
    caveats: list[str] = field(default_factory=list)
    config_summary: dict[str, Any] = field(default_factory=dict)

    def failure_analysis(self) -> dict[str, Any]:
        """Which layer failed, and for whom.

        The point of this breakdown is that "Unknown" is not a diagnosis. A
        person missed by the face detector and a person whose face was too
        small need different fixes, and this is what tells them apart.
        """
        by_layer: dict[str, int] = {}
        by_person: dict[str, dict[str, int]] = {}
        for outcome in self.outcomes:
            if outcome.correct or outcome.failure is None:
                continue
            layer = outcome.failure.layer
            by_layer[layer] = by_layer.get(layer, 0) + 1
            person = outcome.sample.true_identity or "<unknown>"
            bucket = by_person.setdefault(person, {})
            key = f"{outcome.sample.condition}/{outcome.failure.value}"
            bucket[key] = bucket.get(key, 0) + 1
        return {
            "by_layer": dict(sorted(by_layer.items(), key=lambda kv: -kv[1])),
            "by_person": {
                person: dict(sorted(cases.items(), key=lambda kv: -kv[1])[:6])
                for person, cases in sorted(by_person.items())
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "synthetic": self.synthetic,
            "caveats": self.caveats,
            "registered_identities": self.registered,
            "config": self.config_summary,
            "overall": self.overall.to_dict(),
            "conditions": {k: v.to_dict() for k, v in sorted(self.conditions.items())},
            "failure_analysis": self.failure_analysis(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "samples": [o.to_dict() for o in self.outcomes],
        }

    def render(self) -> str:
        lines: list[str] = []
        if self.synthetic:
            lines += [
                "=" * 72,
                "  SYNTHETIC EVALUATION SET -- these numbers describe image",
                "  transformations of a few real faces, not real-world conditions.",
                "  They are valid for A/B comparison and pipeline validation only.",
                "=" * 72,
                "",
            ]
        lines += [
            f"Face identification evaluation: {self.dataset}",
            f"  registered identities : {len(self.registered)}  {self.registered}",
            f"  query samples         : {self.overall.total}"
            f"  ({self.overall.known_total} known, {self.overall.unknown_total} unknown)",
            "",
            f"  {'CONDITION':<20} {'n':>4} {'acc':>7} {'rank1':>7} {'TAR':>7} "
            f"{'FAR':>7} {'FRR':>7} {'IDerr':>6}",
        ]
        for name in sorted(self.conditions):
            m = self.conditions[name]
            lines.append(
                f"  {name:<20} {m.total:>4} {m.accuracy:>6.1%} {m.rank1_accuracy:>6.1%} "
                f"{m.tar:>6.1%} {m.far:>6.1%} {m.frr:>6.1%} {m.identity_errors:>6}"
            )
        o = self.overall
        lines += [
            f"  {'-' * 68}",
            f"  {'OVERALL':<20} {o.total:>4} {o.accuracy:>6.1%} {o.rank1_accuracy:>6.1%} "
            f"{o.tar:>6.1%} {o.far:>6.1%} {o.frr:>6.1%} {o.identity_errors:>6}",
            "",
            f"  unknown rejection rate : {o.unknown_rejection_rate:.1%} "
            f"({o.true_rejects}/{o.unknown_total})",
            f"  precision / recall / F1: {o.precision:.3f} / {o.recall:.3f} / {o.f1:.3f}",
        ]

        analysis = self.failure_analysis()
        if analysis["by_layer"]:
            lines += ["", "  Where failures occurred:"]
            for layer, count in analysis["by_layer"].items():
                lines.append(f"    {layer:<20} {count}")
        if analysis["by_person"]:
            lines += ["", "  Hardest cases per person:"]
            for person, cases in analysis["by_person"].items():
                top = ", ".join(f"{k} x{v}" for k, v in list(cases.items())[:3])
                lines.append(f"    {person:<16} {top}")

        if self.caveats:
            lines += ["", "  Caveats:"]
            lines += [f"    - {c}" for c in self.caveats]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def load_dataset(root: Path) -> tuple[dict[str, Path], list[QuerySample], dict[str, Any]]:
    """Read enrollment references, query samples and the manifest."""
    root = Path(root)
    enrollment: dict[str, Path] = {}
    enrol_root = root / "enrollment"
    if enrol_root.is_dir():
        for person_dir in sorted(p for p in enrol_root.iterdir() if p.is_dir()):
            images = iter_images(person_dir, DEFAULT_IMAGE_EXTENSIONS)
            if images:
                enrollment[person_dir.name] = images[0]

    samples: list[QuerySample] = []
    query_root = root / "query"
    if query_root.is_dir():
        for condition_dir in sorted(p for p in query_root.iterdir() if p.is_dir()):
            for person_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
                for image in iter_images(person_dir, DEFAULT_IMAGE_EXTENSIONS):
                    samples.append(
                        QuerySample(image, condition_dir.name, person_dir.name)
                    )

    unknown_root = root / "unknown"
    if unknown_root.is_dir():
        for entry in sorted(unknown_root.iterdir()):
            if entry.is_dir():
                for image in iter_images(entry, DEFAULT_IMAGE_EXTENSIONS):
                    samples.append(QuerySample(image, entry.name, None))
            elif entry.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS:
                samples.append(QuerySample(entry, "normal", None))

    manifest: dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    return enrollment, samples, manifest


def split_samples(
    samples: Sequence[QuerySample], *, calibration_fraction: float = 0.4, seed: int = 11
) -> tuple[list[QuerySample], list[QuerySample]]:
    """Split into disjoint calibration and test sets.

    Stratified by (condition, identity) so both halves see every case, and
    disjoint so a threshold fitted on one is reported against the other. Tuning
    and reporting on the same samples produces a number that does not survive
    contact with new data.
    """
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[str, str | None], list[QuerySample]] = {}
    for sample in samples:
        buckets.setdefault((sample.condition, sample.true_identity), []).append(sample)

    calibration: list[QuerySample] = []
    test: list[QuerySample] = []
    for group in buckets.values():
        indices = rng.permutation(len(group))
        cut = int(round(len(group) * calibration_fraction))
        for position, index in enumerate(indices):
            (calibration if position < cut else test).append(group[index])
    return calibration, test


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FacePipeline:
    """The minimal surface the evaluator needs from a configured system."""

    detector: Any
    encoder: Any
    gallery: Any
    engine: Any
    chip_size: int = 112

    def embed_query(self, image: np.ndarray):
        """Detect, align, assess and embed the dominant face in one image.

        Returns ``(embedding, quality, visibility, face)``; any element may be
        ``None`` when the stage it comes from failed.
        """
        from src.face.align import align_face
        from src.face.quality import assess_face_quality
        from src.face.visibility import classify_visibility
        from src.reid.encoder import l2_normalize

        faces = self.detector.detect(image)
        if not faces:
            return None, None, None, None
        face = max(faces, key=lambda f: f.bbox.area)
        if not face.has_landmarks:
            return None, None, None, face
        visibility = classify_visibility(image, face.bbox, face.landmarks)
        try:
            chip = align_face(image, face.landmarks, self.chip_size)
        except Exception:  # noqa: BLE001 - cv2 raises broadly
            return None, None, visibility, face
        quality = assess_face_quality(chip, face, visibility)
        return l2_normalize(self.encoder.embed(chip)), quality, visibility, face


def collect_calibration_samples(
    pipeline: FacePipeline, samples: Sequence[QuerySample]
) -> list[CalibrationSample]:
    """Score the calibration split against the gallery, labelled."""
    collected: list[CalibrationSample] = []
    for sample in samples:
        try:
            image = imread(sample.path)
        except Exception:  # noqa: BLE001
            continue
        embedding, quality, visibility, _ = pipeline.embed_query(image)
        if embedding is None or pipeline.gallery.is_empty:
            continue
        match = pipeline.gallery.match(embedding)
        condition = visibility.visibility.threshold_key if visibility else "full"
        for identity_id, score in match.ranked:
            collected.append(
                CalibrationSample(
                    similarity=float(score),
                    is_genuine=(identity_id == sample.true_identity),
                    quality=quality.overall if quality else 0.0,
                    condition=condition,
                )
            )
    return collected


def evaluate(
    pipeline: FacePipeline,
    samples: Sequence[QuerySample],
    *,
    dataset: str = "",
    registered: Sequence[str] = (),
    synthetic: bool = False,
    caveats: Sequence[str] = (),
    progress: Callable[[int, int], None] | None = None,
) -> EvaluationReport:
    """Run every query through the decision engine and measure the result."""
    report = EvaluationReport(
        dataset=dataset,
        registered=list(registered),
        synthetic=synthetic,
        caveats=list(caveats),
        calibration=pipeline.engine.calibration,
    )

    for index, sample in enumerate(samples, start=1):
        if progress and index % 25 == 0:
            progress(index, len(samples))
        try:
            image = imread(sample.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping unreadable %s: %s", sample.path, exc)
            continue

        embedding, quality, visibility, face = pipeline.embed_query(image)
        from src.face.visibility import FaceVisibility

        decision = pipeline.engine.decide(
            detector_confidence=1.0,          # the query crop is the person
            face_detection_confidence=face.score if face else 0.0,
            embedding=embedding,
            quality=quality,
            visibility=visibility.visibility if visibility else FaceVisibility.NO_USABLE_FACE,
            track_state=None,                 # still images: no temporal evidence
        )

        predicted = decision.identity_id if decision.is_named else None
        outcome = QueryOutcome(
            sample=sample,
            status=decision.status,
            predicted_identity=predicted,
            similarity=decision.face_similarity,
            quality=decision.face_quality,
            visibility=decision.visibility.value,
            confidence=decision.identity_confidence,
            failure=decision.failure,
            reason=decision.reason,
        )
        report.outcomes.append(outcome)

        metrics = report.conditions.setdefault(
            sample.condition, ConditionMetrics(sample.condition)
        )
        for bucket in (metrics, report.overall):
            bucket.total += 1
            if sample.is_known:
                bucket.known_total += 1
            else:
                bucket.unknown_total += 1
            if outcome.correct:
                bucket.correct += 1
            if outcome.is_false_accept:
                bucket.false_accepts += 1
            if outcome.is_false_reject:
                bucket.false_rejects += 1
            if outcome.is_identity_error:
                bucket.identity_errors += 1
            if not sample.is_known and predicted is None:
                bucket.true_rejects += 1
            if outcome.failure is not None and not outcome.correct:
                key = outcome.failure.value
                bucket.failures[key] = bucket.failures.get(key, 0) + 1

        # Rank-1: did the right person top the ranking, threshold aside? This
        # separates a matching failure from a threshold failure.
        if sample.is_known and decision.match is not None:
            if decision.match.best_identity_id == sample.true_identity:
                metrics.rank1_hits += 1
                report.overall.rank1_hits += 1
            for identity_id, score in decision.match.ranked:
                target = (
                    metrics.genuine_scores
                    if identity_id == sample.true_identity
                    else metrics.impostor_scores
                )
                target.append(float(score))

    return report
