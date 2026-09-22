"""Threshold calibration and evaluation.

Threshold choice is the hardest practical problem in one-shot ReID, and there
is no universally correct value: it depends on the encoder, the cameras, the
lighting and the population. This tool measures the similarity distributions on
*your* data so the threshold is an empirical deployment parameter rather than a
guess.

Expected dataset layout::

    evaluation/
      known/
        john_doe/      # directory name == a person id in the configuration
          img_01.jpg
        jane_smith/
      unknown/         # people who are NOT registered
        stranger_01.jpg

Genuine scores come from ``known/<id>/*`` compared against that id's gallery
entry; impostor scores come from every other comparison, including all of
``unknown/``. The reported operating points are computed from those two
distributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import ConfigurationError
from src.core.types import RecognitionStatus
from src.identity.matcher import IdentityMatcher
from src.pipeline.engine import Engine
from src.pipeline.processor import ReIDPipeline
from src.utils.image import DEFAULT_IMAGE_EXTENSIONS, imread, iter_images
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Sample:
    """One evaluation crop and the ground truth attached to it."""

    path: str
    true_identity: str | None
    predicted_identity: str | None
    similarity: float
    status: RecognitionStatus
    genuine_similarity: float | None = None
    """Similarity against the *correct* identity, when that identity exists."""


@dataclass(slots=True)
class OperatingPoint:
    threshold: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    identity_errors: int
    """Accepted above threshold but matched to the wrong registered person."""

    @property
    def accuracy(self) -> float:
        total = (
            self.true_accepts
            + self.false_accepts
            + self.true_rejects
            + self.false_rejects
            + self.identity_errors
        )
        return (self.true_accepts + self.true_rejects) / total if total else 0.0

    @property
    def precision(self) -> float:
        accepted = self.true_accepts + self.false_accepts + self.identity_errors
        return self.true_accepts / accepted if accepted else 0.0

    @property
    def recall(self) -> float:
        genuine = self.true_accepts + self.false_rejects + self.identity_errors
        return self.true_accepts / genuine if genuine else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "true_accepts": self.true_accepts,
            "false_accepts": self.false_accepts,
            "true_rejects": self.true_rejects,
            "false_rejects": self.false_rejects,
            "identity_errors": self.identity_errors,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass(slots=True)
class CalibrationReport:
    """Measured similarity distributions and candidate operating points."""

    genuine_scores: list[float] = field(default_factory=list)
    impostor_scores: list[float] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    operating_points: list[OperatingPoint] = field(default_factory=list)
    configured_threshold: float = 0.0
    skipped: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _stats(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float32)
        return {
            "count": int(array.size),
            "min": round(float(array.min()), 4),
            "p05": round(float(np.percentile(array, 5)), 4),
            "mean": round(float(array.mean()), 4),
            "median": round(float(np.median(array)), 4),
            "p95": round(float(np.percentile(array, 95)), 4),
            "max": round(float(array.max()), 4),
            "std": round(float(array.std()), 4),
        }

    @property
    def best_point(self) -> OperatingPoint | None:
        """Best operating point, breaking F1 ties by margin.

        Many thresholds often tie at the top when the two distributions are
        well separated. Returning the first of them would sit right on the edge
        of the impostor distribution, so the *middle* of the tied run is chosen
        instead -- the point furthest from both failure modes.
        """
        if not self.operating_points:
            return None
        best_f1 = max(p.f1 for p in self.operating_points)
        tied = [p for p in self.operating_points if p.f1 == best_f1]
        return tied[len(tied) // 2]

    @property
    def separation(self) -> dict[str, float] | None:
        """Gap between the impostor and genuine distributions.

        A positive gap means the two never overlap on this data, and any
        threshold inside it separates them perfectly. The midpoint is the
        safest choice: it is as far as possible from both a false accept and a
        false reject.
        """
        if not self.genuine_scores or not self.impostor_scores:
            return None
        genuine_floor = min(self.genuine_scores)
        impostor_ceiling = max(self.impostor_scores)
        return {
            "impostor_max": round(impostor_ceiling, 4),
            "genuine_min": round(genuine_floor, 4),
            "gap": round(genuine_floor - impostor_ceiling, 4),
            "midpoint": round((genuine_floor + impostor_ceiling) / 2.0, 4),
        }

    @property
    def suggested_threshold(self) -> float | None:
        """The threshold this tool actually recommends for this data."""
        separation = self.separation
        if separation is not None and separation["gap"] > 0.0:
            return separation["midpoint"]
        best = self.best_point
        return round(best.threshold, 4) if best else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "genuine": self._stats(self.genuine_scores),
            "impostor": self._stats(self.impostor_scores),
            "configured_threshold": self.configured_threshold,
            "operating_points": [p.to_dict() for p in self.operating_points],
            "best_f1_threshold": round(self.best_point.threshold, 4) if self.best_point else None,
            "separation": self.separation,
            "suggested_threshold": self.suggested_threshold,
            "skipped": self.skipped,
            "samples": [
                {
                    "path": s.path,
                    "true": s.true_identity,
                    "predicted": s.predicted_identity,
                    "similarity": round(s.similarity, 4),
                    "genuine_similarity": (
                        round(s.genuine_similarity, 4) if s.genuine_similarity is not None else None
                    ),
                    "status": s.status.value,
                }
                for s in self.samples
            ],
        }

    def render(self) -> str:
        lines = ["Threshold calibration (measured on the supplied dataset)", ""]
        genuine, impostor = self._stats(self.genuine_scores), self._stats(self.impostor_scores)
        if genuine:
            lines.append(
                f"  genuine  (same person)  n={genuine['count']:<4} "
                f"min={genuine['min']:.3f} p05={genuine['p05']:.3f} "
                f"mean={genuine['mean']:.3f} max={genuine['max']:.3f}"
            )
        if impostor:
            lines.append(
                f"  impostor (other/unknown) n={impostor['count']:<4} "
                f"min={impostor['min']:.3f} mean={impostor['mean']:.3f} "
                f"p95={impostor['p95']:.3f} max={impostor['max']:.3f}"
            )
        lines += ["", "  threshold  TA   FA   TR   FR   IDerr  prec   recall  F1"]
        for point in self.operating_points:
            lines.append(
                f"  {point.threshold:8.2f}  {point.true_accepts:<4} {point.false_accepts:<4} "
                f"{point.true_rejects:<4} {point.false_rejects:<4} {point.identity_errors:<5}  "
                f"{point.precision:.3f}  {point.recall:.3f}   {point.f1:.3f}"
            )
        best = self.best_point
        separation = self.separation
        if best is not None:
            lines += [
                "",
                f"  Best F1 on THIS dataset: threshold = {best.threshold:.2f} "
                f"(F1 {best.f1:.3f}); configured = {self.configured_threshold:.2f}.",
            ]
        if separation is not None:
            if separation["gap"] > 0:
                lines.append(
                    f"  Clean separation: impostors peak at {separation['impostor_max']:.3f}, "
                    f"genuine matches bottom out at {separation['genuine_min']:.3f} "
                    f"(gap {separation['gap']:.3f})."
                )
            else:
                lines.append(
                    f"  OVERLAP: impostors reach {separation['impostor_max']:.3f} while "
                    f"genuine matches fall to {separation['genuine_min']:.3f}. No threshold "
                    "separates them perfectly -- decide which error you prefer."
                )
        if self.suggested_threshold is not None:
            lines.append(f"  Suggested threshold for this data: {self.suggested_threshold:.2f}")
        lines += [
            "  This is a measurement of this dataset, not a universal value:",
            "  re-run it per deployment (cameras, lighting, population).",
        ]
        if self.skipped:
            lines += ["", f"  Skipped {len(self.skipped)} image(s); see the JSON report."]
        return "\n".join(lines)


def _collect(directory: Path, extensions: Sequence[str]) -> list[Path]:
    return iter_images(directory, extensions) if directory.is_dir() else []


def evaluate_dataset(
    engine: Engine,
    pipeline: ReIDPipeline,
    dataset_dir: Path,
    *,
    thresholds: Sequence[float] | None = None,
    extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> CalibrationReport:
    """Measure genuine/impostor similarity distributions on a labelled dataset."""
    dataset_dir = Path(dataset_dir)
    known_dir = dataset_dir / "known"
    unknown_dir = dataset_dir / "unknown"
    if not known_dir.is_dir() and not unknown_dir.is_dir():
        raise ConfigurationError(
            f"evaluation dataset {dataset_dir} must contain a 'known/<person_id>/' "
            "directory and/or an 'unknown/' directory"
        )

    view = engine.gallery.view
    if view.is_empty:
        raise ConfigurationError(
            "the gallery is empty; run 'python main.py gallery build' before evaluating"
        )
    matcher = IdentityMatcher(engine.config.matching)
    report = CalibrationReport(configured_threshold=engine.config.matching.recognition_threshold)

    tasks: list[tuple[Path, str | None]] = []
    for person_dir in sorted(p for p in known_dir.iterdir() if p.is_dir()) if known_dir.is_dir() else []:
        if person_dir.name not in view.ids:
            logger.warning(
                "Evaluation directory has no matching gallery identity; treating it as unknown",
                extra={"directory": person_dir.name},
            )
            tasks.extend((path, None) for path in _collect(person_dir, extensions))
            continue
        tasks.extend((path, person_dir.name) for path in _collect(person_dir, extensions))
    tasks.extend((path, None) for path in _collect(unknown_dir, extensions))

    if not tasks:
        raise ConfigurationError(f"no evaluation images found under {dataset_dir}")

    for path, true_identity in tasks:
        try:
            image = imread(path)
        except Exception as exc:  # noqa: BLE001
            report.skipped[str(path)] = str(exc)
            continue

        frame_result = pipeline.process_image(image, source_id=f"eval:{path.name}")
        if not frame_result.detections:
            report.skipped[str(path)] = "no person detected"
            continue

        # Use the largest detection: evaluation crops are single-person by convention.
        detection = max(frame_result.detections, key=lambda d: d.bbox.area)
        region = pipeline.extract_region(image, detection.bbox)
        if not region.ok:
            # A sample with no usable face is *not* evidence about any
            # threshold, so it is excluded rather than scored as a miss.
            report.skipped[str(path)] = region.detail or region.status.value
            continue
        embedding = pipeline.embed_crop(region.image)
        if embedding is None or not np.any(embedding):
            report.skipped[str(path)] = "encoder returned an empty embedding"
            continue

        scores = matcher.similarity_matrix(embedding.reshape(1, -1), view)[0]
        best_index = int(np.argmax(scores))
        best_id, best_score = view.ids[best_index], float(scores[best_index])

        genuine_score: float | None = None
        if true_identity is not None and true_identity in view.ids:
            genuine_score = float(scores[view.ids.index(true_identity)])
            report.genuine_scores.append(genuine_score)
            report.impostor_scores.extend(
                float(scores[i]) for i, ident in enumerate(view.ids) if ident != true_identity
            )
        else:
            report.impostor_scores.extend(float(value) for value in scores)

        report.samples.append(
            Sample(
                path=str(path),
                true_identity=true_identity,
                predicted_identity=best_id,
                similarity=best_score,
                status=detection.recognition_status,
                genuine_similarity=genuine_score,
            )
        )

    report.operating_points = compute_operating_points(report.samples, thresholds)
    return report


def compute_operating_points(
    samples: Sequence[Sample], thresholds: Sequence[float] | None = None
) -> list[OperatingPoint]:
    """Score each candidate threshold against the labelled samples."""
    if thresholds is None:
        thresholds = [round(v, 2) for v in np.arange(0.20, 0.96, 0.05)]

    points: list[OperatingPoint] = []
    for threshold in thresholds:
        true_accepts = false_accepts = true_rejects = false_rejects = identity_errors = 0
        for sample in samples:
            accepted = sample.similarity >= threshold
            if sample.true_identity is None:
                # An unregistered person: acceptance of any identity is a false accept.
                if accepted:
                    false_accepts += 1
                else:
                    true_rejects += 1
                continue
            if not accepted:
                false_rejects += 1
            elif sample.predicted_identity == sample.true_identity:
                true_accepts += 1
            else:
                identity_errors += 1
        points.append(
            OperatingPoint(
                threshold=float(threshold),
                true_accepts=true_accepts,
                false_accepts=false_accepts,
                true_rejects=true_rejects,
                false_rejects=false_rejects,
                identity_errors=identity_errors,
            )
        )
    return points
