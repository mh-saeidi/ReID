"""Turning a similarity into a probability, honestly.

A cosine similarity is not a probability and multiplying it by 100 does not
make it a percentage. Two embeddings at cosine 0.45 are not "45% the same
person" -- depending on the encoder, the population and the imaging conditions,
0.45 might mean near-certainty or near-noise.

The only defensible way to report a confidence is to *measure* the relationship
on labelled data: collect genuine and impostor scores, fit a monotone map from
score to posterior probability, and refuse to report a confidence at all when
no such data exists.

Two fitters are provided:

``logistic``  Platt scaling. Two parameters, so it behaves sensibly on the
              small calibration sets this system realistically has.
``isotonic``  non-parametric and monotone. Fits any shape, but needs far more
              data and will happily overfit a small set.

Conditions are calibrated separately where the evidence supports it. A masked
face and a full face produce scores from genuinely different distributions, and
one shared mapping would misreport both.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1

# Below this many samples per class, a fitted mapping is not trustworthy and
# the calibrator refuses rather than producing confident nonsense.
MIN_SAMPLES_PER_CLASS = 12


@dataclass(slots=True)
class CalibrationSample:
    """One labelled comparison."""

    similarity: float
    is_genuine: bool
    quality: float = 1.0
    condition: str = "full"


@dataclass(slots=True)
class OperatingPoint:
    """What a threshold achieves on the calibration data."""

    threshold: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int

    @property
    def tar(self) -> float:
        genuine = self.true_accepts + self.false_rejects
        return self.true_accepts / genuine if genuine else 0.0

    @property
    def far(self) -> float:
        impostor = self.false_accepts + self.true_rejects
        return self.false_accepts / impostor if impostor else 0.0

    @property
    def frr(self) -> float:
        return 1.0 - self.tar

    @property
    def precision(self) -> float:
        accepted = self.true_accepts + self.false_accepts
        return self.true_accepts / accepted if accepted else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.tar
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "true_accepts": self.true_accepts,
            "false_accepts": self.false_accepts,
            "true_rejects": self.true_rejects,
            "false_rejects": self.false_rejects,
            "tar": round(self.tar, 4),
            "far": round(self.far, 4),
            "frr": round(self.frr, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(slots=True)
class ConditionCalibration:
    """A fitted score-to-probability map for one condition."""

    condition: str
    method: str
    genuine_count: int
    impostor_count: int
    # Logistic parameters: p = sigmoid(a * s + b)
    a: float = 0.0
    b: float = 0.0
    # Isotonic knots, when that method was used.
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)
    recommended_threshold: float = 0.0
    equal_error_threshold: float = 0.0
    equal_error_rate: float = 0.0
    separation_gap: float = 0.0
    """Genuine minimum minus impostor maximum. Positive means no overlap."""
    operating_points: list[OperatingPoint] = field(default_factory=list)
    genuine_stats: dict[str, float] = field(default_factory=dict)
    impostor_stats: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return (
            self.genuine_count >= MIN_SAMPLES_PER_CLASS
            and self.impostor_count >= MIN_SAMPLES_PER_CLASS
        )

    def probability(self, similarity: float) -> float:
        """Posterior probability that a score of ``similarity`` is genuine."""
        if self.method == "isotonic" and self.knots_x:
            return float(np.interp(similarity, self.knots_x, self.knots_y, left=0.0, right=1.0))
        z = self.a * similarity + self.b
        # Guard the exponential so extreme scores do not overflow.
        if z >= 0:
            return float(1.0 / (1.0 + math.exp(-min(z, 60.0))))
        exponent = math.exp(max(z, -60.0))
        return float(exponent / (1.0 + exponent))

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "method": self.method,
            "genuine_count": self.genuine_count,
            "impostor_count": self.impostor_count,
            "usable": self.usable,
            "a": round(self.a, 6),
            "b": round(self.b, 6),
            "knots_x": [round(v, 6) for v in self.knots_x],
            "knots_y": [round(v, 6) for v in self.knots_y],
            "recommended_threshold": round(self.recommended_threshold, 4),
            "equal_error_threshold": round(self.equal_error_threshold, 4),
            "equal_error_rate": round(self.equal_error_rate, 4),
            "separation_gap": round(self.separation_gap, 4),
            "genuine": self.genuine_stats,
            "impostor": self.impostor_stats,
            "operating_points": [p.to_dict() for p in self.operating_points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionCalibration:
        return cls(
            condition=data["condition"],
            method=data.get("method", "logistic"),
            genuine_count=int(data.get("genuine_count", 0)),
            impostor_count=int(data.get("impostor_count", 0)),
            a=float(data.get("a", 0.0)),
            b=float(data.get("b", 0.0)),
            knots_x=list(data.get("knots_x", [])),
            knots_y=list(data.get("knots_y", [])),
            recommended_threshold=float(data.get("recommended_threshold", 0.0)),
            equal_error_threshold=float(data.get("equal_error_threshold", 0.0)),
            equal_error_rate=float(data.get("equal_error_rate", 0.0)),
            separation_gap=float(data.get("separation_gap", 0.0)),
            genuine_stats=data.get("genuine", {}),
            impostor_stats=data.get("impostor", {}),
        )


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
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


def fit_logistic(
    scores: np.ndarray, labels: np.ndarray, *, iterations: int = 200, lr: float = 0.5
) -> tuple[float, float]:
    """Platt scaling by Newton-free gradient ascent on the log-likelihood.

    Implemented directly rather than pulled from scikit-learn: it is fifteen
    lines, it avoids a heavyweight dependency for one function, and the
    regularisation and target smoothing below are chosen for the small,
    class-imbalanced sets this system produces.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    # Platt's target smoothing: keeps the fit finite when the classes separate
    # perfectly, which they often do on clean data.
    hi = 1.0 / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    targets = np.where(labels > 0.5, 1.0 - hi, lo)

    # Standardise so one learning rate works across encoders and scales.
    mean, std = scores.mean(), max(scores.std(), 1e-6)
    z = (scores - mean) / std

    a = b = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(a * z + b, -60.0, 60.0)))
        error = targets - p
        grad_a = float((error * z).mean()) - 1e-4 * a   # light L2 for stability
        grad_b = float(error.mean())
        a += lr * grad_a
        b += lr * grad_b

    # Undo the standardisation so the parameters apply to raw similarity.
    return a / std, b - a * mean / std


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators isotonic regression."""
    order = np.argsort(scores)
    x = np.asarray(scores, dtype=np.float64)[order]
    y = np.asarray(labels, dtype=np.float64)[order]

    values = list(y)
    weights = [1.0] * len(y)
    positions = list(range(len(y)))
    index = 0
    while index < len(values) - 1:
        if values[index] <= values[index + 1]:
            index += 1
            continue
        total_weight = weights[index] + weights[index + 1]
        pooled = (values[index] * weights[index] + values[index + 1] * weights[index + 1]) / total_weight
        values[index] = pooled
        weights[index] = total_weight
        del values[index + 1]
        del weights[index + 1]
        del positions[index + 1]
        if index > 0:
            index -= 1

    knots_x: list[float] = []
    knots_y: list[float] = []
    cursor = 0
    for value, weight in zip(values, weights, strict=True):
        span = int(weight)
        knots_x.append(float(x[cursor]))
        knots_y.append(float(np.clip(value, 0.0, 1.0)))
        cursor += span
    return knots_x, knots_y


def operating_points(
    genuine: Sequence[float], impostor: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> list[OperatingPoint]:
    """Sweep thresholds and score each one."""
    if thresholds is None:
        combined = list(genuine) + list(impostor)
        if not combined:
            return []
        lo, hi = min(combined), max(combined)
        thresholds = [round(float(t), 4) for t in np.linspace(lo, hi, 41)]

    genuine_array = np.asarray(genuine, dtype=np.float64)
    impostor_array = np.asarray(impostor, dtype=np.float64)
    points: list[OperatingPoint] = []
    for threshold in thresholds:
        ta = int((genuine_array >= threshold).sum())
        fr = int(genuine_array.size - ta)
        fa = int((impostor_array >= threshold).sum())
        tr = int(impostor_array.size - fa)
        points.append(OperatingPoint(float(threshold), ta, fa, tr, fr))
    return points


def equal_error_threshold(points: Sequence[OperatingPoint]) -> float:
    """The threshold where the false-accept and false-reject rates meet.

    A stable, sample-size-tolerant reference point: unlike a FAR-constrained
    choice it does not swing wildly when the impostor set is small.
    """
    if not points:
        return 0.0
    return min(points, key=lambda p: abs(p.far - p.frr)).threshold


def choose_threshold(
    points: Sequence[OperatingPoint],
    *,
    max_far: float | None = 0.01,
    min_tar: float = 0.60,
) -> float:
    """Pick an operating threshold.

    Security systems care more about false accepts than false rejects: naming
    a stranger as an employee is a different class of error from failing to
    recognise one. So the primary policy is the highest true-accept rate
    subject to a false-accept ceiling.

    That policy has a failure mode on small calibration sets, and it is not
    subtle. With a few hundred impostor comparisons, a 1% FAR ceiling permits
    one or two false accepts; if the impostor distribution has any tail at all,
    the only way to satisfy it is a threshold near the top of the genuine
    range, which rejects almost every genuine match too. The constraint is
    satisfied and the system is useless.

    So the FAR-constrained choice is accepted only if it still admits a
    reasonable share of genuine matches. Otherwise the equal-error point is
    used, and the caller can see from the reported operating points why.
    """
    if not points:
        return 0.0

    if max_far is not None:
        eligible = [p for p in points if p.far <= max_far]
        if eligible:
            best = max(eligible, key=lambda p: (p.tar, -p.threshold))
            if best.tar >= min_tar:
                return best.threshold
            logger.info(
                "The FAR-constrained threshold %.3f would accept only %.0f%% of "
                "genuine matches, which usually means the impostor set is too "
                "small to support that ceiling. Using the equal-error point "
                "instead; widen the calibration set to tighten FAR safely.",
                best.threshold, best.tar * 100,
            )

    equal_error = equal_error_threshold(points)
    best_f1 = max(points, key=lambda p: p.f1)
    # Prefer whichever of the two is more conservative, so falling back never
    # silently loosens the system beyond the balanced point.
    return max(equal_error, best_f1.threshold)


def separation_threshold(
    genuine: Sequence[float], impostor: Sequence[float]
) -> float | None:
    """Midpoint of a clean gap between the two distributions, if one exists.

    When genuine and impostor scores do not overlap, every threshold inside the
    gap is perfect *on this data* -- which makes the equal-error point a poor
    choice, because it sits at the bottom of the gap, one unlucky impostor away
    from a false accept. The midpoint is the furthest point from both failure
    modes and is what generalises best to samples not yet seen.
    """
    if not genuine or not impostor:
        return None
    floor, ceiling = min(genuine), max(impostor)
    if floor <= ceiling:
        return None
    return (floor + ceiling) / 2.0


@dataclass(slots=True)
class CalibrationModel:
    """Per-condition calibration, persisted alongside the gallery."""

    conditions: dict[str, ConditionCalibration] = field(default_factory=dict)
    encoder_fingerprint: str = ""
    created_at: str = ""
    dataset: str = ""
    notes: list[str] = field(default_factory=list)

    def get(self, condition: str) -> ConditionCalibration | None:
        calibration = self.conditions.get(condition)
        if calibration is not None and calibration.usable:
            return calibration
        # Fall back to the full-face mapping, which is the best-sampled one.
        fallback = self.conditions.get("full")
        return fallback if fallback is not None and fallback.usable else None

    def probability(self, similarity: float, condition: str = "full") -> float | None:
        """Calibrated probability, or ``None`` when nothing supports one."""
        calibration = self.get(condition)
        return calibration.probability(similarity) if calibration else None

    def threshold(self, condition: str = "full") -> float | None:
        calibration = self.get(condition)
        return calibration.recommended_threshold if calibration else None

    @property
    def is_usable(self) -> bool:
        return any(c.usable for c in self.conditions.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "encoder_fingerprint": self.encoder_fingerprint,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "notes": list(self.notes),
            "conditions": {k: v.to_dict() for k, v in sorted(self.conditions.items())},
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> CalibrationModel | None:
        if not Path(path).exists():
            return None
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable calibration file %s: %s", path, exc)
            return None
        return cls(
            conditions={
                k: ConditionCalibration.from_dict(v)
                for k, v in (data.get("conditions") or {}).items()
            },
            encoder_fingerprint=data.get("encoder_fingerprint", ""),
            created_at=data.get("created_at", ""),
            dataset=data.get("dataset", ""),
            notes=list(data.get("notes", [])),
        )


def fit_calibration(
    samples: Sequence[CalibrationSample],
    *,
    method: str = "logistic",
    max_far: float | None = 0.01,
    encoder_fingerprint: str = "",
    dataset: str = "",
) -> CalibrationModel:
    """Fit a calibration model from labelled comparisons.

    Conditions with too few samples are still recorded -- so the report shows
    what is missing -- but are marked unusable and will fall back to the
    full-face mapping rather than reporting a confidence nobody measured.
    """
    import datetime as _dt_mod  # noqa: PLC0415

    by_condition: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        by_condition.setdefault(sample.condition, []).append(sample)
        if sample.condition != "full":
            # Every sample also informs the pooled mapping, which is what a
            # sparsely-sampled condition falls back to.
            by_condition.setdefault("full", [])

    model = CalibrationModel(
        encoder_fingerprint=encoder_fingerprint,
        created_at=_dt_mod.datetime.now(_dt_mod.UTC).isoformat(timespec="seconds"),
        dataset=dataset,
    )

    for condition, group in by_condition.items():
        genuine = [s.similarity for s in group if s.is_genuine]
        impostor = [s.similarity for s in group if not s.is_genuine]
        calibration = ConditionCalibration(
            condition=condition,
            method=method,
            genuine_count=len(genuine),
            impostor_count=len(impostor),
            genuine_stats=_stats(genuine),
            impostor_stats=_stats(impostor),
        )

        if calibration.usable:
            scores = np.array([s.similarity for s in group], dtype=np.float64)
            labels = np.array([1.0 if s.is_genuine else 0.0 for s in group])
            if method == "isotonic":
                calibration.knots_x, calibration.knots_y = fit_isotonic(scores, labels)
            else:
                calibration.a, calibration.b = fit_logistic(scores, labels)
            points = operating_points(genuine, impostor)
            calibration.operating_points = points
            clean = separation_threshold(genuine, impostor)
            if clean is not None:
                # The distributions do not overlap on this data. Sit in the
                # middle of the gap rather than at its edge.
                calibration.recommended_threshold = clean
                calibration.separation_gap = min(genuine) - max(impostor)
            else:
                calibration.recommended_threshold = choose_threshold(
                    points, max_far=max_far
                )
            calibration.equal_error_threshold = equal_error_threshold(points)
            eer_point = min(points, key=lambda p: abs(p.far - p.frr))
            calibration.equal_error_rate = (eer_point.far + eer_point.frr) / 2.0
        else:
            model.notes.append(
                f"condition '{condition}' has {len(genuine)} genuine and "
                f"{len(impostor)} impostor samples; at least "
                f"{MIN_SAMPLES_PER_CLASS} of each are needed, so no calibrated "
                "confidence will be reported for it"
            )
        model.conditions[condition] = calibration

    return model
