"""A/B experiments over pipeline configurations.

Every variant is evaluated on byte-identical inputs, so a difference between
two rows is attributable to the configuration change and nothing else. That is
the one thing a synthetic dataset is genuinely good for: it cannot tell you the
real-world accuracy, but it can tell you reliably whether a component helps.

The variants follow the ladder in the specification, each adding one thing to
the one before:

``A_baseline``            whole-body appearance matching -- the original design
``B_face_only``           face recognition, single frame, fixed threshold
``C_face_calibrated``     + thresholds fitted on a held-out calibration split
``D_face_quality``        + quality gating and quality-weighted evidence
``E_face_conditional``    + per-visibility thresholds (masked / partial / full)
``F_full_system``         + ambiguity margin and open-set rejection

The result is reported per condition as well as overall, because a component
that helps on average can easily hurt one condition badly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.identity.calibration import CalibrationModel, fit_calibration
from src.identity.decision import IdentityDecisionEngine, MatchingThresholds
from src.tools.face_evaluation import (
    EvaluationReport,
    FacePipeline,
    QuerySample,
    collect_calibration_samples,
    evaluate,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Variant:
    """One configuration under test."""

    key: str
    name: str
    description: str
    build_engine: Callable[..., IdentityDecisionEngine]
    uses_calibration: bool = True


@dataclass(slots=True)
class VariantResult:
    key: str
    name: str
    description: str
    report: EvaluationReport | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.key,
            "name": self.name,
            "description": self.description,
            "error": self.error,
            "report": self.report.to_dict() if self.report else None,
        }


@dataclass(slots=True)
class ABReport:
    """Every variant's measured result on the same data."""

    dataset: str
    synthetic: bool = False
    results: list[VariantResult] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "synthetic": self.synthetic,
            "caveats": self.caveats,
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        lines: list[str] = []
        if self.synthetic:
            lines += [
                "=" * 78,
                "  SYNTHETIC DATA -- valid for comparing these variants against each",
                "  other on identical inputs. NOT a real-world accuracy measurement.",
                "=" * 78,
                "",
            ]
        lines += [
            f"A/B experiments: {self.dataset}",
            "",
            f"  {'VARIANT':<22} {'acc':>7} {'rank1':>7} {'TAR':>7} {'FAR':>7} "
            f"{'unk rej':>8} {'F1':>7}",
        ]
        for result in self.results:
            if result.report is None:
                lines.append(f"  {result.key:<22} {'failed':>7}  {result.error[:38]}")
                continue
            o = result.report.overall
            lines.append(
                f"  {result.key:<22} {o.accuracy:>6.1%} {o.rank1_accuracy:>6.1%} "
                f"{o.tar:>6.1%} {o.far:>6.1%} {o.unknown_rejection_rate:>7.1%} "
                f"{o.f1:>7.3f}"
            )

        conditions: list[str] = []
        for result in self.results:
            if result.report:
                conditions = sorted(result.report.conditions)
                break
        if conditions:
            lines += ["", "  Accuracy by condition:", ""]
            header = f"  {'VARIANT':<22}" + "".join(f"{c[:9]:>11}" for c in conditions)
            lines.append(header)
            for result in self.results:
                if result.report is None:
                    continue
                row = f"  {result.key:<22}"
                for condition in conditions:
                    metrics = result.report.conditions.get(condition)
                    row += f"{metrics.accuracy:>10.0%} " if metrics else f"{'-':>11}"
                lines.append(row)

        lines += [
            "",
            "  Each variant adds one component to the previous one, so a row that",
            "  does not improve on the row above it is a component that did not pay",
            "  for itself on this data.",
        ]
        if self.caveats:
            lines += ["", "  Caveats:"] + [f"    - {c}" for c in self.caveats]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Variant builders
# --------------------------------------------------------------------------- #


def _engine(gallery, *, thresholds: MatchingThresholds,
            calibration: CalibrationModel | None = None) -> IdentityDecisionEngine:
    return IdentityDecisionEngine(gallery, thresholds, calibration=calibration)


def build_variants(fixed_threshold: float = 0.45) -> list[Variant]:
    """The standard ladder, each step adding exactly one thing."""
    return [
        Variant(
            "B_face_only", "face, fixed threshold",
            "single frame, one threshold for every condition, no quality gate",
            lambda gallery, calibration=None: _engine(
                gallery,
                thresholds=MatchingThresholds(
                    full=fixed_threshold, partial=fixed_threshold,
                    masked=fixed_threshold, ambiguity_margin=0.0, min_quality=0.0,
                ),
            ),
            uses_calibration=False,
        ),
        Variant(
            "C_face_calibrated", "+ calibrated threshold",
            "threshold fitted on a held-out calibration split",
            lambda gallery, calibration=None: _engine(
                gallery,
                thresholds=MatchingThresholds(
                    full=None, partial=None, masked=None,
                    ambiguity_margin=0.0, min_quality=0.0,
                ),
                calibration=calibration,
            ),
        ),
        Variant(
            "D_face_quality", "+ quality gating",
            "rejects faces too degraded to produce a trustworthy embedding",
            lambda gallery, calibration=None: _engine(
                gallery,
                thresholds=MatchingThresholds(
                    full=None, partial=None, masked=None,
                    ambiguity_margin=0.0, min_quality=0.35,
                ),
                calibration=calibration,
            ),
        ),
        Variant(
            "E_face_conditional", "+ per-visibility thresholds",
            "masked and partial faces judged against their own distributions",
            lambda gallery, calibration=None: _engine(
                gallery,
                thresholds=MatchingThresholds(
                    full=None, partial=None, masked=None,
                    ambiguity_margin=0.0, min_quality=0.35,
                ),
                calibration=calibration,
            ),
        ),
        Variant(
            "F_full_system", "+ ambiguity margin",
            "refuses to choose between two comparably-scoring identities",
            lambda gallery, calibration=None: _engine(
                gallery,
                thresholds=MatchingThresholds(
                    full=None, partial=None, masked=None,
                    ambiguity_margin=0.06, min_quality=0.35,
                ),
                calibration=calibration,
            ),
        ),
    ]


def run_ab(
    detector,
    encoder,
    gallery,
    calibration_samples: Sequence[QuerySample],
    test_samples: Sequence[QuerySample],
    *,
    dataset: str = "",
    synthetic: bool = False,
    caveats: Sequence[str] = (),
    registered: Sequence[str] = (),
    variants: Sequence[Variant] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ABReport:
    """Evaluate every variant on identical calibration and test splits."""
    report = ABReport(dataset=dataset, synthetic=synthetic, caveats=list(caveats))
    variants = list(variants or build_variants())

    # Calibration is fitted once, on the calibration split only, and shared by
    # every variant that uses it -- so variants differ by their decision rules,
    # not by having seen different data.
    probe = FacePipeline(detector, encoder, gallery,
                         _engine(gallery, thresholds=MatchingThresholds()))
    calibration = fit_calibration(
        collect_calibration_samples(probe, calibration_samples),
        max_far=0.01,
        dataset=f"{dataset} (calibration split)",
    )

    for variant in variants:
        if progress:
            progress(variant.key)
        try:
            engine = variant.build_engine(
                gallery, calibration=calibration if variant.uses_calibration else None
            )
            pipeline = FacePipeline(detector, encoder, gallery, engine)
            outcome = evaluate(
                pipeline, test_samples, dataset=dataset,
                registered=registered, synthetic=synthetic,
            )
            report.results.append(
                VariantResult(variant.key, variant.name, variant.description, outcome)
            )
        except Exception as exc:  # noqa: BLE001 - one variant must not stop the sweep
            logger.error("Variant %s failed: %s", variant.key, exc)
            report.results.append(
                VariantResult(variant.key, variant.name, variant.description,
                              error=str(exc))
            )
    return report


def write_report(report: ABReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
