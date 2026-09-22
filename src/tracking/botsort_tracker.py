"""BoT-SORT tracking via Ultralytics.

Ultralytics runs association inside the model call, so this adapter delegates
to :meth:`Detector.track` rather than re-implementing association. Keeping the
:class:`Tracker` seam means a standalone tracker (an external BoT-SORT, OC-SORT
or a custom Kalman tracker) can be substituted without touching the pipeline.

BoT-SORT's own ``with_reid`` appearance cue helps it survive occlusions; it is
*association* appearance, not identity. Identity still comes from matching the
crop embedding against the registered gallery.
"""

from __future__ import annotations

import numpy as np

from src.core.types import Detection
from src.detection.detector import Detector
from src.tracking.tracker import Tracker


class UltralyticsTracker(Tracker):
    """Adapter around the detector's built-in tracking mode."""

    def __init__(self, detector: Detector, *, tracker_name: str = "botsort", persist: bool = True):
        self._detector = detector
        self._name = tracker_name
        self._persist = persist

    def update(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Run detection+association in one pass.

        ``detections`` is ignored: Ultralytics couples detection and association,
        and re-running the detector separately would double the cost.
        """
        return self._detector.track(image, persist=self._persist)

    def reset(self) -> None:
        self._detector.reset_tracker()

    @property
    def name(self) -> str:
        return self._name


class NullTracker(Tracker):
    """Pass-through used for still images and when tracking is disabled."""

    def update(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        return detections

    def reset(self) -> None:
        return None

    @property
    def name(self) -> str:
        return "none"
