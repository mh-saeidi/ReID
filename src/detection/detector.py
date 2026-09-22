"""Detector interface.

The pipeline depends on this abstraction only, so the underlying model family
or inference backend can change without touching anything downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.types import Detection


@dataclass(frozen=True, slots=True)
class DetectorInfo:
    """What was actually loaded -- logged at startup and reported by the API."""

    name: str
    model_path: str
    device: str
    backend: str
    fp16: bool
    imgsz: int
    load_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Detector(ABC):
    """Detects people in a frame, optionally with tracker-assigned ids."""

    @abstractmethod
    def load(self) -> DetectorInfo:
        """Load weights and warm the model up. Idempotent."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Stateless detection on a single BGR frame."""

    @abstractmethod
    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Stateless detection on a batch of BGR frames."""

    @abstractmethod
    def track(self, image: np.ndarray, *, persist: bool = True) -> list[Detection]:
        """Stateful detection: returns detections carrying ``track_id``.

        A track id is a *temporary, source-local* handle for temporal continuity.
        It is never an identity -- identity comes from the ReID gallery match.
        """

    @abstractmethod
    def reset_tracker(self) -> None:
        """Drop tracker state (call when switching input source)."""

    @property
    @abstractmethod
    def info(self) -> DetectorInfo:
        """Information about the loaded model."""

    def close(self) -> None:
        """Release resources. Safe to call multiple times."""

    def __enter__(self) -> Detector:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
