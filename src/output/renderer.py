"""Annotation / visualisation.

All drawing lives here. The detector, encoder and matcher never touch pixels
for display purposes, so the engine stays usable headless and behind an API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from src.config.schema import DisplayConfig
from src.core.types import DetectionResult, FrameResult, RecognitionStatus

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_DEFAULT_COLOR = (200, 200, 200)


@dataclass(frozen=True, slots=True)
class RenderMetrics:
    """Numbers for the optional performance overlay."""

    fps: float = 0.0
    persons: int = 0
    recognized: int = 0
    unknown: int = 0
    tracks: int = 0
    detector_ms: float = 0.0
    reid_ms: float = 0.0
    total_ms: float = 0.0

    def lines(self) -> list[str]:
        return [
            f"FPS: {self.fps:.1f}",
            f"Persons: {self.persons}",
            f"Recognized: {self.recognized}",
            f"Unknown: {self.unknown}",
            f"Tracks: {self.tracks}",
            f"Det: {self.detector_ms:.1f}ms  ReID: {self.reid_ms:.1f}ms",
        ]


class Renderer:
    """Draws boxes, identity labels and the metrics overlay."""

    def __init__(self, config: DisplayConfig, *, score_label: str = "ReID") -> None:
        self._config = config
        self._score_label = score_label
        self._colors = {
            RecognitionStatus.RECOGNIZED: tuple(config.colors.get("recognized", (60, 200, 60))),
            RecognitionStatus.LOW_CONFIDENCE: tuple(
                config.colors.get("low_confidence", (40, 190, 235))
            ),
            RecognitionStatus.UNKNOWN: tuple(config.colors.get("unknown", (70, 70, 230))),
            RecognitionStatus.REJECTED: tuple(config.colors.get("rejected", (180, 120, 255))),
            RecognitionStatus.PENDING: tuple(config.colors.get("pending", (170, 170, 170))),
            RecognitionStatus.NO_FACE: tuple(config.colors.get("no_face", (150, 150, 150))),
        }

    def color_for(self, status: RecognitionStatus) -> tuple[int, int, int]:
        return self._colors.get(status, _DEFAULT_COLOR)

    # ------------------------------------------------------------------ text
    def label_lines(self, detection: DetectionResult) -> list[str]:
        """Build the label block for one detection.

        Recognised::

            [7] John Doe
            Manager
            ReID: 0.73

        Unknown::

            [7] Unknown
            ReID: 0.42
        """
        recognition = detection.effective
        name = recognition.identity_name or "Unknown"
        head = f"[{detection.track_id}] {name}" if (
            self._config.show_track_id and detection.track_id is not None and detection.track_id >= 0
        ) else name

        lines = [head]
        if self._config.show_title and recognition.identity_title:
            lines.append(recognition.identity_title)
        if recognition.status is RecognitionStatus.NO_FACE and not recognition.identity_id:
            # Say *why* nobody was named, so "no face visible" is never mistaken
            # for "this person is not registered".
            return lines
        if self._config.show_similarity and recognition.status not in (
            RecognitionStatus.PENDING,
            RecognitionStatus.NO_FACE,
        ):
            lines.append(f"{self._score_label}: {recognition.similarity:.2f}")
        elif recognition.status is RecognitionStatus.NO_FACE:
            lines.append("face hidden")
        return lines

    # ------------------------------------------------------------------ draw
    def draw_detection(self, image: np.ndarray, detection: DetectionResult) -> np.ndarray:
        color = self.color_for(detection.recognition_status)
        thickness = self._config.box_thickness
        x1, y1, x2, y2 = detection.bbox.clip(image.shape[1], image.shape[0]).to_int()
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # In face mode, show which face drove the decision -- it makes a wrong
        # match immediately explainable.
        if self._config.show_face_box and detection.face_bbox is not None:
            fx1, fy1, fx2, fy2 = (
                detection.face_bbox.clip(image.shape[1], image.shape[0]).to_int()
            )
            cv2.rectangle(image, (fx1, fy1), (fx2, fy2), color, max(1, thickness - 1))

        self._draw_label_block(image, self.label_lines(detection), (x1, y1), color)
        return image

    def _draw_label_block(
        self,
        image: np.ndarray,
        lines: Sequence[str],
        anchor: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        if not lines:
            return
        scale = self._config.font_scale
        thickness = max(1, int(round(scale * 2)))
        pad = 4
        sizes = [cv2.getTextSize(line, _FONT, scale, thickness)[0] for line in lines]
        line_h = max(size[1] for size in sizes) + 6
        block_w = max(size[0] for size in sizes) + pad * 2
        block_h = line_h * len(lines) + pad

        x, y = anchor
        top = y - block_h
        if top < 0:  # not enough room above the box: draw inside it
            top = y
        left = min(max(0, x), max(0, image.shape[1] - block_w))
        bottom = min(image.shape[0], top + block_h)
        right = min(image.shape[1], left + block_w)

        overlay = image[top:bottom, left:right]
        if overlay.size:
            # Semi-transparent plate keeps text readable over any background.
            cv2.addWeighted(overlay, 0.35, np.full_like(overlay, color), 0.65, 0, overlay)

        text_y = top + line_h - 4
        for line in lines:
            cv2.putText(
                image, line, (left + pad, text_y), _FONT, scale, (255, 255, 255), thickness,
                cv2.LINE_AA,
            )
            text_y += line_h

    def draw_metrics(self, image: np.ndarray, metrics: RenderMetrics) -> np.ndarray:
        if not self._config.show_metrics:
            return image
        scale = self._config.font_scale
        thickness = max(1, int(round(scale * 2)))
        lines = metrics.lines()
        sizes = [cv2.getTextSize(line, _FONT, scale, thickness)[0] for line in lines]
        width = max(size[0] for size in sizes) + 16
        line_h = max(size[1] for size in sizes) + 8
        height = line_h * len(lines) + 8

        panel = image[0 : min(height, image.shape[0]), 0 : min(width, image.shape[1])]
        if panel.size:
            cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, panel)
        y = line_h
        for line in lines:
            cv2.putText(
                image, line, (8, y), _FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
            y += line_h
        return image

    def render(
        self,
        image: np.ndarray,
        result: FrameResult,
        metrics: RenderMetrics | None = None,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """Annotate a frame with every detection and the optional overlay."""
        canvas = image.copy() if copy else image
        for detection in result.detections:
            self.draw_detection(canvas, detection)
        if metrics is not None:
            self.draw_metrics(canvas, metrics)
        return canvas
