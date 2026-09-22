"""YOLO26 person detector built on the Ultralytics API.

YOLO26 is *only* the detector here: it answers "is there a person, and where".
It never answers "who is this" -- that is the ReID encoder plus gallery.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config.schema import AppConfig
from src.core.exceptions import ModelLoadError
from src.core.types import BBox, Detection
from src.detection.detector import Detector, DetectorInfo
from src.utils.device import DeviceInfo
from src.utils.logging import get_logger

logger = get_logger(__name__)

_BUILTIN_TRACKERS = {"botsort.yaml", "bytetrack.yaml"}


class YOLO26Detector(Detector):
    """Ultralytics YOLO26 detector restricted to the ``person`` class.

    Args:
        config: Validated application configuration.
        device: Resolved compute device.
        model_path: Absolute path (or Ultralytics asset name) of the weights.
        tracker_path: Materialised tracker YAML, produced by :func:`build_tracker_config`.
    """

    def __init__(
        self,
        config: AppConfig,
        device: DeviceInfo,
        model_path: str | Path,
        tracker_path: str | Path | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._model_path = str(model_path)
        self._tracker_path = str(tracker_path) if tracker_path else None
        self._model: Any | None = None
        self._info: DetectorInfo | None = None

    # ---------------------------------------------------------------- loading
    def load(self) -> DetectorInfo:
        if self._model is not None and self._info is not None:
            return self._info

        try:
            from ultralytics import YOLO  # noqa: PLC0415 - heavy optional import
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ModelLoadError(
                "the 'ultralytics' package is required for YOLO26 detection: "
                "pip install ultralytics"
            ) from exc

        started = time.perf_counter()
        try:
            model = YOLO(self._model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"cannot load the YOLO26 detector from '{self._model_path}': {exc}. "
                "Check models.detector in the configuration; it must be an existing "
                "weights file or a downloadable Ultralytics asset such as 'yolo26n.pt'."
            ) from exc

        self._model = model
        self._verify_person_class(model)
        elapsed = time.perf_counter() - started

        self._info = DetectorInfo(
            name=Path(self._model_path).stem,
            model_path=self._model_path,
            device=self._device.device,
            backend="ultralytics",
            fp16=self._use_half(),
            imgsz=self._config.detector.imgsz,
            load_time_s=elapsed,
            metadata={
                "task": getattr(model, "task", "detect"),
                "classes": len(getattr(model, "names", {}) or {}),
                "tracker": self._tracker_path,
            },
        )
        logger.info(
            "YOLO26 detector loaded",
            extra={
                "model": self._info.name,
                "device": self._info.device,
                "imgsz": self._info.imgsz,
                "fp16": self._info.fp16,
                "load_ms": round(elapsed * 1000, 1),
            },
        )
        return self._info

    def _verify_person_class(self, model: Any) -> None:
        """Fail early when the weights cannot produce the configured person class."""
        names = getattr(model, "names", None) or {}
        class_id = self._config.detector.person_class_id
        if names and class_id not in names:
            raise ModelLoadError(
                f"detector.person_class_id={class_id} is not present in "
                f"'{self._model_path}' (it exposes {len(names)} classes). "
                "Use a COCO-pretrained YOLO26 model or set the correct class id."
            )
        label = str(names.get(class_id, "")).lower() if names else ""
        if label and label != "person":
            logger.warning(
                "detector.person_class_id=%s maps to '%s', not 'person' -- "
                "ReID results will be meaningless unless this is intentional.",
                class_id,
                label,
            )

    def _use_half(self) -> bool:
        explicit = self._config.detector.half
        return self._device.fp16_enabled if explicit is None else bool(explicit)

    # -------------------------------------------------------------- inference
    def _predict_kwargs(self) -> dict[str, Any]:
        det = self._config.detector
        return {
            "imgsz": det.imgsz,
            "conf": det.confidence,
            "iou": det.iou,
            "classes": [det.person_class_id],
            "max_det": det.max_detections,
            "agnostic_nms": det.agnostic_nms,
            "device": self._device.device,
            # Ultralytics replaced the deprecated `half` flag with `quantize`:
            # 16 selects fp16 compute, None leaves the model at fp32.
            "quantize": 16 if self._use_half() else None,
            "verbose": False,
        }

    def _require_model(self) -> Any:
        if self._model is None:
            self.load()
        assert self._model is not None  # noqa: S101 - load() guarantees this
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        model = self._require_model()
        results = model.predict(source=image, stream=False, **self._predict_kwargs())
        return self._to_detections(results[0]) if results else []

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        model = self._require_model()
        results = model.predict(source=list(images), stream=False, **self._predict_kwargs())
        return [self._to_detections(r) for r in results]

    def track(self, image: np.ndarray, *, persist: bool = True) -> list[Detection]:
        model = self._require_model()
        kwargs = self._predict_kwargs()
        if self._tracker_path:
            kwargs["tracker"] = self._tracker_path
        results = model.track(source=image, stream=False, persist=persist, **kwargs)
        return self._to_detections(results[0]) if results else []

    def reset_tracker(self) -> None:
        model = self._model
        if model is None:
            return
        predictor = getattr(model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()
            logger.debug("Tracker state reset", extra={"trackers": len(trackers)})

    # ---------------------------------------------------------------- helpers
    def _to_detections(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # One synchronising device->host transfer instead of four. Each .cpu()
        # call on a CUDA tensor is a separate synchronisation point, so on a
        # Jetson this alone removes three stalls per frame.
        data = boxes.data
        try:
            host = data.detach().to("cpu", non_blocking=False).numpy()
        except AttributeError:  # pragma: no cover - already a numpy array
            host = np.asarray(data)

        xyxy = host[:, :4]
        raw_ids = boxes.id
        if raw_ids is not None and host.shape[1] >= 7:
            # Ultralytics packs tracked results as [x1,y1,x2,y2,id,conf,cls].
            track_ids = host[:, 4].astype(int)
            confidences = host[:, 5]
            class_ids = host[:, 6].astype(int)
        else:
            track_ids = None
            confidences = host[:, 4]
            class_ids = host[:, 5].astype(int)
        names = getattr(result, "names", {}) or {}

        detections: list[Detection] = []
        for index in range(len(xyxy)):
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    bbox=BBox.from_xyxy(xyxy[index]),
                    confidence=float(confidences[index]),
                    class_id=class_id,
                    class_name=str(names.get(class_id, "person")),
                    track_id=int(track_ids[index]) if track_ids is not None else None,
                )
            )
        return detections

    @property
    def info(self) -> DetectorInfo:
        if self._info is None:
            return self.load()
        return self._info

    def close(self) -> None:
        self._model = None


# --------------------------------------------------------------------------- #
# Tracker configuration materialisation
# --------------------------------------------------------------------------- #


def build_tracker_config(
    config: AppConfig,
    output_dir: Path,
    *,
    reid_model: str | None = None,
) -> Path:
    """Write a tracker YAML that reflects the application's tracking settings.

    Ultralytics reads tracker parameters from a YAML file, so configured values
    (``track_buffer``, ``with_reid``, the appearance model) are merged into a
    copy of the chosen base tracker and written to ``output_dir``.
    """
    base_name = config.tracking.tracker
    base_path = Path(base_name)
    if not base_path.exists():
        if base_path.name not in _BUILTIN_TRACKERS:
            raise ModelLoadError(
                f"tracker configuration '{base_name}' not found and is not one of "
                f"the built-in trackers {sorted(_BUILTIN_TRACKERS)}"
            )
        from ultralytics.utils import ROOT  # noqa: PLC0415

        base_path = Path(ROOT) / "cfg" / "trackers" / base_path.name
        if not base_path.exists():  # pragma: no cover - broken installation
            raise ModelLoadError(f"built-in tracker file missing: {base_path}")

    with base_path.open("r", encoding="utf-8") as handle:
        settings: dict[str, Any] = yaml.safe_load(handle) or {}

    settings["track_buffer"] = config.tracking.track_buffer
    if settings.get("tracker_type") == "botsort":
        settings["with_reid"] = config.tracking.with_reid
        if config.tracking.with_reid:
            # "auto" reuses detector backbone features (cheap); an explicit model
            # path gives stronger appearance cues at extra cost.
            settings["model"] = reid_model or "auto"
    settings.update(config.tracking.overrides)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"tracker_{Path(base_name).stem}.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(settings, handle, sort_keys=False)
    logger.debug("Tracker configuration written", extra={"path": str(target)})
    return target
