"""Person detection (YOLO26)."""

from src.detection.detector import Detector, DetectorInfo
from src.detection.yolo26_detector import YOLO26Detector, build_tracker_config

__all__ = ["Detector", "DetectorInfo", "YOLO26Detector", "build_tracker_config"]
