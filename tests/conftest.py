"""Shared fixtures and test doubles.

Model inference is mocked so the unit suite runs on any machine in seconds and
without a GPU or network. The real models are exercised separately by the
``slow`` integration tests.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.loader import config_from_dict
from src.config.paths import ProjectPaths
from src.config.schema import AppConfig
from src.core.types import BBox, Detection
from src.detection.detector import Detector, DetectorInfo
from src.face.detector import FaceDetector
from src.face.types import FaceDetection, FaceDetectorInfo
from src.reid.encoder import EncoderInfo, ReIDEncoder, l2_normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
EMBED_DIM = 32


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeDetector(Detector):
    """Returns a scripted list of detections, one script entry per call."""

    def __init__(self, script: Sequence[Sequence[Detection]] | None = None) -> None:
        self.script = [list(frame) for frame in (script or [])]
        self.calls = 0
        self.track_calls = 0
        self.resets = 0
        self.loaded = False
        self._info = DetectorInfo(
            name="fake-detector",
            model_path="fake://detector",
            device="cpu",
            backend="fake",
            fp16=False,
            imgsz=640,
        )

    def _next(self) -> list[Detection]:
        if not self.script:
            return []
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        # Fresh Detection objects: the pipeline mutates track_id in place.
        return [
            Detection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                track_id=d.track_id,
            )
            for d in self.script[index]
        ]

    def load(self) -> DetectorInfo:
        self.loaded = True
        return self._info

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self._next()

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        return [self._next() for _ in images]

    def track(self, image: np.ndarray, *, persist: bool = True) -> list[Detection]:
        self.track_calls += 1
        return self._next()

    def reset_tracker(self) -> None:
        self.resets += 1

    @property
    def info(self) -> DetectorInfo:
        return self._info


class FakeEncoder(ReIDEncoder):
    """Deterministic embeddings derived from a crop's mean colour.

    Crops of a similar colour land close together on the unit sphere, so
    thresholds, matching and stabilization can be tested with meaningful,
    reproducible similarity values and no model.
    """

    def __init__(self, dimension: int = EMBED_DIM) -> None:
        self._dimension = dimension
        self.calls = 0
        self.crops_seen = 0
        self._info = EncoderInfo(
            name="fake-encoder",
            model_path="fake://reid",
            backend="fake",
            device="cpu",
            fp16=False,
            input_size=(256, 128),
            embedding_dimension=dimension,
        )

    def load(self) -> EncoderInfo:
        return self._info

    def vector_for_color(self, color: Sequence[float]) -> np.ndarray:
        """The embedding this encoder would produce for a solid colour."""
        rng = np.random.default_rng(1234)
        basis = rng.normal(size=(3, self._dimension)).astype(np.float32)
        return l2_normalize(np.asarray(color, dtype=np.float32) / 255.0 @ basis)

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(crops), self._dimension), dtype=np.float32)
        for index, crop in enumerate(crops):
            if crop is None or not getattr(crop, "size", 0):
                continue
            self.crops_seen += 1
            out[index] = self.vector_for_color(np.asarray(crop).reshape(-1, 3).mean(axis=0))
        return out

    @property
    def info(self) -> EncoderInfo:
        return self._info


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #


def solid_image(
    width: int = 320,
    height: int = 480,
    color: tuple[int, int, int] = (60, 120, 200),
) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def person_scene(
    color: tuple[int, int, int] = (60, 120, 200),
    *,
    size: tuple[int, int] = (640, 480),
    box: tuple[int, int, int, int] = (200, 80, 320, 440),
) -> np.ndarray:
    """A grey frame with one solid rectangle standing in for a person."""
    width, height = size
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = color
    return image


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int = 10) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened(), f"cannot open a writer for {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def base_config_dict(tmp_path: Path) -> dict:
    """A minimal but complete configuration rooted in a temporary directory."""
    return {
        "application": {"log_level": "WARNING"},
        # Pinned explicitly: unit tests must not silently follow the shipped
        # default mode, and the fakes model whole-body appearance.
        "recognition": {"mode": "person_reid"},
        "models": {"detector": "fake://detector", "reid": "fake://reid"},
        "matching": {"recognition_threshold": 0.70, "high_confidence_threshold": 0.85},
        "gallery": {"directory": str(tmp_path / "gallery")},
        "output": {"directory": str(tmp_path / "output"), "save_snapshots": False},
        "display": {"show_window": False},
        "people": [],
    }


@pytest.fixture
def config(base_config_dict: dict, tmp_path: Path) -> AppConfig:
    return config_from_dict(base_config_dict, base_dir=tmp_path)


@pytest.fixture
def paths(config: AppConfig) -> ProjectPaths:
    project_paths = ProjectPaths.from_config(config)
    project_paths.ensure(
        project_paths.gallery_dir,
        project_paths.embeddings_dir,
        project_paths.metadata_dir,
        project_paths.output_dir,
    )
    return project_paths


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def person_bbox() -> BBox:
    return BBox(200.0, 80.0, 320.0, 440.0)


@pytest.fixture
def demo_available() -> bool:
    return (DEMO_DIR / "persons" / "person_a.jpg").exists()


def requires_demo():
    return pytest.mark.skipif(
        not (DEMO_DIR / "persons" / "person_a.jpg").exists(),
        reason="demo dataset missing; run: python scripts/build_demo.py",
    )


def requires_models():
    return pytest.mark.skipif(
        not (PROJECT_ROOT / "models" / "yolo26n.pt").exists()
        or not (PROJECT_ROOT / "models" / "yolo26n-reid.onnx").exists(),
        reason="model weights missing; see README 'Model setup'",
    )


class FakeFaceDetector(FaceDetector):
    """Returns scripted faces, so the face pipeline is testable without a model."""

    def __init__(self, faces: Sequence[FaceDetection] | None = None) -> None:
        self.faces = list(faces or [])
        self.calls = 0
        self._info = FaceDetectorInfo(
            name="fake-face-detector",
            model_path="fake://yunet",
            backend="fake",
            device="cpu",
            input_size=(320, 320),
        )

    def load(self) -> FaceDetectorInfo:
        return self._info

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        self.calls += 1
        return list(self.faces)

    @property
    def info(self) -> FaceDetectorInfo:
        return self._info


def make_face(
    box: tuple[float, float, float, float] = (240.0, 100.0, 300.0, 175.0),
    score: float = 0.95,
    *,
    landmarks: bool = True,
) -> FaceDetection:
    """A face detection whose landmarks are a real similarity transform of the
    ArcFace template.

    Generating them any other way would make alignment mathematically
    impossible to satisfy, and the resulting test would assert something the
    algorithm is not supposed to achieve.
    """
    from src.face.align import ARCFACE_TEMPLATE_112

    x1, y1, x2, y2 = box
    points = None
    if landmarks:
        scale = (x2 - x1) / 112.0
        points = (ARCFACE_TEMPLATE_112 * scale + np.array([x1, y1], np.float32)).astype(
            np.float32
        )
    return FaceDetection(bbox=BBox(x1, y1, x2, y2), score=score, landmarks=points)


@pytest.fixture
def face_config_dict(tmp_path: Path) -> dict:
    """Face-mode configuration pointing at fake models."""
    return {
        "application": {"log_level": "WARNING"},
        "recognition": {"mode": "face"},
        "models": {"detector": "fake://detector", "reid": "fake://reid"},
        "face": {
            "detector_model": "fake://yunet",
            "recognition_model": "fake://sface",
            "min_face_size": 40,
            "min_eye_distance": 10.0,
            "identity_hold_frames": 5,
        },
        "gallery": {"directory": str(tmp_path / "gallery")},
        "output": {"directory": str(tmp_path / "output"), "save_snapshots": False},
        "display": {"show_window": False},
        "people": [],
    }


@pytest.fixture
def fake_face_detector() -> FakeFaceDetector:
    return FakeFaceDetector([make_face()])


def requires_face_models():
    models = PROJECT_ROOT / "models"
    missing = not (models / "face_detection_yunet_2023mar.onnx").exists() or not (
        (models / "w600k_r50.onnx").exists()
        or (models / "face_recognition_sface_2021dec.onnx").exists()
    )
    return pytest.mark.skipif(
        missing,
        reason="face models missing; run: python scripts/fetch_face_models.py",
    )


_EXIT_STATUS: list[int] = []


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Remember the real exit status for :func:`pytest_unconfigure`."""
    _EXIT_STATUS.append(int(exitstatus))


def pytest_unconfigure(config):  # noqa: ARG001
    """Exit without unwinding the native libraries' static destructors.

    A single pytest process builds several engines, each holding torch, OpenCV
    DNN and ONNX Runtime handles. At interpreter shutdown those libraries tear
    down their own global state in an order none of them agrees on, which
    intermittently aborts with "recursive_mutex lock failed" *after* every test
    has already passed -- turning a green run into exit code 134.

    The application itself is unaffected: the CLI exits 0 consistently, and this
    is purely an artefact of loading that many native models into one process.
    This hook runs after pytest has written its report, so flushing the streams
    and exiting directly preserves both the summary and the real status while
    skipping the teardown that races.

    Set ``REID_TESTS_NO_FAST_EXIT=1`` to disable it (for coverage tooling, which
    needs normal interpreter shutdown to write its data file).
    """
    if os.environ.get("REID_TESTS_NO_FAST_EXIT"):
        return
    status = _EXIT_STATUS[-1] if _EXIT_STATUS else 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
