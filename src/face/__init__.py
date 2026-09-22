"""Face detection, alignment and recognition.

Used when ``recognition.mode: face``, which makes identity depend on the face
alone and therefore independent of what the person is wearing.
"""

from src.face.align import ARCFACE_TEMPLATE_112, align_face, estimate_transform
from src.face.detector import FaceDetector, YuNetFaceDetector
from src.face.embedder import ArcFaceOnnxEmbedder, SFaceEmbedder
from src.face.factory import build_face_detector, build_face_encoder, select_backend
from src.face.preprocess import FaceCropPreprocessor
from src.face.types import FaceDetection, FaceDetectorInfo, FaceQuality

__all__ = [
    "FaceDetector",
    "YuNetFaceDetector",
    "FaceDetection",
    "FaceDetectorInfo",
    "FaceQuality",
    "SFaceEmbedder",
    "ArcFaceOnnxEmbedder",
    "FaceCropPreprocessor",
    "align_face",
    "estimate_transform",
    "ARCFACE_TEMPLATE_112",
    "build_face_detector",
    "build_face_encoder",
    "select_backend",
]
