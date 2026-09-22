"""Input sources: webcam, video file, image, directory, network stream."""

from src.sources.base import BaseSource
from src.sources.directory import ImageDirectorySource
from src.sources.factory import build_source, infer_kind
from src.sources.image import ImageSource
from src.sources.stream import NetworkStreamSource
from src.sources.video import VideoFileSource
from src.sources.webcam import WebcamSource

__all__ = [
    "BaseSource",
    "WebcamSource",
    "VideoFileSource",
    "ImageSource",
    "ImageDirectorySource",
    "NetworkStreamSource",
    "build_source",
    "infer_kind",
]
