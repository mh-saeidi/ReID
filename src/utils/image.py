"""Image I/O, cropping, quality measurement and filename helpers.

Centralised so no component hard-codes extensions, encoding parameters or
overwrite behaviour.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.exceptions import OutputError, SourceError
from src.core.types import BBox

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def imread(path: str | Path) -> np.ndarray:
    """Read an image as BGR, raising an actionable error on failure.

    ``cv2.imdecode`` on raw bytes is used so non-ASCII paths work on every OS.
    """
    path = Path(path)
    if not path.exists():
        raise SourceError(f"image not found: {path}")
    if not path.is_file():
        raise SourceError(f"not a file: {path}")
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise SourceError(f"cannot read {path}: {exc}") from exc
    if buffer.size == 0:
        raise SourceError(f"image file is empty: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise SourceError(
            f"cannot decode image {path}: unsupported or corrupt file "
            f"(supported: {', '.join(DEFAULT_IMAGE_EXTENSIONS)})"
        )
    return image


def imwrite(path: str | Path, image: np.ndarray, *, jpeg_quality: int = 92) -> Path:
    """Write an image, creating parent directories and reporting clear errors."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"cannot create directory {path.parent}: {exc}") from exc

    params: list[int] = []
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)]

    ok, encoded = cv2.imencode(suffix or ".jpg", image, params)
    if not ok:
        raise OutputError(f"cannot encode image for {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise OutputError(
            f"cannot write {path}: {exc}. Check output.directory and filesystem permissions."
        ) from exc
    return path


def iter_images(
    directory: str | Path,
    extensions: tuple[str, ...] | list[str] = DEFAULT_IMAGE_EXTENSIONS,
    *,
    recursive: bool = False,
) -> list[Path]:
    """List image files in a directory, sorted for reproducible processing order."""
    directory = Path(directory)
    if not directory.exists():
        raise SourceError(f"directory not found: {directory}")
    if not directory.is_dir():
        raise SourceError(f"not a directory: {directory}")
    allowed = {e.lower() for e in extensions}
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in allowed
    )


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #


def timestamp_slug(moment: _dt.datetime | None = None, *, with_micros: bool = False) -> str:
    moment = moment or _dt.datetime.now()
    return moment.strftime("%Y%m%d_%H%M%S_%f" if with_micros else "%Y%m%d_%H%M%S")


def unique_path(path: str | Path, *, overwrite: bool = False) -> Path:
    """Return a non-colliding path by appending ``_1``, ``_2``, ... when needed.

    Output files are never silently overwritten unless the caller opts in.
    """
    path = Path(path)
    if overwrite or not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for index in range(1, 10_000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise OutputError(f"cannot find a free filename for {path} after 10000 attempts")


def sanitize_filename(value: str, *, fallback: str = "unnamed") -> str:
    """Make an arbitrary label safe to embed in a filename."""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value.strip())
    cleaned = cleaned.strip("._") or fallback
    return cleaned[:96]


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #


def crop_bbox(
    image: np.ndarray,
    bbox: BBox,
    *,
    padding: float = 0.0,
    min_size: int = 1,
) -> np.ndarray | None:
    """Crop a person box with optional fractional padding.

    Returns ``None`` when the resulting region is degenerate, so callers can
    skip the detection instead of feeding garbage into the encoder.
    """
    height, width = image.shape[:2]
    if padding > 0.0:
        pad_x = bbox.width * padding
        pad_y = bbox.height * padding
        bbox = BBox(bbox.x1 - pad_x, bbox.y1 - pad_y, bbox.x2 + pad_x, bbox.y2 + pad_y)
    x1, y1, x2, y2 = bbox.clip(width, height).to_int()
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def resize_crop(crop: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize a crop to (height, width) using area/linear interpolation."""
    target_h, target_w = size_hw
    h, w = crop.shape[:2]
    interpolation = cv2.INTER_AREA if (h > target_h or w > target_w) else cv2.INTER_LINEAR
    return cv2.resize(crop, (target_w, target_h), interpolation=interpolation)


def letterbox(
    image: np.ndarray, size_hw: tuple[int, int], color: tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """Aspect-preserving resize with padding (keeps a person's body proportions)."""
    target_h, target_w = size_hw
    h, w = image.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = resize_crop(image, (new_h, new_w))
    canvas = np.full((target_h, target_w, image.shape[2]), color, dtype=image.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


# --------------------------------------------------------------------------- #
# Quality metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ImageQuality:
    """Cheap, explainable quality signals for a reference crop."""

    blur_variance: float
    brightness: float
    contrast: float
    width: int
    height: int

    def to_dict(self) -> dict[str, float]:
        return {
            "blur_variance": round(self.blur_variance, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "width": self.width,
            "height": self.height,
        }


def measure_quality(image: np.ndarray) -> ImageQuality:
    """Variance-of-Laplacian sharpness plus mean/std intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return ImageQuality(
        blur_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
    )
