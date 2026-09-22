"""INT8 calibration from a folder of representative images.

Only used when ``backend.tensorrt.precision: int8``. Calibration data must come
from the deployment's own cameras: calibrating a face encoder on unrelated
images produces a quantisation that is confidently wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.image import DEFAULT_IMAGE_EXTENSIONS, imread, iter_images, resize_crop
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _base_class() -> Any:  # pragma: no cover - requires TensorRT
    import tensorrt as trt

    return trt.IInt8EntropyCalibrator2


class ImageFolderCalibrator:  # pragma: no cover - requires TensorRT + CUDA
    """Feeds batches of preprocessed images to the TensorRT INT8 calibrator.

    Implemented as a lazy subclass so importing this module never requires
    TensorRT to be installed.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import tensorrt as trt

        namespace = {
            "__init__": _init,
            "get_batch_size": _get_batch_size,
            "get_batch": _get_batch,
            "read_calibration_cache": _read_cache,
            "write_calibration_cache": _write_cache,
        }
        concrete = type("_ImageFolderCalibrator", (trt.IInt8EntropyCalibrator2,), namespace)
        instance = concrete.__new__(concrete)
        instance.__init__(*args, **kwargs)
        return instance


def _init(self, directory: Path, chw: tuple[int, int, int], *,
          batch_size: int = 4, cache_path: Path | None = None) -> None:  # pragma: no cover
    import pycuda.driver as cuda
    import tensorrt as trt

    trt.IInt8EntropyCalibrator2.__init__(self)
    self.directory = Path(directory)
    self.chw = chw
    self.batch_size = max(1, batch_size)
    self.cache_path = cache_path or self.directory / "int8_calibration.cache"
    self.files = iter_images(self.directory, DEFAULT_IMAGE_EXTENSIONS, recursive=True)
    if not self.files:
        raise ValueError(
            f"no calibration images found in {self.directory}; INT8 calibration "
            "needs representative crops from the deployment's own cameras"
        )
    logger.info(
        "INT8 calibration set", extra={"images": len(self.files), "batch": self.batch_size}
    )
    self.index = 0
    channels, height, width = chw
    self.device_input = cuda.mem_alloc(
        self.batch_size * channels * height * width * np.float32().itemsize
    )


def _get_batch_size(self) -> int:  # pragma: no cover
    return self.batch_size


def _get_batch(self, names: Any = None, p_str: Any = None) -> list[int] | None:  # noqa: ARG001 - TensorRT calibrator API
    import pycuda.driver as cuda

    if self.index + self.batch_size > len(self.files):
        return None
    channels, height, width = self.chw
    batch = np.empty((self.batch_size, height, width, channels), dtype=np.float32)
    for slot in range(self.batch_size):
        image = imread(self.files[self.index + slot])
        batch[slot] = resize_crop(image, (height, width))[:, :, ::-1]
    self.index += self.batch_size
    tensor = np.ascontiguousarray(
        ((batch - 127.5) / 127.5).transpose(0, 3, 1, 2), dtype=np.float32
    )
    cuda.memcpy_htod(self.device_input, tensor)
    return [int(self.device_input)]


def _read_cache(self) -> bytes | None:  # pragma: no cover
    return self.cache_path.read_bytes() if self.cache_path.exists() else None


def _write_cache(self, cache: bytes) -> None:  # pragma: no cover
    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
    self.cache_path.write_bytes(cache)
