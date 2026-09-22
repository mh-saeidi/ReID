"""TensorRT engine execution.

A thin, explicit wrapper: load a serialized engine, allocate device buffers
once, and run batches through it. Deliberately not a general-purpose framework
-- the models here have a single input and a single output, and keeping the
code that shape makes the CUDA memory handling auditable.

Buffers are allocated once at the maximum batch size and reused, so a steady
state involves no allocations at all. Host-side staging buffers are pinned when
CUDA can provide them, because pageable-memory transfers force an extra copy
inside the driver.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.backends.tensorrt_builder import TensorRTUnavailable
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TensorRTSession:
    """Runs one TensorRT engine on one CUDA stream.

    Not thread-safe by construction; an internal lock serialises calls so the
    asynchronous pipeline can share a session between workers without
    corrupting the execution context.
    """

    def __init__(self, engine_path: Path, *, max_batch_size: int = 8,
                 device_index: int = 0) -> None:
        self._engine_path = Path(engine_path)
        self._max_batch = max(1, max_batch_size)
        self._device_index = device_index
        self._lock = threading.Lock()

        self._trt: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._cuda: Any = None

        self._input_name = ""
        self._output_name = ""
        self._input_shape: tuple[int, ...] = ()
        self._output_dim = 0
        self._device_input: Any = None
        self._device_output: Any = None
        self._host_output: np.ndarray | None = None
        self._loaded = False

    # --------------------------------------------------------------- loading
    def load(self) -> None:  # pragma: no cover - requires CUDA + TensorRT
        if self._loaded:
            return
        try:
            import tensorrt as trt  # noqa: PLC0415
        except ImportError as exc:
            raise TensorRTUnavailable(
                "the tensorrt Python bindings are required to run an engine"
            ) from exc
        try:
            import pycuda.autoinit  # noqa: F401, PLC0415 - initialises the context
            import pycuda.driver as cuda  # noqa: PLC0415
        except ImportError as exc:
            raise TensorRTUnavailable(
                "pycuda is required to run TensorRT engines: pip install pycuda"
            ) from exc

        if not self._engine_path.exists():
            raise TensorRTUnavailable(f"engine not found: {self._engine_path}")

        self._trt, self._cuda = trt, cuda
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(self._engine_path.read_bytes())
        if self._engine is None:
            raise TensorRTUnavailable(
                f"could not deserialize {self._engine_path.name}. Engines are "
                "not portable between GPUs or TensorRT versions; rebuild with "
                "'main.py models build-tensorrt --force'."
            )
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._discover_bindings()
        self._allocate()
        self._loaded = True
        logger.info(
            "TensorRT engine loaded",
            extra={
                "engine": self._engine_path.name,
                "input": f"{self._input_name}{self._input_shape}",
                "output_dim": self._output_dim,
                "max_batch": self._max_batch,
            },
        )

    def _discover_bindings(self) -> None:  # pragma: no cover - requires TRT
        engine, trt = self._engine, self._trt
        # TensorRT 10 replaced the binding-index API with named tensors.
        if hasattr(engine, "num_io_tensors"):
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                mode = engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._input_name = name
                    self._input_shape = tuple(engine.get_tensor_shape(name))
                else:
                    self._output_name = name
                    shape = tuple(engine.get_tensor_shape(name))
                    self._output_dim = int(np.prod([d for d in shape[1:] if d > 0]))
        else:
            for index in range(engine.num_bindings):
                name = engine.get_binding_name(index)
                shape = tuple(engine.get_binding_shape(index))
                if engine.binding_is_input(index):
                    self._input_name, self._input_shape = name, shape
                else:
                    self._output_name = name
                    self._output_dim = int(np.prod([d for d in shape[1:] if d > 0]))
        if not self._input_name or not self._output_name:
            raise TensorRTUnavailable(
                f"{self._engine_path.name} does not expose one input and one output"
            )

    def _allocate(self) -> None:  # pragma: no cover - requires CUDA
        cuda = self._cuda
        channels, height, width = (int(d) for d in self._input_shape[-3:])
        input_elements = self._max_batch * channels * height * width
        output_elements = self._max_batch * self._output_dim

        self._device_input = cuda.mem_alloc(input_elements * np.float32().itemsize)
        self._device_output = cuda.mem_alloc(output_elements * np.float32().itemsize)
        # Pinned host memory: the driver can DMA straight out of it.
        self._host_output = cuda.pagelocked_empty(
            (self._max_batch, self._output_dim), dtype=np.float32
        )
        self._chw = (channels, height, width)

    # ------------------------------------------------------------- inference
    @property
    def input_chw(self) -> tuple[int, int, int]:
        return self._chw if self._loaded else (0, 0, 0)

    @property
    def output_dimension(self) -> int:
        return self._output_dim

    @property
    def max_batch_size(self) -> int:
        return self._max_batch

    def infer(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover - requires CUDA
        """Run one NCHW float32 batch, returning ``(N, D)``."""
        if not self._loaded:
            self.load()
        if batch.ndim != 4:
            raise ValueError(f"expected an NCHW batch, got shape {batch.shape}")
        count = batch.shape[0]
        if count > self._max_batch:
            raise ValueError(
                f"batch of {count} exceeds the engine's max batch {self._max_batch}; "
                "raise backend.tensorrt.max_batch_size and rebuild the engine"
            )

        contiguous = np.ascontiguousarray(batch, dtype=np.float32)
        cuda = self._cuda
        with self._lock:
            if hasattr(self._context, "set_input_shape"):
                self._context.set_input_shape(self._input_name, contiguous.shape)
            cuda.memcpy_htod_async(self._device_input, contiguous, self._stream)

            if hasattr(self._context, "set_tensor_address"):
                self._context.set_tensor_address(self._input_name, int(self._device_input))
                self._context.set_tensor_address(self._output_name, int(self._device_output))
                self._context.execute_async_v3(stream_handle=self._stream.handle)
            else:
                self._context.execute_async_v2(
                    bindings=[int(self._device_input), int(self._device_output)],
                    stream_handle=self._stream.handle,
                )

            view = self._host_output[:count]
            cuda.memcpy_dtoh_async(view, self._device_output, self._stream)
            self._stream.synchronize()
            # Copy out of the reused pinned buffer before the next call overwrites it.
            return np.array(view, dtype=np.float32, copy=True)

    def close(self) -> None:
        with self._lock:
            self._context = None
            self._engine = None
            self._device_input = None
            self._device_output = None
            self._host_output = None
            self._stream = None
            self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded
