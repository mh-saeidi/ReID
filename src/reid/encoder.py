"""ReID encoder interface and embedding maths.

The rest of the application never learns whether the backend is ONNX Runtime,
PyTorch or TensorRT -- it only sees :class:`ReIDEncoder`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class EncoderInfo:
    """What was actually loaded, reported at startup and by the API."""

    name: str
    model_path: str
    backend: str
    device: str
    fp16: bool
    input_size: tuple[int, int]
    embedding_dimension: int
    load_time_s: float = 0.0
    max_batch_size: int = 0
    """Largest batch this model can actually execute. 0 means "unknown or
    unconstrained"; 1 means the graph has a fixed batch dimension and batching
    it is impossible regardless of what the configuration asks for."""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_batching(self) -> bool:
        return self.max_batch_size != 1

    @property
    def fingerprint(self) -> str:
        """Identifies the embedding space.

        Cached gallery embeddings are only reusable when this string is
        unchanged: a different model or input size means a different space.
        """
        h, w = self.input_size
        return f"{self.backend}:{self.name}:{h}x{w}:{self.embedding_dimension}"


def l2_normalize(vectors: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """L2-normalise so cosine similarity reduces to a dot product.

    Zero vectors are left at zero (their similarity to anything is 0), which
    keeps a degenerate crop from matching an identity by accident.
    """
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    out = np.zeros_like(array)
    return np.divide(array, norms, out=out, where=norms > _EPS)


class ReIDEncoder(ABC):
    """Turns person crops into fixed-length appearance embeddings."""

    @abstractmethod
    def load(self) -> EncoderInfo:
        """Load the model and determine the embedding dimension. Idempotent."""

    @abstractmethod
    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR crops.

        Returns:
            Array of shape ``(len(crops), D)``. Rows for invalid crops are zero.
        """

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Embed a single BGR crop into a ``(D,)`` vector."""
        return self.embed_batch([crop])[0]

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2-normalise a single embedding or a batch of them."""
        return l2_normalize(embedding)

    def get_embedding_dimension(self) -> int:
        """Embedding width, read from the model rather than assumed."""
        return self.info.embedding_dimension

    @property
    @abstractmethod
    def info(self) -> EncoderInfo:
        """Information about the loaded model."""

    def close(self) -> None:
        """Release resources. Safe to call multiple times."""

    def __enter__(self) -> ReIDEncoder:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
