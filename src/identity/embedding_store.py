"""Persistent embedding storage.

:class:`EmbeddingStore` is the seam a vector database would slot into later
(FAISS, Qdrant, Milvus, pgvector). The shipped implementation keeps embeddings
as ``.npy`` files plus JSON metadata, which is transparent, dependency-free and
fast enough for the identity counts this system targets.

Search is deliberately *not* part of this interface: similarity is computed by
:mod:`src.identity.matcher` against the ``[N, D]`` matrix the store exposes, so
an ANN-backed store can add its own search without changing the matcher's
contract.
"""

from __future__ import annotations

import datetime as _dt
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import EmbeddingStoreError
from src.utils.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class StoredEmbedding:
    """One persisted identity vector and everything needed to validate it."""

    id: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return int(self.embedding.shape[-1])


class EmbeddingStore(ABC):
    """Persistence for identity embeddings."""

    @abstractmethod
    def put(self, item: StoredEmbedding) -> None:
        """Insert or replace an entry."""

    @abstractmethod
    def get(self, identity_id: str) -> StoredEmbedding | None:
        """Fetch one entry, or ``None`` when absent."""

    @abstractmethod
    def delete(self, identity_id: str) -> bool:
        """Remove an entry; returns whether anything was removed."""

    @abstractmethod
    def ids(self) -> list[str]:
        """All stored identity ids, in insertion-independent sorted order."""

    @abstractmethod
    def all(self) -> list[StoredEmbedding]:
        """All stored entries."""

    def __contains__(self, identity_id: object) -> bool:
        return isinstance(identity_id, str) and self.get(identity_id) is not None

    def __len__(self) -> int:
        return len(self.ids())

    def __iter__(self) -> Iterator[StoredEmbedding]:
        return iter(self.all())

    def matrix(self, ids: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Return a ``[N, D]`` matrix plus the row-aligned identity ids.

        Vectorised matching against this matrix is what keeps recognition cheap
        as the number of registered people grows.
        """
        items = (
            [item for item in self.all() if item.id in set(ids)] if ids is not None else self.all()
        )
        items.sort(key=lambda entry: entry.id)
        if not items:
            return np.zeros((0, 0), dtype=np.float32), []
        dimensions = {item.dimension for item in items}
        if len(dimensions) > 1:
            raise EmbeddingStoreError(
                f"stored embeddings have mixed dimensions {sorted(dimensions)}; "
                "rebuild the gallery after changing the ReID model "
                "(python main.py gallery build --rebuild)"
            )
        return (
            np.stack([item.embedding.astype(np.float32, copy=False) for item in items], axis=0),
            [item.id for item in items],
        )


class LocalNumpyStore(EmbeddingStore):
    """Filesystem store: ``embeddings/<id>.npy`` + ``metadata/<id>.json``."""

    def __init__(self, embeddings_dir: Path, metadata_dir: Path) -> None:
        self._embeddings_dir = Path(embeddings_dir)
        self._metadata_dir = Path(metadata_dir)
        self._cache: dict[str, StoredEmbedding] = {}
        self._scanned = False

    # ------------------------------------------------------------- filesystem
    def _ensure_dirs(self) -> None:
        for directory in (self._embeddings_dir, self._metadata_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise EmbeddingStoreError(
                    f"cannot create gallery directory {directory}: {exc}"
                ) from exc

    def _embedding_path(self, identity_id: str) -> Path:
        return self._embeddings_dir / f"{identity_id}.npy"

    def _metadata_path(self, identity_id: str) -> Path:
        return self._metadata_dir / f"{identity_id}.json"

    # ------------------------------------------------------------------- API
    def put(self, item: StoredEmbedding) -> None:
        self._ensure_dirs()
        embedding = np.asarray(item.embedding, dtype=np.float32).reshape(-1)
        if embedding.size == 0:
            raise EmbeddingStoreError(f"refusing to store an empty embedding for '{item.id}'")

        metadata = dict(item.metadata)
        metadata.update(
            {
                "id": item.id,
                "schema_version": SCHEMA_VERSION,
                "embedding_file": str(self._embedding_path(item.id)),
                "embedding_dimension": int(embedding.shape[0]),
                "updated_at": utc_now(),
            }
        )
        metadata.setdefault("created_at", metadata["updated_at"])

        try:
            np.save(self._embedding_path(item.id), embedding)
            self._metadata_path(item.id).write_text(
                json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
            )
        except OSError as exc:
            raise EmbeddingStoreError(
                f"cannot persist embedding for '{item.id}': {exc}. "
                "Check gallery.directory and filesystem permissions."
            ) from exc

        self._cache[item.id] = StoredEmbedding(item.id, embedding, metadata)
        logger.debug(
            "Embedding stored", extra={"identity": item.id, "dim": int(embedding.shape[0])}
        )

    def get(self, identity_id: str) -> StoredEmbedding | None:
        if identity_id in self._cache:
            return self._cache[identity_id]
        embedding_path = self._embedding_path(identity_id)
        if not embedding_path.exists():
            return None
        item = self._load(identity_id)
        if item is not None:
            self._cache[identity_id] = item
        return item

    def _load(self, identity_id: str) -> StoredEmbedding | None:
        embedding_path = self._embedding_path(identity_id)
        metadata_path = self._metadata_path(identity_id)
        try:
            embedding = np.load(embedding_path).astype(np.float32).reshape(-1)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Corrupt embedding cache discarded; it will be regenerated",
                extra={"identity": identity_id, "path": str(embedding_path), "error": str(exc)},
            )
            self._discard(identity_id)
            return None
        if embedding.size == 0:
            logger.warning("Empty embedding cache discarded", extra={"identity": identity_id})
            self._discard(identity_id)
            return None

        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Corrupt gallery metadata ignored; the entry will be re-enrolled",
                    extra={"identity": identity_id, "error": str(exc)},
                )
                self._discard(identity_id)
                return None
        return StoredEmbedding(identity_id, embedding, metadata)

    def _discard(self, identity_id: str) -> None:
        """Delete an unusable cache entry so the next run regenerates it."""
        for path in (self._embedding_path(identity_id), self._metadata_path(identity_id)):
            try:
                path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort cleanup
                logger.debug("Could not remove %s", path)
        self._cache.pop(identity_id, None)

    def delete(self, identity_id: str) -> bool:
        existed = self._embedding_path(identity_id).exists()
        self._discard(identity_id)
        return existed

    def ids(self) -> list[str]:
        if not self._embeddings_dir.exists():
            return []
        return sorted(path.stem for path in self._embeddings_dir.glob("*.npy"))

    def all(self) -> list[StoredEmbedding]:
        items: list[StoredEmbedding] = []
        for identity_id in self.ids():
            item = self.get(identity_id)
            if item is not None:
                items.append(item)
        self._scanned = True
        return items

    def clear(self) -> int:
        """Remove every stored entry. Returns how many were deleted."""
        count = 0
        for identity_id in self.ids():
            count += int(self.delete(identity_id))
        return count


class InMemoryEmbeddingStore(EmbeddingStore):
    """Non-persistent store used by tests and by transient API sessions."""

    def __init__(self) -> None:
        self._items: dict[str, StoredEmbedding] = {}

    def put(self, item: StoredEmbedding) -> None:
        embedding = np.asarray(item.embedding, dtype=np.float32).reshape(-1)
        metadata = dict(item.metadata)
        metadata.setdefault("created_at", utc_now())
        metadata["updated_at"] = utc_now()
        metadata["embedding_dimension"] = int(embedding.shape[0])
        self._items[item.id] = StoredEmbedding(item.id, embedding, metadata)

    def get(self, identity_id: str) -> StoredEmbedding | None:
        return self._items.get(identity_id)

    def delete(self, identity_id: str) -> bool:
        return self._items.pop(identity_id, None) is not None

    def ids(self) -> list[str]:
        return sorted(self._items)

    def all(self) -> list[StoredEmbedding]:
        return [self._items[key] for key in self.ids()]


def build_store(kind: str, embeddings_dir: Path, metadata_dir: Path) -> EmbeddingStore:
    """Instantiate the configured embedding store."""
    if kind == "local_numpy":
        return LocalNumpyStore(embeddings_dir, metadata_dir)
    raise EmbeddingStoreError(  # pragma: no cover - enum-constrained
        f"unsupported gallery.store '{kind}'. Available: local_numpy."
    )
