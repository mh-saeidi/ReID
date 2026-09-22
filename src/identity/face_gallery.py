"""Multi-embedding face identity gallery.

One passport photo remains the only thing an operator must supply. But a single
reference embedding is a single point in a space the query will approach from
many directions -- a different camera, a decade later, wearing glasses. The
gallery is therefore structured from the start to hold several embeddings per
person, with the passport reference always present and always privileged.

Layout on disk, one directory per person:

    data/people/person_001/
        reference/passport.jpg      a copy; the operator's original is untouched
        face/reference.npy          the passport embedding
        face/live_001.npy           conservatively added observations
        metadata.json

Matching takes the *maximum* similarity across a person's embeddings rather
than the mean. Averaging embeddings from different conditions produces a
centroid that resembles none of them; taking the best match asks the question
that is actually being posed -- "does this query look like any view of this
person that we trust".
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.core.exceptions import GalleryError
from src.identity.types import FaceMatchResult
from src.reid.encoder import l2_normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)

REFERENCE_KEY = "reference"
SCHEMA_VERSION = 2


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class FaceEmbeddingRecord:
    """One stored embedding and where it came from."""

    key: str
    vector: np.ndarray
    source: str = "reference"
    """``reference`` (the passport) or ``live`` (an adapted observation)."""
    quality: float = 0.0
    similarity_at_capture: float = 0.0
    created_at: str = field(default_factory=_utc_now)
    frame_index: int = -1
    track_id: int = -1
    notes: str = ""

    @property
    def is_reference(self) -> bool:
        return self.key == REFERENCE_KEY

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "source": self.source,
            "quality": round(self.quality, 4),
            "similarity_at_capture": round(self.similarity_at_capture, 4),
            "created_at": self.created_at,
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "dimension": int(self.vector.shape[-1]),
            "notes": self.notes,
        }


@dataclass(slots=True)
class FaceIdentity:
    """A registered person and every embedding held for them."""

    id: str
    name: str
    title: str = ""
    enabled: bool = True
    reference_image: str = ""
    embeddings: dict[str, FaceEmbeddingRecord] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    encoder_fingerprint: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    @property
    def reference(self) -> FaceEmbeddingRecord | None:
        return self.embeddings.get(REFERENCE_KEY)

    @property
    def live_count(self) -> int:
        return sum(1 for r in self.embeddings.values() if not r.is_reference)

    @property
    def display_label(self) -> str:
        return f"{self.name} ({self.title})" if self.title else self.name

    def matrix(self) -> tuple[np.ndarray, list[str]]:
        """``[K, D]`` stack of this person's embeddings, plus their keys."""
        keys = sorted(self.embeddings)
        if not keys:
            return np.zeros((0, 0), dtype=np.float32), []
        return (
            np.stack([self.embeddings[k].vector for k in keys]).astype(np.float32),
            keys,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "enabled": self.enabled,
            "reference_image": self.reference_image,
            "encoder_fingerprint": self.encoder_fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "quality": self.quality,
            "embeddings": {k: r.to_dict() for k, r in sorted(self.embeddings.items())},
        }


class FaceGallery:
    """Registered identities, their face embeddings, and matching.

    Args:
        root: Directory holding one subdirectory per person.
        max_live_per_identity: Ceiling on adapted embeddings, so a long-running
            deployment cannot grow a person's entry without bound.
    """

    def __init__(self, root: Path, *, max_live_per_identity: int = 20) -> None:
        self._root = Path(root)
        self._max_live = max_live_per_identity
        self._identities: dict[str, FaceIdentity] = {}
        self._matrix: np.ndarray | None = None
        self._row_owner: list[tuple[str, str]] = []

    # ------------------------------------------------------------ accessors
    @property
    def root(self) -> Path:
        return self._root

    @property
    def identities(self) -> list[FaceIdentity]:
        return [self._identities[k] for k in sorted(self._identities)]

    @property
    def active(self) -> list[FaceIdentity]:
        return [i for i in self.identities if i.enabled and i.embeddings]

    def get(self, identity_id: str) -> FaceIdentity | None:
        return self._identities.get(identity_id)

    def __len__(self) -> int:
        return len(self._identities)

    def __iter__(self) -> Iterator[FaceIdentity]:
        return iter(self.identities)

    @property
    def embedding_count(self) -> int:
        return sum(len(i.embeddings) for i in self.active)

    # -------------------------------------------------------------- storage
    def person_dir(self, identity_id: str) -> Path:
        return self._root / identity_id

    def put(self, identity: FaceIdentity) -> None:
        self._identities[identity.id] = identity
        self._invalidate()

    def save(self, identity: FaceIdentity) -> Path:
        """Persist one person's directory."""
        directory = self.person_dir(identity.id)
        (directory / "face").mkdir(parents=True, exist_ok=True)
        (directory / "reference").mkdir(parents=True, exist_ok=True)
        identity.updated_at = _utc_now()

        for key, record in identity.embeddings.items():
            np.save(directory / "face" / f"{key}.npy", record.vector.astype(np.float32))
        # Remove embedding files that are no longer part of this identity.
        for existing in (directory / "face").glob("*.npy"):
            if existing.stem not in identity.embeddings:
                existing.unlink(missing_ok=True)

        metadata = directory / "metadata.json"
        metadata.write_text(
            json.dumps(identity.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        self._identities[identity.id] = identity
        self._invalidate()
        return metadata

    def copy_reference_image(self, identity_id: str, source: Path) -> Path:
        """Keep a copy of the passport photo beside the embeddings.

        The operator's original file is only ever read; this copy exists so the
        gallery entry can be audited or rebuilt later without depending on a
        path outside the project.
        """
        target_dir = self.person_dir(identity_id) / "reference"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"passport{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, target)
        return target

    def load(self, encoder_fingerprint: str | None = None) -> int:
        """Read every person directory. Returns how many are usable."""
        self._identities.clear()
        if not self._root.exists():
            return 0

        for directory in sorted(p for p in self._root.iterdir() if p.is_dir()):
            metadata_path = directory / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Skipping unreadable gallery entry",
                    extra={"path": str(metadata_path), "error": str(exc)},
                )
                continue

            if (
                encoder_fingerprint
                and data.get("encoder_fingerprint")
                and data["encoder_fingerprint"] != encoder_fingerprint
            ):
                logger.warning(
                    "Gallery entry was built with a different face encoder; "
                    "re-enroll with 'gallery build --rebuild'",
                    extra={"identity": data.get("id", directory.name)},
                )
                continue

            identity = FaceIdentity(
                id=data.get("id", directory.name),
                name=data.get("name", directory.name),
                title=data.get("title", ""),
                enabled=bool(data.get("enabled", True)),
                reference_image=data.get("reference_image", ""),
                quality=data.get("quality", {}) or {},
                encoder_fingerprint=data.get("encoder_fingerprint", ""),
                created_at=data.get("created_at", _utc_now()),
                updated_at=data.get("updated_at", _utc_now()),
            )
            for key, record in (data.get("embeddings") or {}).items():
                vector_path = directory / "face" / f"{key}.npy"
                if not vector_path.exists():
                    continue
                try:
                    vector = l2_normalize(np.load(vector_path).astype(np.float32).reshape(-1))
                except (OSError, ValueError) as exc:
                    logger.warning("Dropping unreadable embedding %s: %s", vector_path, exc)
                    continue
                identity.embeddings[key] = FaceEmbeddingRecord(
                    key=key,
                    vector=vector,
                    source=record.get("source", "reference"),
                    quality=float(record.get("quality", 0.0)),
                    similarity_at_capture=float(record.get("similarity_at_capture", 0.0)),
                    created_at=record.get("created_at", identity.created_at),
                    frame_index=int(record.get("frame_index", -1)),
                    track_id=int(record.get("track_id", -1)),
                    notes=record.get("notes", ""),
                )
            if identity.embeddings:
                self._identities[identity.id] = identity

        self._invalidate()
        logger.info(
            "Face gallery loaded",
            extra={
                "identities": len(self.active),
                "embeddings": self.embedding_count,
                "root": str(self._root),
            },
        )
        return len(self.active)

    def remove(self, identity_id: str) -> bool:
        removed = self._identities.pop(identity_id, None) is not None
        directory = self.person_dir(identity_id)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            removed = True
        self._invalidate()
        return removed

    # --------------------------------------------------------------- search
    def _invalidate(self) -> None:
        self._matrix = None
        self._row_owner = []

    def _build_matrix(self) -> None:
        """Flatten every embedding of every active identity into one matrix.

        One matrix multiplication then scores the whole gallery, and each row
        remembers which person and which embedding it came from.
        """
        rows: list[np.ndarray] = []
        owners: list[tuple[str, str]] = []
        for identity in self.active:
            for key in sorted(identity.embeddings):
                rows.append(identity.embeddings[key].vector)
                owners.append((identity.id, key))
        if not rows:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._row_owner = []
            return
        dimensions = {r.shape[-1] for r in rows}
        if len(dimensions) > 1:
            raise GalleryError(
                f"gallery holds mixed embedding dimensions {sorted(dimensions)}; "
                "rebuild after changing the face encoder"
            )
        self._matrix = l2_normalize(np.stack(rows).astype(np.float32), axis=1)
        self._row_owner = owners

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._build_matrix()
        assert self._matrix is not None  # noqa: S101
        return self._matrix

    @property
    def is_empty(self) -> bool:
        return self.matrix.shape[0] == 0

    def match(self, embedding: np.ndarray) -> FaceMatchResult:
        """Score one query against every embedding, best-per-identity.

        The best embedding of a person wins for that person; averaging across
        a person's stored views would blur exactly the variation they were
        stored to capture.
        """
        if self.is_empty:
            return FaceMatchResult(gallery_size=0)

        query = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if query.shape[0] != self.matrix.shape[1]:
            raise GalleryError(
                f"query embedding has dimension {query.shape[0]} but the gallery "
                f"holds {self.matrix.shape[1]}; rebuild the gallery after "
                "changing the face encoder"
            )

        scores = self.matrix @ query
        best_per_identity: dict[str, tuple[float, str]] = {}
        for row, (identity_id, key) in enumerate(self._row_owner):
            score = float(scores[row])
            current = best_per_identity.get(identity_id)
            if current is None or score > current[0]:
                best_per_identity[identity_id] = (score, key)

        ranked = sorted(
            ((i, s) for i, (s, _) in best_per_identity.items()), key=lambda p: -p[1]
        )
        best_id, best_score = ranked[0]
        runner_id, runner_score = (ranked[1] if len(ranked) > 1 else (None, 0.0))
        return FaceMatchResult(
            best_identity_id=best_id,
            best_similarity=best_score,
            runner_up_id=runner_id,
            runner_up_similarity=runner_score,
            matched_embedding_key=best_per_identity[best_id][1],
            ranked=ranked,
            gallery_size=len(best_per_identity),
        )

    # ----------------------------------------------------------- adaptation
    def add_live_embedding(
        self,
        identity_id: str,
        vector: np.ndarray,
        *,
        quality: float,
        similarity: float,
        frame_index: int = -1,
        track_id: int = -1,
        notes: str = "",
    ) -> FaceEmbeddingRecord | None:
        """Add a verified live observation. Returns the record, or ``None``.

        The caller is responsible for deciding this observation is trustworthy;
        see :mod:`src.identity.adaptation`, which owns that gating. This method
        only enforces the storage invariants.
        """
        identity = self._identities.get(identity_id)
        if identity is None:
            return None
        if identity.live_count >= self._max_live:
            # Replace the weakest live sample rather than growing without bound
            # -- but never the passport reference, which is the ground truth.
            live = [r for r in identity.embeddings.values() if not r.is_reference]
            weakest = min(live, key=lambda r: r.quality)
            if weakest.quality >= quality:
                return None
            del identity.embeddings[weakest.key]
            logger.debug(
                "Evicted the weakest live embedding to make room",
                extra={"identity": identity_id, "evicted": weakest.key},
            )

        index = 1
        while f"live_{index:03d}" in identity.embeddings:
            index += 1
        record = FaceEmbeddingRecord(
            key=f"live_{index:03d}",
            vector=l2_normalize(np.asarray(vector, dtype=np.float32).reshape(-1)),
            source="live",
            quality=quality,
            similarity_at_capture=similarity,
            frame_index=frame_index,
            track_id=track_id,
            notes=notes,
        )
        identity.embeddings[record.key] = record
        self.save(identity)
        return record

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "id": i.id,
                "name": i.name,
                "title": i.title,
                "enabled": i.enabled,
                "embeddings": len(i.embeddings),
                "live_embeddings": i.live_count,
                "has_reference": i.reference is not None,
                "reference_image": i.reference_image,
                "dimension": (
                    int(i.reference.vector.shape[-1]) if i.reference is not None else None
                ),
                "created_at": i.created_at,
                "updated_at": i.updated_at,
                "quality_warnings": list(i.quality.get("warnings", []) or []),
            }
            for i in self.identities
        ]
