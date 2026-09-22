"""The identity gallery: registered people and their reference embeddings.

Embeddings are expensive to compute, so they are cached on disk and only
regenerated when something that affects them changed -- the reference image, or
the encoder fingerprint (model, input geometry, embedding width). Starting a
video therefore does not re-enroll anybody.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.paths import ProjectPaths
from src.config.schema import AppConfig, PersonConfig
from src.core.exceptions import EnrollmentError, GalleryError
from src.core.types import PersonIdentity
from src.identity.embedding_store import (
    EmbeddingStore,
    StoredEmbedding,
    build_store,
    utc_now,
)
from src.identity.enrollment import Enroller
from src.identity.matcher import GalleryView, build_gallery_view
from src.reid.encoder import EncoderInfo, l2_normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def file_fingerprint(path: Path) -> str:
    """Cheap change detector for a reference image (size + mtime + head bytes)."""
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            head = handle.read(8192)
    except OSError:
        return ""
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(int(stat.st_mtime)).encode())
    digest.update(head)
    return digest.hexdigest()[:32]


@dataclass(slots=True)
class BuildReport:
    """Summary of a gallery build, printed by the CLI and returned by the API."""

    enrolled: list[str]
    reused: list[str]
    skipped: list[str]
    failed: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def total(self) -> int:
        return len(self.enrolled) + len(self.reused)

    def to_dict(self) -> dict[str, object]:
        return {
            "enrolled": self.enrolled,
            "reused": self.reused,
            "skipped": self.skipped,
            "failed": self.failed,
            "total_active": self.total,
        }


class IdentityGallery:
    """Loads, validates, caches and serves registered identities."""

    def __init__(
        self,
        config: AppConfig,
        paths: ProjectPaths,
        store: EmbeddingStore | None = None,
    ) -> None:
        self._config = config
        self._paths = paths
        self._store = store or build_store(
            config.gallery.store.value, paths.embeddings_dir, paths.metadata_dir
        )
        self._identities: dict[str, PersonIdentity] = {}
        self._view: GalleryView | None = None

    # ------------------------------------------------------------- properties
    @property
    def store(self) -> EmbeddingStore:
        return self._store

    @property
    def identities(self) -> list[PersonIdentity]:
        return [self._identities[key] for key in sorted(self._identities)]

    @property
    def active_identities(self) -> list[PersonIdentity]:
        return [i for i in self.identities if i.enabled and i.embedding is not None]

    def get(self, identity_id: str) -> PersonIdentity | None:
        return self._identities.get(identity_id)

    def __len__(self) -> int:
        return len(self._identities)

    @property
    def view(self) -> GalleryView:
        """Matching snapshot, rebuilt lazily after any mutation."""
        if self._view is None:
            self._view = build_gallery_view(self.active_identities)
        return self._view

    def _invalidate(self) -> None:
        self._view = None

    # ----------------------------------------------------------------- build
    def build(
        self,
        enroller: Enroller,
        encoder_info: EncoderInfo,
        *,
        force: bool = False,
        only: Iterable[str] | None = None,
    ) -> BuildReport:
        """Ensure every configured person has a valid, current embedding.

        Args:
            enroller: Performs detection + embedding for reference images.
            encoder_info: Identifies the embedding space the cache must match.
            force: Re-enroll even when a valid cached embedding exists.
            only: Restrict the build to these person ids.
        """
        wanted = set(only) if only is not None else None
        report = BuildReport(enrolled=[], reused=[], skipped=[], failed={})

        configured_ids = {person.id for person in self._config.people}
        for person in self._config.people:
            if wanted is not None and person.id not in wanted:
                continue
            if not person.enabled:
                report.skipped.append(person.id)
                self._register_disabled(person)
                continue
            try:
                reused = self._build_one(person, enroller, encoder_info, force=force)
            except (EnrollmentError, GalleryError) as exc:
                report.failed[person.id] = str(exc)
                logger.error(
                    "Enrollment failed", extra={"identity": person.id, "error": str(exc)}
                )
                continue
            (report.reused if reused else report.enrolled).append(person.id)

        if wanted is not None:
            missing = wanted - configured_ids
            if missing:
                raise GalleryError(
                    f"unknown person id(s): {sorted(missing)}. "
                    f"Configured ids: {sorted(configured_ids) or '<none>'}"
                )

        self._prune_orphans(configured_ids)
        self._invalidate()
        logger.info(
            "Identity gallery ready",
            extra={
                "active": len(self.active_identities),
                "enrolled": len(report.enrolled),
                "reused": len(report.reused),
                "failed": len(report.failed),
            },
        )
        return report

    def _build_one(
        self,
        person: PersonConfig,
        enroller: Enroller,
        encoder_info: EncoderInfo,
        *,
        force: bool,
    ) -> bool:
        """Enroll or reuse one person. Returns True when the cache was reused."""
        image_paths = [self._paths.resolve(p) for p in person.all_image_paths]
        missing = [str(p) for p in image_paths if not p.exists()]
        if missing:
            raise EnrollmentError(
                f"reference image(s) for '{person.id}' not found: {', '.join(missing)}"
            )

        if not force:
            cached = self._load_cached(person, image_paths, encoder_info)
            if cached is not None:
                self._identities[person.id] = cached
                logger.debug("Reusing cached embedding", extra={"identity": person.id})
                return True

        if not self._config.enrollment.regenerate_missing_embeddings and not force:
            raise GalleryError(
                f"no valid cached embedding for '{person.id}' and "
                "enrollment.regenerate_missing_embeddings is false. "
                "Run: python main.py gallery build --rebuild"
            )

        result = enroller.enroll(person, image_paths)
        metadata = {
            "id": person.id,
            "name": person.name,
            "title": person.title,
            "image_path": str(image_paths[0]),
            "image_paths": [str(p) for p in image_paths],
            "image_fingerprints": [file_fingerprint(p) for p in image_paths],
            "encoder_fingerprint": encoder_info.fingerprint,
            "encoder_model": encoder_info.name,
            "embedding_dimension": result.dimension,
            "detector_confidence": round(result.detector_confidence, 4),
            "bbox": [round(v, 2) for v in result.bbox.to_list()],
            "crop_path": result.crop_path,
            "quality": result.quality.to_dict(),
            "created_at": utc_now(),
        }
        existing = self._store.get(person.id)
        if existing is not None and existing.metadata.get("created_at"):
            metadata["created_at"] = existing.metadata["created_at"]

        self._store.put(StoredEmbedding(person.id, result.embedding, metadata))
        stored = self._store.get(person.id)
        self._identities[person.id] = self._to_identity(person, result.embedding, stored)
        self._invalidate()
        return False

    def _load_cached(
        self,
        person: PersonConfig,
        image_paths: list[Path],
        encoder_info: EncoderInfo,
    ) -> PersonIdentity | None:
        """Return a cached identity when it is still valid, else ``None``."""
        stored = self._store.get(person.id)
        if stored is None:
            return None

        metadata = stored.metadata
        if metadata.get("encoder_fingerprint") != encoder_info.fingerprint:
            logger.info(
                "ReID model changed; re-enrolling",
                extra={
                    "identity": person.id,
                    "cached": metadata.get("encoder_fingerprint", "<unknown>"),
                    "current": encoder_info.fingerprint,
                },
            )
            return None
        if stored.dimension != encoder_info.embedding_dimension:
            return None

        if self._config.gallery.revalidate_on_load:
            cached_paths = metadata.get("image_paths") or [metadata.get("image_path")]
            if [str(p) for p in image_paths] != [str(p) for p in cached_paths]:
                logger.info("Reference image path changed; re-enrolling",
                            extra={"identity": person.id})
                return None
            cached_fingerprints = metadata.get("image_fingerprints") or []
            current = [file_fingerprint(p) for p in image_paths]
            if list(cached_fingerprints) != current:
                logger.info("Reference image changed on disk; re-enrolling",
                            extra={"identity": person.id})
                return None

        # Display fields are cheap to refresh from configuration.
        if metadata.get("name") != person.name or metadata.get("title") != person.title:
            metadata = {**metadata, "name": person.name, "title": person.title}
            self._store.put(StoredEmbedding(person.id, stored.embedding, metadata))
            stored = self._store.get(person.id) or stored

        return self._to_identity(person, stored.embedding, stored)

    def _to_identity(
        self,
        person: PersonConfig,
        embedding: np.ndarray,
        stored: StoredEmbedding | None,
    ) -> PersonIdentity:
        metadata = stored.metadata if stored else {}
        vector = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        return PersonIdentity(
            id=person.id,
            name=person.name,
            title=person.title,
            image_paths=[str(self._paths.resolve(p)) for p in person.all_image_paths],
            embedding=vector,
            embedding_dimension=int(vector.shape[-1]),
            enabled=person.enabled,
            created_at=str(metadata.get("created_at", "")),
            updated_at=str(metadata.get("updated_at", "")),
            quality=dict(metadata.get("quality", {}) or {}),
            source_model=str(metadata.get("encoder_model", "")),
        )

    def _register_disabled(self, person: PersonConfig) -> None:
        self._identities[person.id] = PersonIdentity(
            id=person.id,
            name=person.name,
            title=person.title,
            image_paths=[str(self._paths.resolve(p)) for p in person.all_image_paths],
            enabled=False,
        )

    def _prune_orphans(self, configured_ids: set[str]) -> None:
        """Drop cached embeddings for people removed from the configuration."""
        for identity_id in self._store.ids():
            if identity_id not in configured_ids:
                self._store.delete(identity_id)
                self._identities.pop(identity_id, None)
                logger.info(
                    "Removed gallery entry not present in the configuration",
                    extra={"identity": identity_id},
                )

    # ----------------------------------------------------------- load / mutate
    def load(self, encoder_info: EncoderInfo | None = None) -> int:
        """Load cached identities without running any enrollment."""
        self._identities.clear()
        for person in self._config.people:
            stored = self._store.get(person.id)
            if stored is None:
                self._register_disabled(person) if not person.enabled else None
                continue
            if (
                encoder_info is not None
                and stored.metadata.get("encoder_fingerprint") != encoder_info.fingerprint
            ):
                logger.warning(
                    "Cached embedding was produced by a different ReID model; "
                    "run 'gallery build --rebuild'",
                    extra={"identity": person.id},
                )
                continue
            identity = self._to_identity(person, stored.embedding, stored)
            identity.enabled = person.enabled
            self._identities[person.id] = identity
        self._invalidate()
        return len(self.active_identities)

    def set_enabled(self, identity_id: str, enabled: bool) -> PersonIdentity:
        identity = self._identities.get(identity_id)
        if identity is None:
            raise GalleryError(f"unknown identity '{identity_id}'")
        identity.enabled = enabled
        self._invalidate()
        return identity

    def remove(self, identity_id: str) -> bool:
        removed = self._store.delete(identity_id)
        self._identities.pop(identity_id, None)
        self._invalidate()
        return removed

    def clear(self) -> int:
        count = 0
        for identity_id in list(self._store.ids()):
            count += int(self._store.delete(identity_id))
        self._identities.clear()
        self._invalidate()
        return count

    def summary(self) -> list[dict[str, object]]:
        """Serialisable listing used by ``gallery list`` and the API."""
        rows: list[dict[str, object]] = []
        for identity in self.identities:
            rows.append(
                {
                    "id": identity.id,
                    "name": identity.name,
                    "title": identity.title,
                    "enabled": identity.enabled,
                    "has_embedding": identity.embedding is not None,
                    "embedding_dimension": identity.embedding_dimension,
                    "image_path": identity.image_path,
                    "created_at": identity.created_at,
                    "updated_at": identity.updated_at,
                    "source_model": identity.source_model,
                    "quality_warnings": list(identity.quality.get("warnings", []) or []),
                }
            )
        return rows
