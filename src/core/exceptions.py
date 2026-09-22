"""Application exception hierarchy.

Every exception carries an actionable message: what failed, which artefact was
involved and what the operator can do about it.
"""

from __future__ import annotations


class ReIDSystemError(Exception):
    """Base class for all errors raised by this application."""


class ConfigurationError(ReIDSystemError):
    """Invalid, missing or contradictory configuration."""


class ModelLoadError(ReIDSystemError):
    """A detector / ReID backend could not be loaded."""


class BackendUnavailableError(ModelLoadError):
    """The requested inference backend is not installed."""


class SourceError(ReIDSystemError):
    """An input source could not be opened or read."""


class EnrollmentError(ReIDSystemError):
    """A reference image could not be turned into a gallery entry."""


class NoPersonFoundError(EnrollmentError):
    """No person was detected in the reference image."""


class AmbiguousEnrollmentError(EnrollmentError):
    """Several persons were detected and automatic selection is not allowed."""


class NoFaceFoundError(EnrollmentError):
    """Face mode: the reference image has no usable face.

    Never downgraded to body-appearance enrollment -- that would silently
    reintroduce the clothing dependence face mode exists to remove.
    """


class GalleryError(ReIDSystemError):
    """The identity gallery could not be read, written or validated."""


class EmbeddingStoreError(GalleryError):
    """The persistent embedding store is corrupt or unwritable."""


class OutputError(ReIDSystemError):
    """Snapshots / videos / metadata could not be written."""
