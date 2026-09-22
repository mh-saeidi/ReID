"""Identity enrollment, storage and matching."""

from src.identity.embedding_store import (
    EmbeddingStore,
    InMemoryEmbeddingStore,
    LocalNumpyStore,
    StoredEmbedding,
)
from src.identity.enrollment import Enroller, EnrollmentResult, select_detection
from src.identity.gallery import BuildReport, IdentityGallery
from src.identity.matcher import GalleryView, IdentityMatcher, build_gallery_view

__all__ = [
    "EmbeddingStore",
    "LocalNumpyStore",
    "InMemoryEmbeddingStore",
    "StoredEmbedding",
    "Enroller",
    "EnrollmentResult",
    "select_detection",
    "IdentityGallery",
    "BuildReport",
    "IdentityMatcher",
    "GalleryView",
    "build_gallery_view",
]
