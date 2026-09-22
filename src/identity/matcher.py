"""Open-set identity matching.

Query embeddings are compared against the whole gallery in one vectorised
operation (``[M, D] @ [D, N]``), and the best match is only *accepted* when it
clears the configured threshold. There is always an ``unknown`` outcome: an
unregistered person must never be forced onto the nearest registered identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config.schema import MatchingConfig, SimilarityMetric
from src.core.types import PersonIdentity, RecognitionResult, RecognitionStatus
from src.reid.encoder import l2_normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GalleryView:
    """Immutable snapshot of the gallery used for a batch of comparisons."""

    matrix: np.ndarray
    """``[N, D]`` L2-normalised reference embeddings."""
    ids: list[str]
    identities: dict[str, PersonIdentity]

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0

    @property
    def is_empty(self) -> bool:
        return self.size == 0


def cosine_similarity(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Cosine similarity for arbitrary inputs, shape ``[M, N]``.

    Both sides are re-normalised defensively: correctness must not depend on a
    caller remembering to normalise.
    """
    if queries.size == 0 or gallery.size == 0:
        return np.zeros((queries.shape[0] if queries.ndim > 1 else 1, gallery.shape[0]), np.float32)
    return l2_normalize(np.atleast_2d(queries), axis=1) @ l2_normalize(gallery, axis=1).T


def euclidean_similarity(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Map Euclidean distance between unit vectors onto ``[0, 1]``.

    For L2-normalised vectors ``d = sqrt(2 - 2*cos)``, with ``d`` in ``[0, 2]``,
    so ``1 - d/2`` is a bounded similarity comparable to a cosine threshold.
    """
    q = l2_normalize(np.atleast_2d(queries), axis=1)
    g = l2_normalize(gallery, axis=1)
    distances = np.linalg.norm(q[:, None, :] - g[None, :, :], axis=2)
    return (1.0 - distances / 2.0).astype(np.float32)


def dot_similarity(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Raw inner product -- for encoders whose output scale is meaningful."""
    return np.atleast_2d(np.asarray(queries, dtype=np.float32)) @ np.asarray(
        gallery, dtype=np.float32
    ).T


_METRICS = {
    SimilarityMetric.COSINE: cosine_similarity,
    SimilarityMetric.EUCLIDEAN: euclidean_similarity,
    SimilarityMetric.DOT: dot_similarity,
}


class IdentityMatcher:
    """Decides, per query embedding, which registered identity it is -- if any."""

    def __init__(self, config: MatchingConfig) -> None:
        self._config = config
        self._similarity = _METRICS[config.metric]

    @property
    def config(self) -> MatchingConfig:
        return self._config

    def similarity_matrix(self, queries: np.ndarray, view: GalleryView) -> np.ndarray:
        """``[M, N]`` similarities between query embeddings and the gallery."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        if view.is_empty:
            return np.zeros((queries.shape[0], 0), dtype=np.float32)
        if queries.shape[1] != view.dimension:
            raise ValueError(
                f"embedding dimension mismatch: query has {queries.shape[1]}, "
                f"gallery has {view.dimension}. Rebuild the gallery after changing "
                "the ReID model: python main.py gallery build --rebuild"
            )
        return np.asarray(self._similarity(queries, view.matrix), dtype=np.float32)

    def match_batch(self, queries: np.ndarray, view: GalleryView) -> list[RecognitionResult]:
        """Match a batch of embeddings in a single matrix multiplication."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        if view.is_empty:
            return [
                RecognitionResult.unknown(self._config.unknown_label)
                for _ in range(queries.shape[0])
            ]
        scores = self.similarity_matrix(queries, view)
        return [self._decide(row, view) for row in scores]

    def match(self, query: np.ndarray, view: GalleryView) -> RecognitionResult:
        """Match a single embedding."""
        return self.match_batch(np.asarray(query).reshape(1, -1), view)[0]

    def _decide(self, scores: np.ndarray, view: GalleryView) -> RecognitionResult:
        """Apply the acceptance rules to one row of the similarity matrix."""
        order = np.argsort(-scores)
        best_index = int(order[0])
        best_score = float(scores[best_index])
        best_id = view.ids[best_index]

        runner_up_id: str | None = None
        runner_up_score = 0.0
        if len(order) > 1:
            runner_index = int(order[1])
            runner_up_id = view.ids[runner_index]
            runner_up_score = float(scores[runner_index])

        top_k = {
            view.ids[int(i)]: round(float(scores[int(i)]), 6)
            for i in order[: self._config.top_k]
        }
        identity = view.identities.get(best_id)
        common = {
            "similarity": best_score,
            "runner_up_id": runner_up_id,
            "runner_up_similarity": runner_up_score,
            "all_similarities": top_k,
        }

        # Open-set rejection: below the threshold the person is Unknown, full stop.
        if best_score < self._config.recognition_threshold:
            return RecognitionResult(
                status=RecognitionStatus.UNKNOWN,
                identity_name=self._config.unknown_label,
                **common,
            )

        # Ambiguity guard: two gallery identities too close to separate.
        margin = self._config.ambiguity_margin
        if margin > 0.0 and runner_up_id is not None and (best_score - runner_up_score) < margin:
            logger.debug(
                "Match rejected as ambiguous",
                extra={
                    "best": best_id,
                    "runner_up": runner_up_id,
                    "margin": round(best_score - runner_up_score, 4),
                    "required": margin,
                },
            )
            return RecognitionResult(
                status=RecognitionStatus.REJECTED,
                identity_name=self._config.unknown_label,
                **common,
            )

        status = (
            RecognitionStatus.RECOGNIZED
            if best_score >= self._config.high_confidence_threshold
            else RecognitionStatus.LOW_CONFIDENCE
        )
        return RecognitionResult(
            status=status,
            identity_id=best_id,
            identity_name=identity.name if identity else best_id,
            identity_title=identity.title if identity else "",
            **common,
        )

    def score_pairs(self, queries: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Row-wise similarity between two aligned sets (used by calibration)."""
        q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        r = np.atleast_2d(np.asarray(references, dtype=np.float32))
        if q.shape != r.shape:
            raise ValueError(f"shape mismatch: {q.shape} vs {r.shape}")
        if self._config.metric is SimilarityMetric.COSINE:
            return np.sum(l2_normalize(q, axis=1) * l2_normalize(r, axis=1), axis=1)
        return np.array(
            [float(self._similarity(q[i : i + 1], r[i : i + 1])[0, 0]) for i in range(q.shape[0])],
            dtype=np.float32,
        )


def build_gallery_view(
    identities: Sequence[PersonIdentity], *, normalize: bool = True
) -> GalleryView:
    """Build a matching snapshot from enabled identities that have embeddings."""
    usable = [i for i in identities if i.enabled and i.embedding is not None]
    usable.sort(key=lambda identity: identity.id)
    if not usable:
        return GalleryView(np.zeros((0, 0), dtype=np.float32), [], {})

    dimensions = {int(i.embedding.shape[-1]) for i in usable}  # type: ignore[union-attr]
    if len(dimensions) > 1:
        raise ValueError(
            f"gallery contains mixed embedding dimensions {sorted(dimensions)}; "
            "rebuild it after changing the ReID model"
        )
    matrix = np.stack([np.asarray(i.embedding, dtype=np.float32).reshape(-1) for i in usable])
    if normalize:
        matrix = l2_normalize(matrix, axis=1)
    return GalleryView(matrix, [i.id for i in usable], {i.id: i for i in usable})
