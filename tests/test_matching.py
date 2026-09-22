"""Similarity maths and open-set identity decisions."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.schema import MatchingConfig, SimilarityMetric
from src.core.types import PersonIdentity, RecognitionStatus
from src.identity.matcher import (
    IdentityMatcher,
    build_gallery_view,
    cosine_similarity,
    euclidean_similarity,
)
from src.reid.encoder import l2_normalize


def identity(identity_id: str, vector: np.ndarray, name: str | None = None) -> PersonIdentity:
    return PersonIdentity(
        id=identity_id,
        name=name or identity_id.replace("_", " ").title(),
        title="Tester",
        embedding=l2_normalize(np.asarray(vector, dtype=np.float32)),
        embedding_dimension=len(vector),
    )


@pytest.fixture
def view():
    return build_gallery_view(
        [
            identity("alice", [1.0, 0.0, 0.0, 0.0]),
            identity("bob", [0.0, 1.0, 0.0, 0.0]),
            identity("carol", [0.0, 0.0, 1.0, 0.0]),
        ]
    )


class TestSimilarityMaths:
    def test_cosine_of_identical_vectors_is_one(self) -> None:
        vector = np.array([[0.3, 0.4, 0.5]], dtype=np.float32)
        assert cosine_similarity(vector, vector)[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_cosine_of_orthogonal_vectors_is_zero(self) -> None:
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        assert cosine_similarity(a, b)[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_cosine_is_scale_invariant(self) -> None:
        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        assert cosine_similarity(a, a * 17.0)[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_normalisation_leaves_zero_vectors_at_zero(self) -> None:
        # A degenerate crop must not accidentally resemble an identity.
        result = l2_normalize(np.zeros((1, 8), dtype=np.float32))
        assert np.all(result == 0.0)

    def test_normalised_vectors_have_unit_norm(self) -> None:
        rng = np.random.default_rng(0)
        normalized = l2_normalize(rng.normal(size=(5, 16)).astype(np.float32), axis=1)
        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-5)

    def test_euclidean_similarity_is_bounded_and_ordered(self) -> None:
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        same = euclidean_similarity(a, a)[0, 0]
        opposite = euclidean_similarity(a, -a)[0, 0]
        assert same == pytest.approx(1.0, abs=1e-5)
        assert opposite == pytest.approx(0.0, abs=1e-5)

    def test_dot_product_equals_cosine_for_normalised_input(self, view) -> None:
        query = l2_normalize(np.array([0.9, 0.3, 0.1, 0.0], dtype=np.float32))
        matcher = IdentityMatcher(MatchingConfig())
        scores = matcher.similarity_matrix(query.reshape(1, -1), view)[0]
        assert scores == pytest.approx(query @ view.matrix.T, abs=1e-6)


class TestOpenSetDecisions:
    def test_a_registered_person_is_recognized(self, view) -> None:
        matcher = IdentityMatcher(MatchingConfig(recognition_threshold=0.6,
                                                 high_confidence_threshold=0.8))
        result = matcher.match(np.array([0.95, 0.1, 0.05, 0.0], dtype=np.float32), view)
        assert result.status is RecognitionStatus.RECOGNIZED
        assert result.identity_id == "alice"
        assert result.identity_name == "Alice"
        assert result.identity_title == "Tester"

    def test_an_unregistered_person_is_unknown_not_nearest_neighbour(self, view) -> None:
        """The core open-set guarantee: no forcing onto the closest identity."""
        matcher = IdentityMatcher(MatchingConfig(recognition_threshold=0.6,
                                                 high_confidence_threshold=0.8))
        # Closest to 'alice', but nowhere near the threshold.
        result = matcher.match(np.array([0.4, 0.3, 0.3, 0.85], dtype=np.float32), view)
        assert result.status is RecognitionStatus.UNKNOWN
        assert result.identity_id is None
        assert result.identity_name == "Unknown"
        # The best candidate is still reported for diagnosis.
        assert result.runner_up_id is not None

    def test_scores_between_the_thresholds_are_low_confidence(self, view) -> None:
        matcher = IdentityMatcher(MatchingConfig(recognition_threshold=0.5,
                                                 high_confidence_threshold=0.9))
        result = matcher.match(np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32), view)
        assert result.status is RecognitionStatus.LOW_CONFIDENCE
        assert result.is_recognized

    def test_threshold_boundary_is_inclusive(self, view) -> None:
        matcher = IdentityMatcher(MatchingConfig(recognition_threshold=0.5,
                                                 high_confidence_threshold=0.5))
        query = l2_normalize(np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))
        # cos with 'alice' is exactly 1/sqrt(2) ~ 0.7071 > 0.5
        assert matcher.match(query, view).is_recognized

        strict = IdentityMatcher(MatchingConfig(recognition_threshold=0.71,
                                                high_confidence_threshold=0.9))
        assert not strict.match(query, view).is_recognized

    def test_ambiguous_top_two_are_rejected_when_a_margin_is_set(self, view) -> None:
        matcher = IdentityMatcher(
            MatchingConfig(
                recognition_threshold=0.5, high_confidence_threshold=0.9, ambiguity_margin=0.2
            )
        )
        # Equidistant between alice and bob.
        result = matcher.match(np.array([0.71, 0.70, 0.0, 0.0], dtype=np.float32), view)
        assert result.status is RecognitionStatus.REJECTED
        assert result.identity_id is None

    def test_an_empty_gallery_yields_unknown_not_an_error(self) -> None:
        matcher = IdentityMatcher(MatchingConfig())
        empty = build_gallery_view([])
        assert empty.is_empty
        result = matcher.match(np.ones(4, dtype=np.float32), empty)
        assert result.status is RecognitionStatus.UNKNOWN

    def test_top_k_similarities_are_reported(self, view) -> None:
        matcher = IdentityMatcher(MatchingConfig(top_k=2))
        result = matcher.match(np.array([0.9, 0.4, 0.1, 0.0], dtype=np.float32), view)
        assert len(result.all_similarities) == 2
        assert "alice" in result.all_similarities


class TestBatchAndGalleryView:
    def test_batch_matching_matches_single_matching(self, view) -> None:
        matcher = IdentityMatcher(MatchingConfig(recognition_threshold=0.5,
                                                 high_confidence_threshold=0.8))
        queries = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.9]],
            dtype=np.float32,
        )
        batch = matcher.match_batch(queries, view)
        singles = [matcher.match(q, view) for q in queries]
        assert [r.identity_id for r in batch] == [r.identity_id for r in singles]
        assert [r.identity_id for r in batch] == ["alice", "bob", None]

    def test_gallery_view_is_matrix_shaped_for_vectorised_search(self, view) -> None:
        assert view.matrix.shape == (3, 4)
        assert view.ids == ["alice", "bob", "carol"]
        assert np.allclose(np.linalg.norm(view.matrix, axis=1), 1.0, atol=1e-5)

    def test_disabled_identities_are_excluded(self) -> None:
        enabled = identity("on", [1.0, 0.0])
        disabled = identity("off", [0.0, 1.0])
        disabled.enabled = False
        view = build_gallery_view([enabled, disabled])
        assert view.ids == ["on"]

    def test_identities_without_embeddings_are_excluded(self) -> None:
        pending = PersonIdentity(id="pending", name="Pending")
        view = build_gallery_view([identity("ok", [1.0, 0.0]), pending])
        assert view.ids == ["ok"]

    def test_dimension_mismatch_is_actionable(self, view) -> None:
        matcher = IdentityMatcher(MatchingConfig())
        with pytest.raises(ValueError, match="Rebuild the gallery"):
            matcher.similarity_matrix(np.ones((1, 8), dtype=np.float32), view)

    def test_mixed_gallery_dimensions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="mixed embedding dimensions"):
            build_gallery_view([identity("a", [1.0, 0.0]), identity("b", [1.0, 0.0, 0.0])])

    def test_metric_selection_is_honoured(self, view) -> None:
        matcher = IdentityMatcher(
            MatchingConfig(metric=SimilarityMetric.EUCLIDEAN, recognition_threshold=0.9,
                           high_confidence_threshold=0.95)
        )
        assert matcher.match(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), view).identity_id == "alice"


class TestScaling:
    def test_many_identities_are_matched_in_one_operation(self) -> None:
        """A vectorised search must stay correct as the gallery grows."""
        rng = np.random.default_rng(3)
        count, dimension = 500, 128
        vectors = rng.normal(size=(count, dimension)).astype(np.float32)
        people = [identity(f"person_{i:03d}", vectors[i]) for i in range(count)]
        view = build_gallery_view(people)
        assert view.matrix.shape == (count, dimension)

        matcher = IdentityMatcher(MatchingConfig(recognition_threshold=0.5,
                                                 high_confidence_threshold=0.9))
        target = view.ids.index("person_042")
        result = matcher.match(view.matrix[target], view)
        assert result.identity_id == "person_042"
        assert result.similarity == pytest.approx(1.0, abs=1e-4)
