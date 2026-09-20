"""Tests for scoring module — sigmoid confidence and matching logic."""

import math

import numpy as np

from app.scoring import (
    cosine_similarity,
    cosine_similarity_matrix,
    sigmoid_confidence,
    match_against_profiles,
    compute_pairwise_results,
)
from app.storage import SpeakerProfile


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == -1.0

    def test_clips_to_valid_range(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


class TestSimilarityMatrix:
    def test_diagonal_is_one(self) -> None:
        embeddings = np.eye(3, dtype=np.float64)
        matrix = cosine_similarity_matrix(embeddings)
        np.testing.assert_array_almost_equal(np.diag(matrix), [1.0, 1.0, 1.0])

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((4, 128))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        matrix = cosine_similarity_matrix(embeddings)
        np.testing.assert_array_almost_equal(matrix, matrix.T)


class TestSigmoidConfidence:
    def test_at_threshold_is_half(self) -> None:
        conf = sigmoid_confidence(0.75, threshold=0.75)
        assert abs(conf - 0.5) < 0.01  # should be ~0.50 at the boundary

    def test_far_above_threshold(self) -> None:
        conf = sigmoid_confidence(0.95, threshold=0.75)
        assert conf > 0.95  # very high confidence

    def test_far_below_threshold(self) -> None:
        conf = sigmoid_confidence(0.30, threshold=0.75)
        assert conf > 0.95  # very high confidence (that it's DIFFERENT)

    def test_always_in_range(self) -> None:
        for sim in (-1.0, -0.5, 0.0, 0.3, 0.7, 0.75, 0.8, 0.95, 1.0):
            conf = sigmoid_confidence(sim, threshold=0.75)
            assert 0.5 <= conf <= 1.0, f"sim={sim}, conf={conf}"

    def test_steepness_effect(self) -> None:
        """Higher steepness should give more extreme confidences."""
        conf_low = sigmoid_confidence(0.85, threshold=0.75, steepness=5.0)
        conf_high = sigmoid_confidence(0.85, threshold=0.75, steepness=30.0)
        assert conf_high > conf_low


class TestMatchAgainstProfiles:
    def _make_profile(
        self,
        speaker_id: str,
        centroid: list[float],
        enrollments: list[list[float]] | None = None,
    ) -> SpeakerProfile:
        return SpeakerProfile(
            speaker_id=speaker_id,
            name=speaker_id,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            embeddings=[{"embedding": embedding} for embedding in (enrollments or [])],
            centroid=centroid,
            enrollment_count=1,
        )

    def test_best_match_is_first(self) -> None:
        query = np.array([1.0, 0.0, 0.0])
        profiles = [
            self._make_profile("alice", [0.0, 1.0, 0.0]),
            self._make_profile("bob", [0.9, 0.1, 0.0]),
        ]
        # Normalize bob's centroid
        profiles[1].centroid = profiles[1].centroid / np.linalg.norm(profiles[1].centroid)

        matches = match_against_profiles(query, profiles, threshold=0.75)
        assert matches[0]["speaker_id"] == "bob"
        assert matches[0]["similarity"] > matches[1]["similarity"]
        assert matches[0]["rank"] == 1
        assert matches[1]["rank"] == 2

    def test_empty_profiles(self) -> None:
        query = np.array([1.0, 0.0])
        matches = match_against_profiles(query, [], threshold=0.75)
        assert matches == []

    def test_top_n_limits_results(self) -> None:
        query = np.array([1.0, 0.0])
        profiles = [self._make_profile(f"spk_{i}", [1.0, 0.0]) for i in range(10)]
        matches = match_against_profiles(query, profiles, threshold=0.75, top_n=3)
        assert len(matches) == 3

    def test_combines_centroid_and_individual_enrollments(self) -> None:
        query = np.array([1.0, 0.0])
        profile = self._make_profile(
            "alice",
            [0.8, 0.6],
            enrollments=[[1.0, 0.0], [0.6, 0.8], [0.8, 0.6]],
        )

        match = match_against_profiles(query, [profile], threshold=0.75)[0]

        assert match["centroid_similarity"] == 0.8
        assert match["enrollment_top_similarity"] == 1.0
        assert match["enrollment_top3_similarity"] == 0.8
        assert match["similarity"] == 0.84
        assert match["match_percentage"] == 84.0
        assert match["confidence_percentage"] == round(match["confidence"] * 100, 2)


class TestPairwiseResults:
    def test_correct_number_of_pairs(self) -> None:
        embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
        clip_ids = ["c1", "c2", "c3"]
        _, comparisons = compute_pairwise_results(embeddings, clip_ids, threshold=0.75)
        # 3 clips → 3 pairs: (c1,c2), (c1,c3), (c2,c3)
        assert len(comparisons) == 3

    def test_same_speaker_detection(self) -> None:
        v = np.array([1.0, 0.0])
        embeddings = [v, v]
        clip_ids = ["c1", "c2"]
        _, comparisons = compute_pairwise_results(embeddings, clip_ids, threshold=0.75)
        assert comparisons[0]["same_speaker"] is True
        assert comparisons[0]["similarity"] == 1.0
