"""Tests for scoring module — sigmoid confidence and matching logic."""

import numpy as np

from src.scoring import (
    cluster,
    compute_pairwise_results,
    cosine_similarity,
    cosine_similarity_matrix,
    rank_speakers,
    sigmoid_confidence,
)


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


def _row(person_id: str, query: np.ndarray, enrollments: list[list[float]]) -> dict:
    """What database.speaker_similarities returns for one person."""
    vectors = np.array(enrollments, dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    centroid = vectors.mean(axis=0)
    return {
        "person_id": person_id,
        "centroid_similarity": float(query @ centroid / np.linalg.norm(centroid)),
        "similarities": [float(query @ v) for v in vectors],
    }


class TestRankSpeakers:
    def test_best_match_is_first(self) -> None:
        query = np.array([1.0, 0.0, 0.0])
        rows = [_row("alice", query, [[0.0, 1.0, 0.0]]), _row("bob", query, [[0.9, 0.1, 0.0]])]

        matches = rank_speakers(rows, threshold=0.75)
        assert matches[0]["person_id"] == "bob"
        assert matches[0]["similarity"] > matches[1]["similarity"]
        assert matches[0]["rank"] == 1
        assert matches[1]["rank"] == 2

    def test_empty_rows(self) -> None:
        assert rank_speakers([], threshold=0.75) == []

    def test_top_n_limits_results(self) -> None:
        query = np.array([1.0, 0.0])
        rows = [_row(f"spk_{i}", query, [[1.0, 0.0]]) for i in range(10)]
        assert len(rank_speakers(rows, threshold=0.75, top_n=3)) == 3

    def test_combines_centroid_and_individual_enrollments(self) -> None:
        row = {"person_id": "alice", "centroid_similarity": 0.8, "similarities": [1.0, 0.6, 0.8]}

        match = rank_speakers([row], threshold=0.75)[0]

        assert match["centroid_similarity"] == 0.8
        assert match["enrollment_top_similarity"] == 1.0
        assert match["enrollment_top3_similarity"] == 0.8
        assert match["similarity"] == 0.84
        assert match["match_percentage"] == 84.0
        assert match["confidence_percentage"] == round(match["confidence"] * 100, 2)
        assert match["above_threshold"] is True

    def test_fewer_than_three_enrollments_drop_the_top3_term(self) -> None:
        """Two samples spread around the query: the centroid matches it better
        than either sample does. The score must not be dragged far below that
        centroid by averaging in the weaker sample."""
        query = np.array([1.0, 0.0, 0.0])
        match = rank_speakers([_row("alice", query, [[0.8, 0.6, 0.0], [0.8, -0.6, 0.0]])], threshold=0.75)[0]

        assert match["centroid_similarity"] == 1.0
        assert match["enrollment_top_similarity"] == 0.8
        # 0.70 * 1.0 + 0.30 * 0.8 — the three-sample formula would give 0.90.
        assert match["similarity"] == 0.94

    def test_single_enrollment_scores_as_its_own_centroid(self) -> None:
        query = np.array([1.0, 0.0])
        match = rank_speakers([_row("alice", query, [[1.0, 0.0]])], threshold=0.75)[0]

        assert match["similarity"] == 1.0
        assert match["match_percentage"] == 100.0

    def test_below_threshold_is_flagged(self) -> None:
        row = {"person_id": "alice", "centroid_similarity": 0.3, "similarities": [0.3]}
        assert rank_speakers([row], threshold=0.75)[0]["above_threshold"] is False


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


class TestCluster:
    def test_groups_similar_embeddings(self) -> None:
        similarities = np.array([[1.0, 0.90, 0.20], [0.90, 1.0, 0.25], [0.20, 0.25, 1.0]])
        labels = cluster(similarities, threshold=0.75)
        assert labels[0] == labels[1]
        assert labels[0] != labels[2]

    def test_single_clip(self) -> None:
        assert list(cluster(np.array([[1.0]]), threshold=0.75)) == [0]

    def test_all_different(self) -> None:
        similarities = np.array([[1.0, 0.10, 0.05], [0.10, 1.0, 0.15], [0.05, 0.15, 1.0]])
        assert len(set(cluster(similarities, threshold=0.75))) == 3
