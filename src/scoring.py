"""Scoring — combined speaker score, sigmoid confidence, and pairwise comparison.

The confidence score uses a sigmoid function centered at the decision
threshold.  Unlike a linear interpolation, sigmoid gives:
- ~50% at the threshold boundary (maximum uncertainty)
- rapid rise to 95%+ for clearly matching pairs
- rapid drop to <5% for clearly non-matching pairs

This better reflects the actual reliability of the decision.
"""

from __future__ import annotations

import math

import numpy as np



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors.

    Since ECAPA-TDNN embeddings are already L2-normalized, this is just
    a dot product.  We clip to [-1, 1] to handle numerical imprecision.
    """
    sim = float(np.dot(a, b))
    return max(-1.0, min(1.0, sim))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (N, D) — N embeddings of dimension D, each L2-normalized.

    Returns
    -------
    np.ndarray
        Shape (N, N) — symmetric similarity matrix with 1s on diagonal.
    """
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)


def sigmoid_confidence(similarity: float, threshold: float, steepness: float = 15.0) -> float:
    """Compute calibrated confidence using a sigmoid function.

    The sigmoid is centered at *threshold*, so:
    - similarity == threshold  →  confidence ≈ 0.50  (maximum uncertainty)
    - similarity >> threshold  →  confidence → 1.00  (high certainty: same speaker)
    - similarity << threshold  →  confidence → 1.00  (high certainty: different speaker)

    The returned value always represents certainty in the *decision*
    (same or different), not certainty that it's the same speaker.

    Parameters
    ----------
    similarity : float
        Cosine similarity in [-1, 1].
    threshold : float
        Decision boundary.
    steepness : float
        Controls how quickly confidence rises away from threshold.
        Higher = sharper transition.  Default 15 works well for ECAPA-TDNN.

    Returns
    -------
    float
        Confidence in [0.5, 1.0].  Values below 0.5 are meaningless for
        a binary decision, so we clamp there.
    """
    # Distance from threshold (positive = above, negative = below)
    delta = similarity - threshold

    # Sigmoid gives probability of "same speaker"
    # For the confidence in the decision, we take max(p, 1-p)
    try:
        p_same = 1.0 / (1.0 + math.exp(-steepness * delta))
    except OverflowError:
        p_same = 0.0 if delta < 0 else 1.0

    # Confidence in the decision = how far p_same is from 0.5
    # Map [0.5, 1.0] and [0.0, 0.5] both to [0.5, 1.0]
    confidence = max(p_same, 1.0 - p_same)
    return round(confidence, 4)


def combine_scores(centroid_similarity: float, similarities: list[float]) -> dict:
    """One score for a speaker from their centroid and individual clips.

    Combines similarity to the speaker's centroid with similarities to their
    individual enrollment embeddings. This keeps the stability of a centroid
    while letting a query match one known session or speaking style well.
    """
    ranked = sorted(similarities, reverse=True)
    if ranked:
        top_sim = ranked[0]
        top3_sim = float(np.mean(ranked[:3]))
    else:
        top_sim = top3_sim = centroid_similarity

    if len(ranked) >= 3:
        sim = 0.50 * centroid_similarity + 0.30 * top3_sim + 0.20 * top_sim
    else:
        # With fewer than three samples "top 3" is just the mean of all of
        # them, which duplicates the centroid while letting one weak sample
        # outvote it — a query could score below the centroid it matched.
        # Drop that term and keep the original centroid:best weighting.
        sim = 0.70 * centroid_similarity + 0.30 * top_sim
    return {
        "similarity": round(sim, 4),
        "centroid_similarity": round(centroid_similarity, 4),
        "enrollment_top_similarity": round(top_sim, 4),
        "enrollment_top3_similarity": round(top3_sim, 4),
        "enrollment_count": len(ranked),
    }


def rank_speakers(
    rows: list[dict],
    threshold: float,
    steepness: float = 15.0,
    top_n: int | None = None,
) -> list[dict]:
    """Score, rank and annotate speakers.

    ``rows`` are ``{"person_id", "centroid_similarity", "similarities"}`` as
    returned by ``database.speaker_similarities``. The result is sorted best
    first, with ``rank``, ``match_percentage`` and a decision ``confidence``.
    """
    results: list[dict] = []
    for row in rows:
        scored = combine_scores(row["centroid_similarity"], row["similarities"])
        sim = scored["similarity"]
        conf = sigmoid_confidence(sim, threshold, steepness)
        results.append({
            "person_id": row["person_id"],
            **scored,
            "match_percentage": round(max(0.0, min(1.0, sim)) * 100, 2),
            "confidence": conf,
            "confidence_percentage": round(conf * 100, 2),
            "above_threshold": bool(sim >= threshold),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    ranked = results if top_n is None else results[:top_n]
    for rank, result in enumerate(ranked, start=1):
        result["rank"] = rank
    return ranked


def compute_pairwise_results(
    embeddings: list[np.ndarray],
    clip_ids: list[str],
    threshold: float,
    steepness: float = 15.0,
) -> tuple[np.ndarray, list[dict]]:
    """Compute pairwise similarity and comparison results for a batch of clips.

    Returns
    -------
    tuple
        - similarities matrix (N x N)
        - list of pairwise comparison dicts
    """
    matrix = np.vstack(embeddings)
    similarities = cosine_similarity_matrix(matrix)

    comparisons: list[dict] = []
    for i in range(len(clip_ids)):
        for j in range(i + 1, len(clip_ids)):
            sim = float(similarities[i, j])
            conf = sigmoid_confidence(sim, threshold, steepness)
            comparisons.append({
                "clip_a": clip_ids[i],
                "clip_b": clip_ids[j],
                "similarity": round(sim, 4),
                "confidence": conf,
                "same_speaker": bool(sim >= threshold),
            })

    return similarities, comparisons


def cluster(similarities: np.ndarray, threshold: float) -> np.ndarray:
    """Group clips by speaker: agglomerative clustering on cosine distance."""
    if len(similarities) == 1:
        return np.array([0])
    from sklearn.cluster import AgglomerativeClustering

    distances = np.clip(1.0 - similarities, 0.0, 2.0)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - threshold,
    )
    return model.fit_predict(distances)
