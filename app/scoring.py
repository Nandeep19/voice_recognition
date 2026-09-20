"""Scoring — cosine similarity, sigmoid confidence, and top-N matching.

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

from app.storage import SpeakerProfile


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


def match_against_profiles(
    query_embedding: np.ndarray,
    profiles: list[SpeakerProfile],
    threshold: float,
    steepness: float = 15.0,
    top_n: int = 5,
) -> list[dict]:
    """Match a query embedding against all enrolled speaker profiles.

    Combines similarity to the profile centroid with similarities to the
    speaker's individual enrollment embeddings. This preserves the stability
    of a centroid while allowing a query to match a known session/style well.

    Parameters
    ----------
    query_embedding : np.ndarray
        L2-normalized embedding of the query audio.
    profiles : list[SpeakerProfile]
        All enrolled speaker profiles.
    threshold : float
        Cosine similarity threshold for same-speaker decision.
    steepness : float
        Sigmoid steepness for confidence scoring.
    top_n : int
        Number of top matches to return.

    Returns
    -------
    list[dict]
        Each dict has speaker metadata, raw scores, percentage scores, and
        rank. Sorted by similarity descending.
    """
    if not profiles:
        return []

    results: list[dict] = []
    for profile in profiles:
        centroid_sim = cosine_similarity(query_embedding, profile.centroid)
        enrollment_sims = sorted(
            (
                cosine_similarity(
                    query_embedding,
                    np.asarray(record["embedding"], dtype=np.float64),
                )
                for record in profile.embeddings
                if "embedding" in record
            ),
            reverse=True,
        )

        # Older or externally-created profiles may contain only a centroid.
        if enrollment_sims:
            top_sim = enrollment_sims[0]
            top3_sim = float(np.mean(enrollment_sims[:3]))
        else:
            top_sim = centroid_sim
            top3_sim = centroid_sim

        sim = 0.50 * centroid_sim + 0.30 * top3_sim + 0.20 * top_sim
        conf = sigmoid_confidence(sim, threshold, steepness)
        sim = round(sim, 4)
        conf = sigmoid_confidence(sim, threshold, steepness)
        results.append({
            "speaker_id": profile.speaker_id,
            "name": profile.name,
            "similarity": sim,
            "match_percentage": round(max(0.0, min(1.0, sim)) * 100, 2),
            "confidence": conf,
            "confidence_percentage": round(conf * 100, 2),
            "same_person": bool(sim >= threshold),
            "enrollment_count": profile.enrollment_count,
            "centroid_similarity": round(centroid_sim, 4),
            "enrollment_top_similarity": round(top_sim, 4),
            "enrollment_top3_similarity": round(top3_sim, 4),
        })

    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    ranked = results[:top_n]
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
