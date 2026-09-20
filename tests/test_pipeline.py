"""Tests for pipeline clustering logic.

These tests exercise the static clustering method directly without
importing heavy dependencies (torch, speechbrain).  The full pipeline
integration test requires ``pip install -r requirements.txt``.
"""

import numpy as np
import pytest


def _get_cluster_fn():
    """Import the cluster function, skip if torch is not installed."""
    try:
        from app.pipeline import VoicePipeline
        return VoicePipeline._cluster
    except ImportError:
        pytest.skip("torch/speechbrain not installed — skipping pipeline tests")


def test_cluster_groups_similar_embeddings() -> None:
    cluster = _get_cluster_fn()
    similarities = np.array(
        [
            [1.0, 0.90, 0.20],
            [0.90, 1.0, 0.25],
            [0.20, 0.25, 1.0],
        ]
    )
    labels = cluster(similarities, threshold=0.75)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_cluster_single_clip() -> None:
    cluster = _get_cluster_fn()
    similarities = np.array([[1.0]])
    labels = cluster(similarities, threshold=0.75)
    assert len(labels) == 1
    assert labels[0] == 0


def test_cluster_all_different() -> None:
    cluster = _get_cluster_fn()
    similarities = np.array(
        [
            [1.0, 0.10, 0.05],
            [0.10, 1.0, 0.15],
            [0.05, 0.15, 1.0],
        ]
    )
    labels = cluster(similarities, threshold=0.75)
    assert len(set(labels)) == 3
