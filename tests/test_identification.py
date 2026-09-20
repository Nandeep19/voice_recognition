"""Tests for identification decision rules."""

import asyncio
from types import SimpleNamespace

import pytest

from app.pipeline import VoicePipeline
from app.storage import SpeakerProfile


def _profile(speaker_id: str, similarity_vector: list[float]) -> SpeakerProfile:
    return SpeakerProfile(
        speaker_id=speaker_id,
        name=speaker_id,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        embeddings=[{"embedding": similarity_vector}],
        centroid=similarity_vector,
        enrollment_count=1,
    )


def test_close_top_matches_are_uncertain(monkeypatch) -> None:
    settings = SimpleNamespace(
        profiles_dir=None,
        speaker_similarity_threshold=0.75,
        identification_min_margin=0.05,
        sigmoid_steepness=15.0,
    )
    pipeline = VoicePipeline(settings, models=None)
    profiles = [_profile("alice", [0.80, 0.60]), _profile("bob", [0.78, 0.625779])]

    monkeypatch.setattr("app.pipeline.load_all_profiles", lambda _: profiles)

    async def fake_process(_file):
        import numpy as np
        return np.array([1.0, 0.0]), 4.0, 3.0

    monkeypatch.setattr(pipeline, "_process_single_clip", fake_process)
    result = asyncio.run(pipeline.identify(file=None))

    assert result.decision == "uncertain"
    assert result.score_margin == pytest.approx(0.02, abs=0.001)
    assert result.top_match is not None
    assert result.top_match.same_person is False
