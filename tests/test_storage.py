"""Tests for JSON speaker profile storage."""

import json
from pathlib import Path

import numpy as np

from app.storage import (
    SpeakerProfile,
    compute_centroid,
    delete_profile,
    enroll_speaker,
    load_all_profiles,
    load_profile,
)


def test_enroll_creates_profile(tmp_path: Path) -> None:
    embedding = np.array([1.0, 0.0, 0.0])
    profile = enroll_speaker(
        profiles_dir=tmp_path,
        name="Alice",
        embedding=embedding,
        source_file="alice.wav",
        duration_seconds=5.0,
    )
    assert profile.speaker_id.startswith("spk_")
    assert profile.name == "Alice"
    assert profile.enrollment_count == 1
    # File should exist
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_enroll_leaves_no_temp_files(tmp_path: Path) -> None:
    """Profiles are written via a temporary file and swapped in; the temporary
    file must not survive, or load_all_profiles would see partial writes."""
    profile = enroll_speaker(tmp_path, "Alice", np.array([1.0, 0.0]), "a.wav", 5.0)
    enroll_speaker(tmp_path, "Alice", np.array([0.9, 0.1]), "b.wav", 5.0,
                   speaker_id=profile.speaker_id)

    assert list(tmp_path.glob("*.tmp")) == []
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert load_profile(tmp_path, profile.speaker_id).enrollment_count == 2


def test_enroll_adds_to_existing(tmp_path: Path) -> None:
    emb1 = np.array([1.0, 0.0, 0.0])
    profile1 = enroll_speaker(
        tmp_path, "Bob", emb1, "bob1.wav", 3.0
    )
    emb2 = np.array([0.9, 0.1, 0.0])
    emb2 = emb2 / np.linalg.norm(emb2)
    profile2 = enroll_speaker(
        tmp_path, "Bob", emb2, "bob2.wav", 4.0,
        speaker_id=profile1.speaker_id,
    )
    assert profile2.enrollment_count == 2
    assert profile2.speaker_id == profile1.speaker_id


def test_centroid_is_normalized() -> None:
    embeddings = [
        {"embedding": [1.0, 0.0, 0.0]},
        {"embedding": [0.0, 1.0, 0.0]},
    ]
    centroid = compute_centroid(embeddings)
    norm = np.linalg.norm(centroid)
    assert abs(norm - 1.0) < 1e-6


def test_load_profile(tmp_path: Path) -> None:
    embedding = np.array([0.0, 1.0, 0.0])
    enrolled = enroll_speaker(tmp_path, "Carol", embedding, "carol.wav", 2.0)
    loaded = load_profile(tmp_path, enrolled.speaker_id)
    assert loaded is not None
    assert loaded.speaker_id == enrolled.speaker_id
    assert loaded.name == "Carol"


def test_load_all_profiles(tmp_path: Path) -> None:
    for name in ("Dave", "Eve", "Frank"):
        emb = np.random.default_rng(42).standard_normal(3)
        emb = emb / np.linalg.norm(emb)
        enroll_speaker(tmp_path, name, emb, f"{name.lower()}.wav", 3.0)
    profiles = load_all_profiles(tmp_path)
    assert len(profiles) == 3


def test_delete_profile(tmp_path: Path) -> None:
    embedding = np.array([1.0, 0.0])
    profile = enroll_speaker(tmp_path, "Gina", embedding, "gina.wav", 1.0)
    assert delete_profile(tmp_path, profile.speaker_id)
    assert load_profile(tmp_path, profile.speaker_id) is None


def test_delete_nonexistent_returns_false(tmp_path: Path) -> None:
    assert not delete_profile(tmp_path, "spk_nobody")


def test_json_roundtrip(tmp_path: Path) -> None:
    embedding = np.array([0.5, 0.5, 0.5, 0.5])
    embedding = embedding / np.linalg.norm(embedding)
    enrolled = enroll_speaker(tmp_path, "Helen", embedding, "helen.wav", 6.0)

    # Read raw JSON
    files = list(tmp_path.glob("*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    roundtripped = SpeakerProfile.from_dict(data)

    assert roundtripped.speaker_id == enrolled.speaker_id
    np.testing.assert_array_almost_equal(roundtripped.centroid, enrolled.centroid)


def test_unique_ids_for_same_name(tmp_path: Path) -> None:
    emb = np.array([1.0, 0.0])
    p1 = enroll_speaker(tmp_path, "Same Name", emb, "a.wav", 1.0)
    p2 = enroll_speaker(tmp_path, "Same Name", emb, "b.wav", 1.0)
    assert p1.speaker_id != p2.speaker_id
