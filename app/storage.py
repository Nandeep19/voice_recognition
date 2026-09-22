"""JSON-based speaker profile storage.

Each enrolled speaker gets a JSON file in ``data/profiles/`` containing:
- metadata (name, timestamps, enrollment count)
- individual embeddings with source file info
- a centroid embedding (mean of all enrollments, L2-normalized)

The centroid is the primary vector used for matching.  It is more robust
than any single enrollment because it averages out per-session variability
(microphone differences, room acoustics, speaking style).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


_storage_lock = Lock()


def _write_profile(path: Path, profile: SpeakerProfile) -> None:
    """Write a profile atomically.

    Profiles hold irreplaceable enrollment data, so the JSON goes to a
    temporary file first and is then swapped in with os.replace. A crash
    mid-write leaves the previous profile intact instead of truncating it.
    """
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    os.replace(temp_path, path)


class SpeakerProfile:
    """In-memory representation of a stored speaker profile."""

    def __init__(
        self,
        speaker_id: str,
        name: str,
        created_at: str,
        updated_at: str,
        embeddings: list[dict[str, Any]],
        centroid: list[float],
        enrollment_count: int,
    ) -> None:
        self.speaker_id = speaker_id
        self.name = name
        self.created_at = created_at
        self.updated_at = updated_at
        self.embeddings = embeddings
        self.centroid = np.array(centroid, dtype=np.float64)
        self.enrollment_count = enrollment_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embeddings": self.embeddings,
            "centroid": self.centroid.tolist(),
            "enrollment_count": self.enrollment_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerProfile:
        return cls(
            speaker_id=data["speaker_id"],
            name=data["name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            embeddings=data["embeddings"],
            centroid=data["centroid"],
            enrollment_count=data["enrollment_count"],
        )


def _sanitize_id(name: str) -> str:
    """Create a filesystem-safe speaker ID from a name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"spk_{slug}" if slug else f"spk_{int(datetime.now(timezone.utc).timestamp())}"


def _profile_path(profiles_dir: Path, speaker_id: str) -> Path:
    """Get the JSON file path for a speaker profile."""
    # Prevent path traversal
    safe_id = re.sub(r"[^a-zA-Z0-9_]", "", speaker_id)
    return profiles_dir / f"{safe_id}.json"


def compute_centroid(embeddings: list[dict[str, Any]]) -> np.ndarray:
    """Compute L2-normalized centroid from a list of embedding records.

    The centroid averages out per-session variability, producing a more
    stable speaker representation than any single enrollment.
    """
    vectors = [np.array(e["embedding"], dtype=np.float64) for e in embeddings]
    if not vectors:
        raise ValueError("Cannot compute centroid from empty embedding list")
    centroid = np.mean(vectors, axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 1e-8 else centroid


def enroll_speaker(
    profiles_dir: Path,
    name: str,
    embedding: np.ndarray,
    source_file: str,
    duration_seconds: float,
    speaker_id: str | None = None,
) -> SpeakerProfile:
    """Enroll a new speaker or add an enrollment to an existing profile.

    If *speaker_id* is provided and a profile exists, the new embedding is
    appended and the centroid is recomputed.  Otherwise a new profile is
    created.

    Parameters
    ----------
    profiles_dir : Path
        Directory where profile JSON files live.
    name : str
        Human-readable speaker name.
    embedding : np.ndarray
        L2-normalized speaker embedding.
    source_file : str
        Original audio filename for audit trail.
    duration_seconds : float
        Duration of the speech used for this enrollment.
    speaker_id : str | None
        If provided, update an existing profile.  Otherwise auto-generate.

    Returns
    -------
    SpeakerProfile
        The created or updated profile.
    """
    now = datetime.now(timezone.utc).isoformat()

    embedding_record = {
        "embedding": embedding.tolist(),
        "source_file": source_file,
        "duration_seconds": round(duration_seconds, 3),
        "enrolled_at": now,
    }

    with _storage_lock:
        if speaker_id:
            path = _profile_path(profiles_dir, speaker_id)
            if path.exists():
                # Update existing profile
                data = json.loads(path.read_text(encoding="utf-8"))
                profile = SpeakerProfile.from_dict(data)
                profile.embeddings.append(embedding_record)
                profile.enrollment_count = len(profile.embeddings)
                profile.centroid = compute_centroid(profile.embeddings)
                profile.updated_at = now
                _write_profile(path, profile)
                return profile

        # Create new profile
        if not speaker_id:
            speaker_id = _sanitize_id(name)
            # Ensure unique ID
            base_id = speaker_id
            counter = 1
            while _profile_path(profiles_dir, speaker_id).exists():
                speaker_id = f"{base_id}_{counter}"
                counter += 1

        centroid = embedding.copy()
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm

        profile = SpeakerProfile(
            speaker_id=speaker_id,
            name=name,
            created_at=now,
            updated_at=now,
            embeddings=[embedding_record],
            centroid=centroid.tolist(),
            enrollment_count=1,
        )

        path = _profile_path(profiles_dir, speaker_id)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        _write_profile(path, profile)
        return profile


def load_profile(profiles_dir: Path, speaker_id: str) -> SpeakerProfile | None:
    """Load a single speaker profile by ID.  Returns None if not found."""
    path = _profile_path(profiles_dir, speaker_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SpeakerProfile.from_dict(data)


def load_all_profiles(profiles_dir: Path) -> list[SpeakerProfile]:
    """Load all speaker profiles from disk."""
    profiles: list[SpeakerProfile] = []
    if not profiles_dir.exists():
        return profiles
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            profiles.append(SpeakerProfile.from_dict(data))
        except (json.JSONDecodeError, KeyError):
            continue  # skip corrupted files
    return profiles


def delete_profile(profiles_dir: Path, speaker_id: str) -> bool:
    """Delete a speaker profile.  Returns True if it existed."""
    path = _profile_path(profiles_dir, speaker_id)
    if path.exists():
        with _storage_lock:
            path.unlink(missing_ok=True)
        return True
    return False
