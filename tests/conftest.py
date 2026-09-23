"""Shared fixtures.

Two tiers:

* Offline tests (always run): media endpoint, input validation, scoring,
  audio helpers. The database is monkeypatched per test and no connection is
  ever opened.
* Integration tests (``pg`` fixture): run against a real PostgreSQL+pgvector
  database named in TEST_DATABASE_URL, and are skipped when it is unset. The
  database name must contain "test"; every table is truncated per test.

The speaker model is never loaded: integration tests replace clip analysis
with deterministic fake embeddings (see ``voice_clip``).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
# Set before src.config is imported; load_dotenv never overrides an existing
# variable, so offline runs can never reach the real database from .env, and
# thresholds tuned in .env do not change what the tests expect.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
for _key, _value in {
    "VOICE_THRESHOLD": "0.75",
    "PERSON_MATCH_THRESHOLD": "0.75",
    "SEARCH_THRESHOLD": "0.60",
    "MATCH_MARGIN": "0.05",
    "DUPLICATE_THRESHOLD": "0.80",
    "CONFIRM_THRESHOLD": "0.75",
    "ENROLL_CONSISTENCY_THRESHOLD": "0.53",
    "MAX_ENROLL_CLIPS": "5",
}.items():
    os.environ[_key] = _value

EMBEDDING_DIM = 192


def voice_clip(speaker: int, take: int | None = None) -> bytes:
    """Fake upload bytes for ``speaker``. Different takes of one speaker embed
    ~0.92 similar to each other; different speakers ~0.0."""
    return f"voice:{speaker}".encode() if take is None else f"voice:{speaker}:{take}".encode()


def fake_embedding(data: bytes) -> np.ndarray:
    parts = data.decode().split(":")
    speaker = int(parts[1])
    vec = np.random.default_rng(speaker).standard_normal(EMBEDDING_DIM)
    if len(parts) > 2:
        vec = vec + 0.3 * np.random.default_rng(speaker * 1000 + int(parts[2])).standard_normal(EMBEDDING_DIM)
    return vec.astype(np.float32)


def fake_analyse(data: bytes, filename: str, min_speech_ms: int, check_voices: bool = False) -> dict:
    from src.audio import encode_wav

    return {
        "filename": filename,
        "embedding": fake_embedding(data),
        "wav_bytes": encode_wav(np.zeros(1600, dtype=np.float32)),
        "duration_seconds": 4.0,
        "speech_seconds": 3.5,
        "warnings": [],
    }


@pytest.fixture(scope="session")
def client():
    """TestClient whose startup does not touch the database."""
    from fastapi.testclient import TestClient
    import app as app_module

    app_module.initialize_system = lambda: None
    app_module.load_models = lambda: None
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def fake_voices(monkeypatch):
    """Deterministic clip analysis, no model and no FFmpeg."""
    import src.voice_service as vs

    monkeypatch.setattr(vs, "_analyse_clip", fake_analyse)
    return vs


@pytest.fixture(scope="session")
def _pg_schema():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set; skipping PostgreSQL integration tests.")
    if "test" not in TEST_DATABASE_URL.rsplit("/", 1)[-1]:
        pytest.exit("TEST_DATABASE_URL must point at a database whose name contains 'test'.")

    from src import database as db

    db.init_db()
    return db


@pytest.fixture
def pg(_pg_schema, fake_voices):
    """A clean database plus fake clip analysis."""
    from sqlalchemy import text

    db = _pg_schema
    with db.get_engine().begin() as conn:
        conn.execute(text("TRUNCATE persons, voice_clips, voice_embeddings, match_logs RESTART IDENTITY CASCADE"))
        conn.execute(text("ALTER SEQUENCE person_id_seq RESTART WITH 1"))
    return db
