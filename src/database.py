"""PostgreSQL + pgvector storage.

One database holds everything: person records, the enrolled voice clips
(BYTEA, 16 kHz mono WAV), the voice embeddings (VECTOR) and the match log.
Because it is one store, an enrollment or a delete is a single transaction: it
either fully happens or leaves no trace.

Similarity is cosine: ``1 - (embedding <=> query)``. Embeddings are stored
L2-normalised, so this equals their inner product. A speaker's centroid is
``avg(embedding)``; cosine distance ignores its length, so it needs no
re-normalising.

Every embedding records the model that produced it (``model``). Only
embeddings from the current model (src/model_files.py) are searched:
embeddings from different models live in different vector spaces, and
comparing them yields meaningless scores without any error.

Search is an exact scan (no ANN index), so the true nearest speaker is never
missed.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    LargeBinary,
    Sequence as DbSequence,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    text,
    type_coerce,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import DATABASE_URL, EMBEDDING_DIM
from src.model_files import EMBEDDING_MODEL_ID

PERSON_FIELDS = (
    "name",
    "dob",
    "gender",
    "father_name",
    "crime_type",
    "crime_details",
    "police_station",
    "jail_name",
    "sentence",
    "status",
)


class Base(DeclarativeBase):
    pass


person_id_seq = DbSequence("person_id_seq", start=1, metadata=Base.metadata)


class Person(Base):
    __tablename__ = "persons"

    person_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    dob: Mapped[Optional[str]] = mapped_column(Text)
    gender: Mapped[Optional[str]] = mapped_column(Text)
    father_name: Mapped[Optional[str]] = mapped_column(Text)
    crime_type: Mapped[Optional[str]] = mapped_column(Text)
    crime_details: Mapped[Optional[str]] = mapped_column(Text)
    police_station: Mapped[Optional[str]] = mapped_column(Text)
    jail_name: Mapped[Optional[str]] = mapped_column(Text)
    sentence: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class VoiceClip(Base):
    """Enrolled audio, kept apart from embeddings so vector scans never drag
    audio bytes through memory."""

    __tablename__ = "voice_clips"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    person_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("persons.person_id", ondelete="CASCADE"), index=True
    )
    audio_data: Mapped[bytes] = mapped_column(LargeBinary)
    content_type: Mapped[str] = mapped_column(String(64), default="audio/wav")
    filename: Mapped[Optional[str]] = mapped_column(Text)
    duration_seconds: Mapped[float] = mapped_column(Float)
    speech_seconds: Mapped[float] = mapped_column(Float)
    # "enroll" or "confirm": how the clip reached the record.
    source: Mapped[str] = mapped_column(String(16), default="enroll")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VoiceEmbedding(Base):
    """One embedding per clip. ``clip_id`` is NULL only for embeddings imported
    from the old JSON profiles, whose audio was never kept."""

    __tablename__ = "voice_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    person_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("persons.person_id", ondelete="CASCADE"), index=True
    )
    clip_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("voice_clips.id", ondelete="CASCADE"), unique=True
    )
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM))
    # The model that produced the embedding; see src/model_files.py.
    model: Mapped[str] = mapped_column(String(128), default=EMBEDDING_MODEL_ID, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MatchLog(Base):
    """Audit trail. person_id is deliberately not a foreign key: the log must
    outlive the person record it refers to."""

    __tablename__ = "match_logs"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    person_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    name: Mapped[Optional[str]] = mapped_column(Text)
    similarity: Mapped[float] = mapped_column(Float)
    matched: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(32), default="success")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# ----------------------------------------------------------------- engine --

_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def _normalise_url(url: str) -> str:
    # Accept plain postgres:// URLs and use the pure-Python pg8000 driver,
    # which has no compiled extension for Windows application control to block.
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+pg8000://" + url[len(prefix):]
    return url


def get_engine() -> Engine:
    """Created lazily so importing this module never opens a connection."""
    global _engine, _SessionFactory
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set. Add it to .env.")
        _engine = create_engine(_normalise_url(DATABASE_URL), pool_pre_ping=True)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """One transaction: commit on success, roll back on any exception."""
    get_engine()
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:
            raise RuntimeError(
                "Could not enable the pgvector extension. Install pgvector for this "
                "PostgreSQL server and make sure the DATABASE_URL user may create it."
            ) from exc
    Base.metadata.create_all(engine)
    _migrate(engine)


def _migrate(engine: Engine) -> None:
    """Bring databases created before ``voice_embeddings.model`` existed up to date.

    Until that column was added, only one model had ever been used, the one
    now pinned as EMBEDDING_MODEL_ID (its files were checked against the
    pinned checksums, and re-embedding stored clips reproduced the stored
    embeddings exactly), so existing rows are tagged with it.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE voice_embeddings ADD COLUMN IF NOT EXISTS model VARCHAR(128)"))
        conn.execute(text("UPDATE voice_embeddings SET model = :model WHERE model IS NULL"), {"model": EMBEDDING_MODEL_ID})
        conn.execute(text("ALTER TABLE voice_embeddings ALTER COLUMN model SET NOT NULL"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_voice_embeddings_model ON voice_embeddings (model)"))


# ---------------------------------------------------------------- helpers --

def normalise(embedding: Sequence[float]) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vec.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected a {EMBEDDING_DIM}-d embedding, got {vec.shape[0]}.")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("Cannot store a zero embedding.")
    return vec / norm


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def clip_ref(person_id: str, clip_id: int) -> str:
    """The URL path (without leading slash) the media endpoint serves."""
    return f"media/{person_id}/clips/{clip_id}.wav"


def _clip_dict(row) -> Dict[str, Any]:
    return {
        "clip_id": row.id,
        "url": clip_ref(row.person_id, row.id),
        "filename": row.filename,
        "duration_seconds": row.duration_seconds,
        "speech_seconds": row.speech_seconds,
        "source": row.source,
        "created_at": _iso(row.created_at),
    }


def _clips_by_person(session: Session, person_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Clip metadata per person, oldest first. Never loads the audio bytes."""
    rows = session.execute(
        select(
            VoiceClip.id,
            VoiceClip.person_id,
            VoiceClip.filename,
            VoiceClip.duration_seconds,
            VoiceClip.speech_seconds,
            VoiceClip.source,
            VoiceClip.created_at,
        )
        .where(VoiceClip.person_id.in_(person_ids))
        .order_by(VoiceClip.id)
    ).all()
    result: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in person_ids}
    for row in rows:
        result[row.person_id].append(_clip_dict(row))
    return result


def _embedding_counts(session: Session, person_ids: Sequence[str]) -> Dict[str, int]:
    rows = session.execute(
        select(VoiceEmbedding.person_id, func.count())
        .where(VoiceEmbedding.person_id.in_(person_ids))
        .group_by(VoiceEmbedding.person_id)
    ).all()
    counts = {pid: 0 for pid in person_ids}
    counts.update({pid: int(n) for pid, n in rows})
    return counts


def _person_dict(person: Person, clips: List[Dict[str, Any]], embedding_count: int) -> Dict[str, Any]:
    doc = {"person_id": person.person_id}
    doc.update({field: getattr(person, field) for field in PERSON_FIELDS})
    doc["clips"] = clips
    doc["embedding_count"] = embedding_count
    doc["created_at"] = _iso(person.created_at)
    return doc


def _log_dict(log: MatchLog) -> Dict[str, Any]:
    return {
        "person_id": log.person_id,
        "name": log.name,
        "similarity": log.similarity,
        "matched": log.matched,
        "status": log.status,
        "timestamp": _iso(log.timestamp),
    }


# ---------------------------------------------------------------- persons --

def get_next_person_id(session: Session) -> str:
    # Skip values already taken by a manually supplied person_id.
    while True:
        sequence = session.execute(select(person_id_seq.next_value())).scalar_one()
        person_id = f"VOICE{int(sequence):06d}"
        if session.get(Person, person_id) is None:
            return person_id


def person_exists(person_id: str, session: Optional[Session] = None) -> bool:
    if session is not None:
        return session.get(Person, person_id) is not None
    with session_scope() as s:
        return s.get(Person, person_id) is not None


def add_person(session: Session, person_id: str, person_data: Dict[str, Any]) -> None:
    session.add(Person(person_id=person_id, **{f: person_data.get(f) for f in PERSON_FIELDS}))
    # Flush the parent first: rows are inserted by table, and the children's
    # foreign keys need the person row to exist.
    session.flush()


def add_clip_with_embedding(session: Session, person_id: str, clip: Dict[str, Any], source: str) -> int:
    """Stage one clip and its embedding. ``clip`` carries ``wav_bytes``,
    ``embedding``, ``filename``, ``duration_seconds`` and ``speech_seconds``."""
    row = VoiceClip(
        person_id=person_id,
        audio_data=clip["wav_bytes"],
        filename=clip.get("filename"),
        duration_seconds=clip["duration_seconds"],
        speech_seconds=clip["speech_seconds"],
        source=source,
    )
    session.add(row)
    session.flush()  # assigns row.id for the embedding's foreign key
    session.add(VoiceEmbedding(person_id=person_id, clip_id=row.id, embedding=normalise(clip["embedding"])))
    return row.id


def add_person_with_clips(
    session: Session,
    person_id: str,
    person_data: Dict[str, Any],
    clips: Sequence[Dict[str, Any]],
) -> List[int]:
    """Stage a person plus their clips and embeddings in ``session``.

    Nothing is visible to other connections until the caller's transaction
    commits.
    """
    add_person(session, person_id, person_data)
    return [add_clip_with_embedding(session, person_id, clip, "enroll") for clip in clips]


def add_clip(person_id: str, clip: Dict[str, Any], source: str = "confirm") -> int:
    with session_scope() as session:
        if session.get(Person, person_id) is None:
            raise ValueError(f"Person {person_id} does not exist.")
        return add_clip_with_embedding(session, person_id, clip, source)


def find_person_by_id(person_id: str) -> Optional[Dict[str, Any]]:
    with session_scope() as session:
        person = session.get(Person, person_id)
        if person is None:
            return None
        clips = _clips_by_person(session, [person_id])[person_id]
        return _person_dict(person, clips, _embedding_counts(session, [person_id])[person_id])


def find_persons_by_ids(person_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    with session_scope() as session:
        persons = session.scalars(select(Person).where(Person.person_id.in_(person_ids))).all()
        ids = [p.person_id for p in persons]
        clips = _clips_by_person(session, ids)
        counts = _embedding_counts(session, ids)
        return {p.person_id: _person_dict(p, clips[p.person_id], counts[p.person_id]) for p in persons}


def get_all_persons() -> List[Dict[str, Any]]:
    with session_scope() as session:
        persons = session.scalars(select(Person).order_by(Person.created_at.desc())).all()
        ids = [p.person_id for p in persons]
        clips = _clips_by_person(session, ids)
        counts = _embedding_counts(session, ids)
        return [_person_dict(p, clips[p.person_id], counts[p.person_id]) for p in persons]


def delete_person(person_id: str) -> Dict[str, int]:
    """Delete a person; clips and embeddings go with them via ON DELETE CASCADE."""
    with session_scope() as session:
        if session.get(Person, person_id) is None:
            raise ValueError(f"Person {person_id} does not exist.")
        counts = {
            "removed_embeddings": session.scalar(
                select(func.count()).select_from(VoiceEmbedding).where(VoiceEmbedding.person_id == person_id)
            ),
            "removed_clips": session.scalar(
                select(func.count()).select_from(VoiceClip).where(VoiceClip.person_id == person_id)
            ),
        }
        counts["removed_persons"] = session.execute(
            delete(Person).where(Person.person_id == person_id)
        ).rowcount
        return counts


# ------------------------------------------------------------------ clips --

def get_clip_audio(person_id: str, clip_id: int) -> Optional[tuple[bytes, str]]:
    """The clip's audio, only if it belongs to ``person_id``."""
    with session_scope() as session:
        clip = session.get(VoiceClip, clip_id)
        if clip is None or clip.person_id != person_id:
            return None
        return clip.audio_data, clip.content_type


def iter_clip_ids() -> List[int]:
    with session_scope() as session:
        return list(session.scalars(select(VoiceClip.id).order_by(VoiceClip.id)))


def get_clip_for_reembed(clip_id: int) -> Optional[tuple[str, bytes]]:
    with session_scope() as session:
        clip = session.get(VoiceClip, clip_id)
        return (clip.person_id, clip.audio_data) if clip else None


def replace_clip_embedding(clip_id: int, embedding: Sequence[float]) -> None:
    with session_scope() as session:
        row = session.scalar(select(VoiceEmbedding).where(VoiceEmbedding.clip_id == clip_id))
        if row is None:
            clip = session.get(VoiceClip, clip_id)
            if clip is None:
                raise ValueError(f"Clip {clip_id} does not exist.")
            session.add(VoiceEmbedding(person_id=clip.person_id, clip_id=clip_id, embedding=normalise(embedding)))
        else:
            row.embedding = normalise(embedding)
            row.model = EMBEDDING_MODEL_ID


# ------------------------------------------------------------- similarity --

def count_embeddings() -> int:
    """Searchable embeddings: those from the current model."""
    with session_scope() as session:
        return int(
            session.scalar(
                select(func.count()).select_from(VoiceEmbedding).where(VoiceEmbedding.model == EMBEDDING_MODEL_ID)
            )
        )


def speaker_similarities(
    embedding: Sequence[float], person_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Per person: similarity to their centroid and to each of their clips.

    Scores are combined in Python (src/scoring.py), so this returns every
    person rather than a top-N: the exact scan reads each vector once either
    way.
    """
    query = normalise(embedding)
    similarity = 1 - VoiceEmbedding.embedding.cosine_distance(query)
    centroid = type_coerce(func.avg(VoiceEmbedding.embedding), Vector(EMBEDDING_DIM))
    stmt = select(
        VoiceEmbedding.person_id,
        (1 - centroid.cosine_distance(query)).label("centroid_similarity"),
        func.array_agg(similarity).label("similarities"),
    ).where(VoiceEmbedding.model == EMBEDDING_MODEL_ID).group_by(VoiceEmbedding.person_id)
    if person_id is not None:
        stmt = stmt.where(VoiceEmbedding.person_id == person_id)
    with session_scope() as session:
        rows = session.execute(stmt).all()
        return [
            {
                "person_id": row.person_id,
                "centroid_similarity": float(row.centroid_similarity),
                "similarities": [float(s) for s in row.similarities],
            }
            for row in rows
        ]


# ------------------------------------------------------------- match logs --

def create_match_log(
    person_id: Optional[str],
    name: Optional[str],
    similarity: float,
    matched: bool,
    status: str = "success",
) -> None:
    with session_scope() as session:
        session.add(
            MatchLog(person_id=person_id, name=name, similarity=float(similarity), matched=matched, status=status)
        )


def get_recent_match_logs(limit: int = 10) -> List[Dict[str, Any]]:
    with session_scope() as session:
        logs = session.scalars(select(MatchLog).order_by(MatchLog.timestamp.desc()).limit(limit)).all()
        return [_log_dict(log) for log in logs]


def get_match_logs_for_person(person_id: str) -> List[Dict[str, Any]]:
    with session_scope() as session:
        logs = session.scalars(
            select(MatchLog).where(MatchLog.person_id == person_id).order_by(MatchLog.timestamp.desc())
        ).all()
        return [_log_dict(log) for log in logs]


def get_match_stats() -> Dict[str, int]:
    with session_scope() as session:
        total, matched = session.execute(
            select(func.count(), func.count().filter(MatchLog.matched.is_(True)))
        ).one()
        return {
            "total_matches": int(total),
            "matched_count": int(matched),
            "unknown_count": int(total) - int(matched),
        }


def get_storage_stats() -> Dict[str, Any]:
    with session_scope() as session:
        clips, audio_bytes, audio_seconds = session.execute(
            select(
                func.count(),
                func.coalesce(func.sum(func.length(VoiceClip.audio_data)), 0),
                func.coalesce(func.sum(VoiceClip.duration_seconds), 0.0),
            ).select_from(VoiceClip)
        ).one()
        return {
            "persons": int(session.scalar(select(func.count()).select_from(Person))),
            "clips": int(clips),
            "embeddings": int(session.scalar(select(func.count()).select_from(VoiceEmbedding))),
            "audio_bytes": int(audio_bytes),
            "audio_seconds": round(float(audio_seconds), 1),
        }


# ---------------------------------------------------------------- health --

def integrity_report() -> Dict[str, Any]:
    """Foreign keys make orphans impossible; what can still be wrong is a
    person with no current-model embedding (unsearchable), a clip whose
    embedding is missing, or embeddings from another model (ignored by search
    until re-embedded). Embeddings without a clip come from the JSON import
    and are reported, not flagged: they work but cannot be re-embedded."""
    with session_scope() as session:
        person_ids = set(session.scalars(select(Person.person_id)))
        with_embeddings = set(
            session.scalars(
                select(VoiceEmbedding.person_id).where(VoiceEmbedding.model == EMBEDDING_MODEL_ID).distinct()
            )
        )
        other_models = {
            model: int(n)
            for model, n in session.execute(
                select(VoiceEmbedding.model, func.count())
                .where(VoiceEmbedding.model != EMBEDDING_MODEL_ID)
                .group_by(VoiceEmbedding.model)
            )
        }
        clips_without_embedding = list(
            session.scalars(
                select(VoiceClip.id)
                .outerjoin(VoiceEmbedding, VoiceEmbedding.clip_id == VoiceClip.id)
                .where(VoiceEmbedding.id.is_(None))
                .order_by(VoiceClip.id)
            )
        )
        embeddings_without_clip = int(
            session.scalar(
                select(func.count()).select_from(VoiceEmbedding).where(VoiceEmbedding.clip_id.is_(None))
            )
        )
        counts = {
            "persons": len(person_ids),
            "clips": int(session.scalar(select(func.count()).select_from(VoiceClip))),
            "embeddings": int(session.scalar(select(func.count()).select_from(VoiceEmbedding))),
        }

    without_embeddings = sorted(person_ids - with_embeddings)
    return {
        "counts": counts,
        "model": EMBEDDING_MODEL_ID,
        "persons_without_embeddings": without_embeddings,
        "clips_without_embedding": clips_without_embedding,
        "embeddings_from_other_models": other_models,
        "embeddings_without_clip": embeddings_without_clip,
        "consistent": not (without_embeddings or clips_without_embedding or other_models),
    }
