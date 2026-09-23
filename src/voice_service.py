"""Core enrollment and matching service.

Audio → FFmpeg (mono 16 kHz) → Silero VAD → ECAPA-TDNN embedding → pgvector.

Every enrolled clip is stored in the database next to its embedding. Probe
clips (match, verify, search, compare) are analysed and discarded; only the
match log records that a lookup happened.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy.exc import IntegrityError

from src import database as db
from src.audio import TARGET_SAMPLE_RATE, decode_audio, encode_wav, load_audio
from src.config import (
    CONFIRM_THRESHOLD,
    DUPLICATE_THRESHOLD,
    ENROLL_CONSISTENCY_THRESHOLD,
    ENROLLMENT_MIN_SPEECH_MS,
    MATCH_MARGIN,
    MAX_CLIP_SECONDS,
    MAX_ENROLL_CLIPS,
    MAX_UPLOAD_MB,
    MIN_SPEECH_MS,
    PERSON_ID_PATTERN,
    PERSON_MATCH_THRESHOLD,
    SEARCH_THRESHOLD,
    SIGMOID_STEEPNESS,
    VAD_MERGE_GAP_MS,
    VAD_MIN_SEGMENT_MS,
    VAD_PAD_MS,
    VAD_THRESHOLD,
    VOICE_THRESHOLD,
)
from src.embed import embed, embed_segments
from src.scoring import cluster, compute_pairwise_results, cosine_similarity, rank_speakers
from src.vad import extract_speech

_PERSON_ID_RE = re.compile(PERSON_ID_PATTERN)
_MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
_MAX_COMPARE_CLIPS = 10

# (filename, bytes) of one uploaded file.
Upload = Tuple[str, bytes]


def _safe_filename(filename: Optional[str]) -> str:
    return Path(filename or "clip").name[:200] or "clip"


def _decode_clip(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise ValueError("Empty audio data provided.")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise ValueError(f"Audio file is larger than the {MAX_UPLOAD_MB:g} MB limit.")
    audio = decode_audio(data, filename)
    duration = len(audio) / TARGET_SAMPLE_RATE
    if duration > MAX_CLIP_SECONDS:
        raise ValueError(
            f"Clip is {duration:.0f}s long; the limit is {MAX_CLIP_SECONDS:g}s. Trim it to the part with the speaker."
        )
    return audio


def _embed_audio(audio: np.ndarray, min_speech_ms: int) -> Tuple[np.ndarray, float]:
    """(embedding, seconds of speech) for a decoded clip.

    A clip with too little speech still produces an embedding, but it is
    noise; refusing it is safer than reporting a confident-looking score.
    """
    speech, _timestamps, segments = extract_speech(
        audio,
        TARGET_SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_ms=min_speech_ms,
        min_segment_ms=VAD_MIN_SEGMENT_MS,
        pad_ms=VAD_PAD_MS,
        merge_gap_ms=VAD_MERGE_GAP_MS,
    )
    if speech.size == 0:
        raise ValueError(f"Not enough clear speech: at least {min_speech_ms / 1000:.1f}s is required.")
    if segments:
        embedding = embed_segments(segments, TARGET_SAMPLE_RATE, VAD_MIN_SEGMENT_MS)
    else:
        embedding = embed(speech, TARGET_SAMPLE_RATE)
    return embedding, round(len(speech) / TARGET_SAMPLE_RATE, 3)


def _analyse_clip(data: bytes, filename: str, min_speech_ms: int) -> Dict[str, object]:
    """Decode, gate and embed one clip. ``wav_bytes`` is the stored form."""
    audio = _decode_clip(data, filename)
    embedding, speech_seconds = _embed_audio(audio, min_speech_ms)
    return {
        "filename": filename,
        "embedding": embedding,
        "wav_bytes": encode_wav(audio),
        "duration_seconds": round(len(audio) / TARGET_SAMPLE_RATE, 3),
        "speech_seconds": speech_seconds,
    }


def reembed_wav(wav_bytes: bytes) -> np.ndarray:
    """Recompute the embedding of a stored clip (see scripts/reembed.py)."""
    audio, _ = load_audio(wav_bytes)
    return _embed_audio(audio, MIN_SPEECH_MS)[0]


def initialize_system() -> None:
    db.init_db()


def _ranked(embedding: np.ndarray, threshold: float, person_id: Optional[str] = None) -> List[Dict[str, object]]:
    return rank_speakers(db.speaker_similarities(embedding, person_id), threshold, SIGMOID_STEEPNESS)


def _named(items: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Attach each person's name; drop anyone deleted since the vector lookup."""
    persons = db.find_persons_by_ids([item["person_id"] for item in items])
    named = []
    for item in items:
        person = persons.get(item["person_id"])
        if person:
            named.append({**item, "name": person.get("name"), "status": person.get("status")})
    return named


def _clean_person_data(person_data: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    cleaned = {}
    for field in db.PERSON_FIELDS:
        value = person_data.get(field)
        value = value.strip() if isinstance(value, str) else value
        cleaned[field] = value or None
    return cleaned


def _check_enrollment_consistency(clips: Sequence[Dict[str, object]]) -> None:
    """All clips of one enrollment must be the same voice.

    Each clip is compared with the average of the others, so one wrong file
    (another person, a recording of the officer) cannot join the record.
    """
    if len(clips) < 2:
        return
    vectors = np.vstack([db.normalise(clip["embedding"]) for clip in clips])
    for index, clip in enumerate(clips):
        others = np.delete(vectors, index, axis=0).mean(axis=0)
        similarity = cosine_similarity(vectors[index], others / np.linalg.norm(others))
        if similarity < ENROLL_CONSISTENCY_THRESHOLD:
            which = "The two clips do not" if len(clips) == 2 else f"Clip {index + 1} ({clip['filename']}) does not"
            raise ValueError(
                f"{which} sound like the same speaker ({similarity:.0%} similar; "
                f"{ENROLL_CONSISTENCY_THRESHOLD:.0%} required). Every clip must be the same person."
            )


def _find_existing_voice(embeddings: Sequence[np.ndarray]) -> Optional[Dict[str, object]]:
    """The best already-enrolled person scoring >= DUPLICATE_THRESHOLD against
    any of ``embeddings``, if there is one."""
    best: Optional[Dict[str, object]] = None
    for embedding in embeddings:
        ranked = _ranked(embedding, DUPLICATE_THRESHOLD)
        if ranked and ranked[0]["above_threshold"] and (best is None or ranked[0]["similarity"] > best["similarity"]):
            best = ranked[0]
    return best


def enroll_person(
    person_data: Dict[str, Optional[str]], uploads: Sequence[Upload], allow_duplicate: bool = False
) -> Dict[str, object]:
    requested_id = (person_data.get("person_id") or "").strip() or None
    person_data = _clean_person_data(person_data)
    if not person_data.get("name"):
        raise ValueError("Name is required.")
    if not uploads:
        raise ValueError("At least one voice clip is required for enrollment.")
    if len(uploads) > MAX_ENROLL_CLIPS:
        raise ValueError(f"At most {MAX_ENROLL_CLIPS} clips can be enrolled at once.")

    if requested_id and not _PERSON_ID_RE.match(requested_id):
        raise ValueError("Person ID may only contain letters, digits, '_' and '-' (max 64).")

    # Decoding, VAD and embedding run before the transaction opens, so a slow
    # model never holds database locks.
    clips: List[Dict[str, object]] = []
    for index, (filename, data) in enumerate(uploads, start=1):
        filename = _safe_filename(filename)
        try:
            clips.append(_analyse_clip(data, filename, ENROLLMENT_MIN_SPEECH_MS))
        except ValueError as exc:
            raise ValueError(f"Clip {index} ({filename}): {exc}") from exc

    _check_enrollment_consistency(clips)

    if not allow_duplicate:
        existing = _find_existing_voice([clip["embedding"] for clip in clips])
        if existing:
            raise ValueError(
                f"This voice is already enrolled as {existing['person_id']} "
                f"({existing['similarity']:.1%} similar). Enroll anyway only if this is a different person."
            )

    # The person, their clips and their embeddings commit together or not at all.
    person_id = requested_id
    try:
        with db.session_scope() as session:
            person_id = requested_id or db.get_next_person_id(session)
            if db.person_exists(person_id, session):
                raise ValueError(f"Person ID {person_id} already exists.")
            clip_ids = db.add_person_with_clips(session, person_id, person_data, clips)
    except IntegrityError:
        # A concurrent enrollment claimed the same ID between check and insert.
        raise ValueError(f"Person ID {person_id} already exists.")

    speech = sum(clip["speech_seconds"] for clip in clips)
    return {
        "success": True,
        "person_id": person_id,
        "clip_ids": clip_ids,
        "speech_seconds": round(speech, 1),
        "message": f"Enrolled {len(clips)} clip(s), {speech:.1f}s of speech.",
    }


def _require_enrolled_voices() -> None:
    if db.count_embeddings() == 0:
        raise ValueError("No voices are enrolled yet. Enroll speakers before matching.")


def _probe(upload: Upload) -> Dict[str, object]:
    filename, data = upload
    return _analyse_clip(data, _safe_filename(filename), MIN_SPEECH_MS)


def match_clip(upload: Upload) -> Dict[str, object]:
    """1:N identification.

    Returns one of three outcomes in ``status``:
      * ``match``     best person >= VOICE_THRESHOLD and clearly ahead of the next person
      * ``ambiguous`` best person >= VOICE_THRESHOLD but the next *different*
                      person is within MATCH_MARGIN, so no identity is asserted
      * ``no_match``  best person below VOICE_THRESHOLD
    """
    _require_enrolled_voices()
    clip = _probe(upload)

    ranked = _named(_ranked(clip["embedding"], VOICE_THRESHOLD))
    if not ranked:
        raise ValueError("No voices are enrolled yet. Enroll speakers before matching.")
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = round(best["similarity"] - runner_up["similarity"], 4) if runner_up else None

    result: Dict[str, object] = {
        "success": True,
        "matched": False,
        "similarity": best["similarity"],
        "confidence": best["confidence"],
        "threshold": VOICE_THRESHOLD,
        "margin": margin,
        "required_margin": MATCH_MARGIN,
        "duration_seconds": clip["duration_seconds"],
        "speech_seconds": clip["speech_seconds"],
        "ranked_matches": ranked[:5],
    }

    if best["similarity"] < VOICE_THRESHOLD:
        return {**result, "status": "no_match", "message": "Unknown speaker"}

    if margin is not None and margin < MATCH_MARGIN:
        return {
            **result,
            "status": "ambiguous",
            "candidates": ranked[:2],
            "message": "Ambiguous: more than one enrolled speaker matches closely. Review the candidates manually.",
        }

    person = db.find_person_by_id(best["person_id"])
    if not person:
        # Deleted between the vector lookup and this read.
        return {**result, "status": "no_match", "message": "Person not found."}

    return {**result, "matched": True, "status": "match", "person": person, "message": f"Matched {person['name']}."}


def match_against_person(person_id: str, upload: Upload) -> Dict[str, object]:
    """1:1 verification: is this clip the given person?"""
    person = db.find_person_by_id(person_id)
    if not person:
        raise ValueError(f"Person {person_id} does not exist.")

    clip = _probe(upload)
    ranked = _ranked(clip["embedding"], PERSON_MATCH_THRESHOLD, person_id)
    base = {
        "success": True,
        "person_id": person_id,
        "name": person.get("name"),
        "threshold": PERSON_MATCH_THRESHOLD,
        "speech_seconds": clip["speech_seconds"],
    }
    if not ranked:
        return {
            **base,
            "matched": False,
            "similarity": 0.0,
            "message": "No stored embeddings found for selected person.",
        }

    score = ranked[0]
    matched = score["above_threshold"]
    return {
        **base,
        **{k: score[k] for k in ("similarity", "confidence", "centroid_similarity", "enrollment_top_similarity")},
        "matched": matched,
        "message": "Voice matches selected person." if matched else "Voice does not match selected person.",
    }


def search_persons(upload: Upload) -> Dict[str, object]:
    """Top five speakers at or above SEARCH_THRESHOLD, for human review."""
    _require_enrolled_voices()
    clip = _probe(upload)

    ranked = [item for item in _ranked(clip["embedding"], SEARCH_THRESHOLD) if item["above_threshold"]]
    results = _named(ranked[:5])
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    if not results:
        return {
            "success": True,
            "results": [],
            "threshold": SEARCH_THRESHOLD,
            "speech_seconds": clip["speech_seconds"],
            "message": f"No matching speakers found above {SEARCH_THRESHOLD:.0%} similarity.",
        }
    return {"success": True, "results": results, "threshold": SEARCH_THRESHOLD, "speech_seconds": clip["speech_seconds"]}


def confirm_match(person_id: str, upload: Upload) -> Dict[str, object]:
    """Store a new clip and embedding for an existing person.

    The clip must already match that person at CONFIRM_THRESHOLD. Otherwise a
    mistaken or malicious confirm would attach a stranger's voice to the
    record, and the stranger would match as this person from then on.
    """
    if not db.person_exists(person_id):
        raise ValueError(f"Person {person_id} does not exist.")

    filename, data = upload
    # Stored permanently, so held to the enrollment speech minimum.
    clip = _analyse_clip(data, _safe_filename(filename), ENROLLMENT_MIN_SPEECH_MS)
    ranked = _ranked(clip["embedding"], CONFIRM_THRESHOLD, person_id)
    best = ranked[0]["similarity"] if ranked else 0.0
    if best < CONFIRM_THRESHOLD:
        raise ValueError(
            f"Clip does not match {person_id} closely enough to add to their record "
            f"({best:.1%}; {CONFIRM_THRESHOLD:.0%} required)."
        )

    clip_id = db.add_clip(person_id, clip, source="confirm")
    return {
        "success": True,
        "person_id": person_id,
        "clip_id": clip_id,
        "similarity": best,
        "message": "Clip and embedding saved.",
    }


def compare_clips(uploads: Sequence[Upload], threshold: Optional[float] = None) -> Dict[str, object]:
    """Compare clips with each other and group them by speaker. Nothing is stored."""
    if len(uploads) < 2:
        raise ValueError("Upload at least two clips to compare.")
    if len(uploads) > _MAX_COMPARE_CLIPS:
        raise ValueError(f"At most {_MAX_COMPARE_CLIPS} clips can be compared at once.")
    threshold = VOICE_THRESHOLD if threshold is None else threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    clips = []
    for index, (filename, data) in enumerate(uploads, start=1):
        filename = _safe_filename(filename)
        try:
            analysed = _analyse_clip(data, filename, MIN_SPEECH_MS)
        except ValueError as exc:
            raise ValueError(f"Clip {index} ({filename}): {exc}") from exc
        clips.append({"clip_id": f"clip_{index:02d}", **analysed})

    embeddings = [db.normalise(clip["embedding"]) for clip in clips]
    similarities, comparisons = compute_pairwise_results(
        embeddings, [clip["clip_id"] for clip in clips], threshold, SIGMOID_STEEPNESS
    )

    groups: Dict[str, List[str]] = {}
    speaker_of_label: Dict[int, str] = {}
    summaries = []
    for clip, label in zip(clips, cluster(similarities, threshold)):
        speaker = speaker_of_label.setdefault(int(label), f"speaker_{len(speaker_of_label) + 1}")
        groups.setdefault(speaker, []).append(clip["clip_id"])
        summaries.append(
            {
                "clip_id": clip["clip_id"],
                "filename": clip["filename"],
                "duration_seconds": clip["duration_seconds"],
                "speech_seconds": clip["speech_seconds"],
                "speaker": speaker,
            }
        )

    return {
        "success": True,
        "threshold": threshold,
        "clips": summaries,
        "comparisons": comparisons,
        "groups": groups,
        "message": f"{len(clips)} clips, {len(groups)} distinct speaker(s).",
    }


def delete_person_cascade(person_id: str) -> Dict[str, object]:
    """Delete a person with their clips and embeddings in one transaction."""
    counts = db.delete_person(person_id)
    return {
        "success": True,
        "person_id": person_id,
        **counts,
        "message": f"Deleted {person_id} with {counts['removed_clips']} clips and "
        f"{counts['removed_embeddings']} embeddings.",
    }


def integrity_report() -> Dict[str, object]:
    return {"success": True, **db.integrity_report()}
