"""Attach the original recordings to speakers imported from the JSON profiles.

    python -m scripts.backfill_audio --audio-dir C:/Users/me/Downloads --dry-run
    python -m scripts.backfill_audio --audio-dir C:/Users/me/Downloads

The JSON profiles name the file each embedding came from (``source_file``).
For every such file found in ``--audio-dir``:

1. the recording is decoded and embedded again;
2. the new embedding must match one of the person's audio-less (imported)
   embeddings at >= --min-similarity, which proves it is the same recording;
3. the clip and new embedding are stored, and that imported embedding is
   deleted, so the recording is not counted twice.

Each person is updated in one transaction: all of their found recordings
are attached, or none. Recordings that are missing or fail the check are
reported and their imported embeddings are left as they are. Re-running is
safe: an imported embedding already replaced is no longer there to match.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select

from src import database as db
from src.config import BASE_DIR, ENROLLMENT_MIN_SPEECH_MS
from src.voice_service import _analyse_clip, initialize_system


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", type=Path, default=BASE_DIR / "data" / "profiles")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--min-similarity", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    initialize_system()
    attached = missing = rejected = 0

    for path in sorted(args.profiles.glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        person_id = profile["speaker_id"]
        if not db.person_exists(person_id):
            print(f"{person_id}: not in the database, skipped (run import_json_profiles first)")
            continue

        with db.session_scope() as session:
            imported = {
                row.id: db.normalise(row.embedding)
                for row in session.execute(
                    select(db.VoiceEmbedding.id, db.VoiceEmbedding.embedding).where(
                        db.VoiceEmbedding.person_id == person_id, db.VoiceEmbedding.clip_id.is_(None)
                    )
                )
            }

        plan = []  # (clip, imported embedding id to replace)
        for record in profile.get("embeddings", []):
            name = record.get("source_file")
            if not name:
                continue
            audio_path = args.audio_dir / name
            if not audio_path.is_file():
                missing += 1
                print(f"{person_id}: {name}: not found, imported embedding kept")
                continue
            try:
                clip = _analyse_clip(audio_path.read_bytes(), name, ENROLLMENT_MIN_SPEECH_MS)
            except ValueError as exc:
                rejected += 1
                print(f"{person_id}: {name}: FAILED - {exc}")
                continue

            new = db.normalise(clip["embedding"])
            candidates = {eid: float(np.dot(vec, new)) for eid, vec in imported.items()}
            if not candidates:
                print(f"{person_id}: {name}: no imported embedding left to replace (already done?)")
                continue
            best_id = max(candidates, key=candidates.get)
            if candidates[best_id] < args.min_similarity:
                rejected += 1
                print(
                    f"{person_id}: {name}: best match to an imported embedding is only "
                    f"{candidates[best_id]:.4f}; not the same recording, skipped"
                )
                continue
            del imported[best_id]  # one recording replaces one embedding
            plan.append((clip, best_id))
            print(f"{person_id}: {name}: {clip['speech_seconds']:.1f}s speech, "
                  f"matches imported embedding {best_id} at {candidates[best_id]:.4f}")

        if plan and not args.dry_run:
            with db.session_scope() as session:
                for clip, old_id in plan:
                    db.add_clip_with_embedding(session, person_id, clip, source="enroll")
                    session.execute(delete(db.VoiceEmbedding).where(db.VoiceEmbedding.id == old_id))
        attached += len(plan)

    verb = "would attach" if args.dry_run else "attached"
    print(f"{verb} {attached} recording(s); {missing} not found; {rejected} rejected.")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
