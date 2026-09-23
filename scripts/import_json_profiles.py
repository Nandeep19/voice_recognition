"""One-time import of the old JSON speaker profiles into PostgreSQL.

    python -m scripts.import_json_profiles --dry-run
    python -m scripts.import_json_profiles            # reads data/profiles/
    python -m scripts.import_json_profiles --dir D:/old/profiles

Each profile keeps its speaker ID (e.g. spk_alice) as its person ID, its name
and its creation time. Its embeddings are imported without audio: the old
storage never kept the clips, so these embeddings cannot be re-embedded.
Re-enrolling those speakers with fresh recordings, then deleting the imported
record, gives them stored clips.

Profiles whose ID already exists in the database are skipped, so the import
can be re-run safely. Each profile is imported in its own transaction.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from src import database as db
from src.config import BASE_DIR, PERSON_ID_PATTERN
from src.voice_service import initialize_system

_PERSON_ID_RE = re.compile(PERSON_ID_PATTERN)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=BASE_DIR / "data" / "profiles")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.dir.glob("*.json"))
    if not paths:
        print(f"No profiles found in {args.dir}.")
        return 0

    initialize_system()
    imported = skipped = failed = 0
    for path in paths:
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
            person_id = profile["speaker_id"]
            vectors = [db.normalise(e["embedding"]) for e in profile.get("embeddings", []) if "embedding" in e]
            if not vectors:
                vectors = [db.normalise(profile["centroid"])]
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            failed += 1
            print(f"{path.name}: FAILED - {exc}")
            continue

        if not _PERSON_ID_RE.match(person_id):
            failed += 1
            print(f"{path.name}: FAILED - unsafe speaker ID {person_id!r}")
            continue
        if db.person_exists(person_id):
            skipped += 1
            print(f"{path.name}: skipped, {person_id} already in the database")
            continue

        print(f"{path.name}: {person_id} ({profile.get('name')}), {len(vectors)} embedding(s)")
        if args.dry_run:
            imported += 1
            continue
        with db.session_scope() as session:
            person = db.Person(person_id=person_id, name=profile.get("name") or person_id)
            if profile.get("created_at"):
                person.created_at = datetime.fromisoformat(profile["created_at"])
            session.add(person)
            session.flush()
            for vector in vectors:
                session.add(db.VoiceEmbedding(person_id=person_id, clip_id=None, embedding=vector))
        imported += 1

    verb = "would import" if args.dry_run else "imported"
    print(f"{verb} {imported}, skipped {skipped}, failed {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
