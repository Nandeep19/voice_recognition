"""Recompute stored embeddings from the stored clips.

    python -m scripts.reembed --dry-run
    python -m scripts.reembed

Run after any change to how embeddings are computed (model, VAD settings,
segment weighting). Each clip's embedding is replaced in its own transaction,
so an interrupted run leaves every clip with either its old or its new
embedding. Embeddings imported from the old JSON profiles have no audio and
are left as they are.
"""

import argparse
import sys

import numpy as np
from sqlalchemy import select

from src import database as db
from src.voice_service import initialize_system, reembed_wav


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report how far each embedding would move; write nothing")
    args = parser.parse_args()

    initialize_system()
    failures = 0
    for clip_id in db.iter_clip_ids():
        found = db.get_clip_for_reembed(clip_id)
        if found is None:
            continue  # deleted meanwhile
        person_id, wav = found
        try:
            embedding = db.normalise(reembed_wav(wav))
        except ValueError as exc:
            failures += 1
            print(f"clip {clip_id} ({person_id}): FAILED - {exc}")
            continue

        with db.session_scope() as session:
            old = session.scalar(
                select(db.VoiceEmbedding.embedding).where(db.VoiceEmbedding.clip_id == clip_id)
            )
        drift = "new" if old is None else f"similarity to old {float(np.dot(np.asarray(old), embedding)):.4f}"
        print(f"clip {clip_id} ({person_id}): {drift}")
        if not args.dry_run:
            db.replace_clip_embedding(clip_id, embedding)

    print("dry run: nothing written." if args.dry_run else "done.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
