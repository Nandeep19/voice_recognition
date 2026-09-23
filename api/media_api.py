"""Audio delivery from the voice_clips table.

Only a well-formed person ID and a numeric clip ID are looked up; any other
request returns 404 before touching the database. A clip is served only under
the person it belongs to. Failures are always 404, so the endpoint does not
reveal whether a person or clip exists.
"""

import re

from fastapi import APIRouter, HTTPException, Response

from src import database as db
from src.config import PERSON_ID_PATTERN

router = APIRouter(tags=["media"])

_PERSON_ID_RE = re.compile(PERSON_ID_PATTERN)
_CLIP_ID_RE = re.compile(r"^[0-9]{1,18}$")
_NOT_FOUND = HTTPException(status_code=404, detail="Clip not found.")


@router.get("/media/{person_id}/clips/{clip_id}.wav")
def get_clip(person_id: str, clip_id: str) -> Response:
    if not _PERSON_ID_RE.match(person_id) or not _CLIP_ID_RE.match(clip_id):
        raise _NOT_FOUND

    clip = db.get_clip_audio(person_id, int(clip_id))
    if clip is None:
        raise _NOT_FOUND

    data, content_type = clip
    return Response(
        content=data,
        media_type=content_type,
        # Private: these are personal records; no shared-cache storage.
        headers={"Cache-Control": "private, max-age=300"},
    )
