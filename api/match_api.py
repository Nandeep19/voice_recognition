from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.uploads import read_upload
from src.database import (
    create_match_log,
    find_person_by_id,
    get_all_persons,
    get_match_logs_for_person,
    get_match_stats,
    get_recent_match_logs,
    get_storage_stats,
)
from src.voice_service import (
    compare_clips,
    confirm_match,
    delete_person_cascade,
    match_against_person,
    match_clip,
    search_persons,
)

router = APIRouter()


def _envelope(message: str, result: dict, data=None) -> dict:
    return {"success": True, "message": message, "data": result if data is None else data, "errors": [], **result}


@router.get("/persons")
def list_persons():
    persons = [
        {
            "person_id": person["person_id"],
            "name": person.get("name", ""),
            "status": person.get("status"),
            "crime_type": person.get("crime_type"),
            "created_at": person.get("created_at"),
            "clip_count": len(person.get("clips", [])),
            "embedding_count": person.get("embedding_count", 0),
        }
        for person in get_all_persons()
    ]
    return {"success": True, "message": "Persons list retrieved successfully.", "data": persons, "errors": []}


@router.get("/persons/{person_id}")
def get_person(person_id: str):
    person = find_person_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    return {
        "success": True,
        "message": "Person details retrieved successfully.",
        "data": {"person": person, "match_history": get_match_logs_for_person(person_id)},
        "errors": [],
    }


@router.delete("/persons/{person_id}")
def remove_person(person_id: str):
    """Delete a person with their clips and embeddings in one transaction."""
    try:
        result = delete_person_cascade(person_id)
        return _envelope(result["message"], result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")


@router.get("/dashboard/summary")
def dashboard_summary():
    """Counts only: aggregate queries, no per-person rows are loaded."""
    storage = get_storage_stats()
    summary = {
        "total_persons": storage["persons"],
        "total_clips": storage["clips"],
        "total_embeddings": storage["embeddings"],
        "audio_seconds": storage["audio_seconds"],
        "audio_bytes": storage["audio_bytes"],
        **get_match_stats(),
    }
    return _envelope("Dashboard summary retrieved successfully.", summary)


@router.get("/match/logs")
def get_logs(limit: int = Query(10, ge=1, le=200)):
    try:
        logs = get_recent_match_logs(limit)
        return {"success": True, "message": "Match logs retrieved successfully.", "data": logs, "errors": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/match")
async def match(file: UploadFile = File(...)):
    """1:N: who is speaking?"""
    try:
        result = await run_in_threadpool(match_clip, await read_upload(file))
        person = result.get("person") or {}
        create_match_log(
            person_id=person.get("person_id"),
            name=person.get("name"),
            similarity=result.get("similarity", 0.0),
            matched=bool(result.get("matched")),
            status=result.get("status", "no_match"),
        )
        return _envelope("Match comparison finished.", result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Match request failed: {exc}")


@router.post("/match/person")
async def match_person(person_id: str = Form(...), file: UploadFile = File(...)):
    """1:1: is this the given person?"""
    try:
        result = await run_in_threadpool(match_against_person, person_id, await read_upload(file))
        create_match_log(
            person_id=person_id,
            name=result.get("name"),
            similarity=result.get("similarity", 0.0),
            matched=result.get("matched", False),
            status="verify",
        )
        return _envelope("Comparison with person finished.", result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Person match request failed: {exc}")


@router.post("/search")
async def search(file: UploadFile = File(...)):
    """Top five candidates above SEARCH_THRESHOLD for human review.

    Logged as ``search``, never as a match: candidates sit below the
    identification threshold and assert no identity.
    """
    try:
        result = await run_in_threadpool(search_persons, await read_upload(file))
        results = result.get("results", [])
        top = results[0] if results else {}
        create_match_log(
            person_id=top.get("person_id"),
            name=top.get("name"),
            similarity=top.get("similarity", 0.0),
            matched=False,
            status="search",
        )
        return _envelope("Search completed.", result, data=results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search request failed: {exc}")


@router.post("/match/confirm")
async def match_confirm(person_id: str = Form(...), file: UploadFile = File(...)):
    """Add a clip (audio + embedding) to an existing person."""
    try:
        result = await run_in_threadpool(confirm_match, person_id, await read_upload(file))
        return _envelope(result.get("message", "Match confirmed."), result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Confirm request failed: {exc}")


@router.post("/compare")
async def compare(
    files: List[UploadFile] = File(..., description="2-10 clips to compare with each other"),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
):
    """Group clips by speaker. Nothing is stored or logged."""
    try:
        uploads = [await read_upload(f) for f in files]
        result = await run_in_threadpool(compare_clips, uploads, threshold)
        return _envelope(result["message"], result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compare request failed: {exc}")
