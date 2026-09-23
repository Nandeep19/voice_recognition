from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.uploads import read_upload
from src.voice_service import enroll_person

router = APIRouter()


@router.post("/enroll")
async def enroll(
    name: str = Form(...),
    dob: str | None = Form(None),
    gender: str | None = Form(None),
    father_name: str | None = Form(None),
    crime_type: str | None = Form(None),
    crime_details: str | None = Form(None),
    police_station: str | None = Form(None),
    jail_name: str | None = Form(None),
    sentence: str | None = Form(None),
    status: str | None = Form(None),
    person_id: str | None = Form(None),
    allow_duplicate: bool = Form(False),
    clips: List[UploadFile] = File(..., description="1-5 recordings of the same speaker"),
):
    try:
        uploads = [await read_upload(clip) for clip in clips]
        # Decoding, VAD and embedding are CPU-heavy; keep them off the event loop.
        result = await run_in_threadpool(
            enroll_person,
            {
                "person_id": person_id,
                "name": name,
                "dob": dob,
                "gender": gender,
                "father_name": father_name,
                "crime_type": crime_type,
                "crime_details": crime_details,
                "police_station": police_station,
                "jail_name": jail_name,
                "sentence": sentence,
                "status": status,
            },
            uploads,
            allow_duplicate,
        )
        return {
            "success": True,
            "message": result.get("message", "Enrollment completed successfully."),
            "person_id": result.get("person_id"),
            "data": result,
            "errors": [],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {exc}")
