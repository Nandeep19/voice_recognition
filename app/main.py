"""FastAPI application — Voice Recognition Pipeline.

Endpoints:
    POST /analyses              — batch compare multiple clips
    GET  /analyses/{id}         — retrieve a previous report
    GET  /analyses/{id}/download — download report as JSON

    POST   /speakers            — enroll a new speaker
    POST   /speakers/{id}/enroll — add more audio to existing profile
    GET    /speakers            — list all enrolled speakers
    DELETE /speakers/{id}       — remove a speaker profile

    POST /identify              — identify who is speaking in a clip

    GET  /health                — service status
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models import ModelRegistry
from app.pipeline import VoicePipeline
from app.schemas import (
    AnalysisReport,
    EnrollmentResponse,
    HealthResponse,
    IdentifyResponse,
    SpeakerProfileResponse,
)
from app.storage import delete_profile, load_all_profiles, load_profile


settings = get_settings()
settings.create_directories()
models = ModelRegistry(settings)
pipeline = VoicePipeline(settings, models)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Upload voice clips to compare speakers, identify enrolled voices, "
        "and manage speaker profiles with ECAPA-TDNN embeddings and Silero VAD."
    ),
)


# -----------------------------------------------------------------------
# Root & Health
# -----------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"message": settings.app_name, "docs": "/docs", "test_console": "/ui"}


@app.get("/ui", include_in_schema=False)
def test_console() -> FileResponse:
    """Serve the browser test console (same-origin, so no CORS setup needed)."""
    return FileResponse(Path(__file__).resolve().parent.parent / "test_ui.html")


@app.get("/health", response_model=HealthResponse, tags=["system"], include_in_schema=False)
def health() -> HealthResponse:
    profiles = load_all_profiles(settings.profiles_dir)
    return HealthResponse(
        status="ok",
        speaker_model_loaded=models.speaker_loaded,
        enrolled_speakers=len(profiles),
    )


# -----------------------------------------------------------------------
# Batch Analysis
# -----------------------------------------------------------------------

@app.post("/analyses", response_model=AnalysisReport, tags=["analysis"], include_in_schema=False)
async def create_analysis(
    files: list[UploadFile] = File(..., description="Two or more voice clips to compare"),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> AnalysisReport:
    """Upload multiple audio clips and compare all pairs for speaker similarity."""
    try:
        return await pipeline.analyze(files, threshold)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _report_path(analysis_id: str) -> Path:
    if not analysis_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid analysis ID")
    path = settings.reports_dir / f"{analysis_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    return path


@app.get("/analyses/{analysis_id}", response_model=AnalysisReport, tags=["analysis"], include_in_schema=False)
def get_analysis(analysis_id: str) -> AnalysisReport:
    """Retrieve a previous analysis report by ID."""
    payload = json.loads(_report_path(analysis_id).read_text(encoding="utf-8"))
    return AnalysisReport.model_validate(payload)


@app.get("/analyses/{analysis_id}/download", tags=["analysis"], include_in_schema=False)
def download_analysis(analysis_id: str) -> FileResponse:
    """Download an analysis report as a JSON file."""
    path = _report_path(analysis_id)
    return FileResponse(path, media_type="application/json", filename=path.name)


# -----------------------------------------------------------------------
# Speaker Enrollment
# -----------------------------------------------------------------------

@app.post("/speakers", response_model=EnrollmentResponse, tags=["speakers"])
async def enroll_new_speaker(
    name: str = Form(..., description="Speaker's name"),
    file: UploadFile = File(..., description="Voice sample audio file"),
) -> EnrollmentResponse:
    """Enroll a new speaker with a voice sample."""
    try:
        result = await pipeline.enroll(name=name, file=file)
        return EnrollmentResponse(**result)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/speakers/{speaker_id}/add-embedding",
    response_model=EnrollmentResponse,
    tags=["speakers"],
    summary="Add Embedding to Existing Profile",
)
async def add_embedding_to_existing_profile(
    speaker_id: str,
    file: UploadFile = File(..., description="Additional voice sample"),
) -> EnrollmentResponse:
    """Add another voice sample to an existing speaker profile.

    Multiple enrollments improve accuracy by averaging out per-session
    variability (different microphones, room acoustics, speaking styles).
    """
    profile = load_profile(settings.profiles_dir, speaker_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
    try:
        result = await pipeline.enroll(
            name=profile.name,
            file=file,
            speaker_id=speaker_id,
        )
        return EnrollmentResponse(**result)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/speakers", response_model=list[SpeakerProfileResponse], tags=["speakers"])
def list_speakers() -> list[SpeakerProfileResponse]:
    """List all enrolled speaker profiles."""
    profiles = load_all_profiles(settings.profiles_dir)
    return [
        SpeakerProfileResponse(
            speaker_id=p.speaker_id,
            name=p.name,
            enrollment_count=p.enrollment_count,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in profiles
    ]


@app.delete("/speakers/{speaker_id}", tags=["speakers"])
def remove_speaker(speaker_id: str) -> dict[str, str]:
    """Delete an enrolled speaker profile."""
    if not delete_profile(settings.profiles_dir, speaker_id):
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
    return {"message": f"Speaker '{speaker_id}' deleted"}


# -----------------------------------------------------------------------
# Speaker Identification
# -----------------------------------------------------------------------

@app.post("/identify", response_model=IdentifyResponse, tags=["analysis"])
async def identify_speaker(
    file: UploadFile = File(..., description="Voice clip to identify"),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
) -> IdentifyResponse:
    """Identify who is speaking by matching against all enrolled speaker profiles.

    Returns the top match with similarity score, confidence level, and
    a same_person/different_person decision.
    """
    try:
        return await pipeline.identify(file, threshold)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
