"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Batch analysis (multi-clip comparison)
# ---------------------------------------------------------------------------

class ClipResult(BaseModel):
    clip_id: str
    filename: str
    duration_seconds: float
    speech_duration_seconds: float
    person_id: str | None = None
    speech_segments: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PairResult(BaseModel):
    clip_a: str
    clip_b: str
    similarity: float
    confidence: float
    same_speaker: bool


class AnalysisReport(BaseModel):
    analysis_id: str
    created_at: datetime
    threshold: float
    clips: list[ClipResult]
    comparisons: list[PairResult]
    groups: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Speaker enrollment
# ---------------------------------------------------------------------------

class SpeakerProfileResponse(BaseModel):
    speaker_id: str
    name: str
    enrollment_count: int
    created_at: str
    updated_at: str


class EnrollmentResponse(BaseModel):
    speaker_id: str
    name: str
    enrollment_count: int
    message: str


# ---------------------------------------------------------------------------
# Speaker identification
# ---------------------------------------------------------------------------

class MatchResult(BaseModel):
    rank: int | None = None
    speaker_id: str
    name: str
    similarity: float
    match_percentage: float
    confidence: float
    confidence_percentage: float
    same_person: bool
    enrollment_count: int
    centroid_similarity: float
    enrollment_top_similarity: float
    enrollment_top3_similarity: float


class IdentifyResponse(BaseModel):
    query_duration_seconds: float
    speech_duration_seconds: float
    threshold: float
    minimum_margin: float = 0.0
    score_margin: float | None = None
    top_match: MatchResult | None = None
    decision: str  # "identified", "uncertain", "unknown", "no_speakers_enrolled"
    all_matches: list[MatchResult] = Field(default_factory=list)
    ranked_matches: list[MatchResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    speaker_model_loaded: bool
    enrolled_speakers: int
