"""Voice recognition pipeline — core orchestration.

Wires together:  Audio → Silero VAD → ECAPA-TDNN → Embedding → Scoring

Provides two workflows:
1. ``analyze()``  — batch compare multiple clips against each other
2. ``identify()`` — match a single clip against enrolled speaker profiles
3. ``enroll()``   — add a voice sample to a speaker profile
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi import UploadFile
from sklearn.cluster import AgglomerativeClustering

from app.audio import TARGET_SAMPLE_RATE, load_audio, normalize_audio
from app.config import Settings
from app.models import ModelRegistry
from app.schemas import AnalysisReport, ClipResult, IdentifyResponse, MatchResult, PairResult
from app.scoring import compute_pairwise_results, match_against_profiles
from app.storage import enroll_speaker, load_all_profiles
from app.vad import extract_speech


class VoicePipeline:
    def __init__(self, settings: Settings, models: ModelRegistry) -> None:
        self.settings = settings
        self.models = models

    # ------------------------------------------------------------------
    # Batch analysis: compare multiple clips
    # ------------------------------------------------------------------

    async def analyze(
        self,
        files: list[UploadFile],
        threshold: float | None = None,
    ) -> AnalysisReport:
        """Upload multiple audio clips, extract embeddings, and compare all pairs."""
        if not files:
            raise ValueError("At least one audio file is required")

        threshold = threshold if threshold is not None else self.settings.speaker_similarity_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0 and 1")

        analysis_id = str(uuid4())
        upload_dir = self.settings.uploads_dir / analysis_id
        processed_dir = self.settings.processed_dir / analysis_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        clip_results: list[ClipResult] = []
        embeddings: list[np.ndarray] = []

        for index, upload in enumerate(files, start=1):
            clip_id = f"clip_{index:03d}"
            safe_name = Path(upload.filename or f"{clip_id}.audio").name
            source_path = upload_dir / f"{clip_id}_{safe_name}"
            with source_path.open("wb") as destination:
                shutil.copyfileobj(upload.file, destination)

            # Normalize → Load → Silero VAD → Multi-segment embed
            normalized_path = processed_dir / f"{clip_id}.wav"
            normalize_audio(source_path, normalized_path)
            audio, sample_rate = load_audio(normalized_path)

            speech, timestamps, segment_arrays = extract_speech(
                audio,
                sample_rate,
                threshold=self.settings.vad_threshold,
                min_speech_ms=self.settings.min_speech_ms,
                min_segment_ms=self.settings.vad_min_segment_ms,
                pad_ms=self.settings.vad_pad_ms,
                merge_gap_ms=self.settings.vad_merge_gap_ms,
            )

            if speech.size == 0:
                raise ValueError(
                    f"{safe_name}: not enough usable speech — at least "
                    f"{self.settings.min_speech_ms / 1000:.1f}s of clear speech is required"
                )

            # Save extracted speech for debugging
            speech_path = processed_dir / f"{clip_id}_speech.wav"
            sf.write(speech_path, speech, TARGET_SAMPLE_RATE, subtype="PCM_16")

            # Multi-segment embedding (weighted average)
            if segment_arrays:
                embedding = self.models.embed_segments(
                    segment_arrays,
                    sample_rate,
                    min_segment_ms=self.settings.vad_min_segment_ms,
                )
            else:
                embedding = self.models.embed(speech, sample_rate)

            embeddings.append(embedding)
            clip_results.append(
                ClipResult(
                    clip_id=clip_id,
                    filename=safe_name,
                    duration_seconds=round(len(audio) / sample_rate, 3),
                    speech_duration_seconds=round(len(speech) / sample_rate, 3),
                    speech_segments=[
                        {"start": round(s, 3), "end": round(e, 3)} for s, e in timestamps
                    ],
                )
            )

        # Pairwise scoring
        _, comparisons_raw = compute_pairwise_results(
            embeddings,
            [c.clip_id for c in clip_results],
            threshold,
            self.settings.sigmoid_steepness,
        )

        # Build similarity matrix for clustering
        matrix = np.vstack(embeddings)
        similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)

        # Cluster into speaker groups
        labels = self._cluster(similarities, threshold)
        groups: dict[str, list[str]] = {}
        label_to_person: dict[int, str] = {}
        for clip, label in zip(clip_results, labels, strict=True):
            person_id = label_to_person.setdefault(
                int(label), f"person_{len(label_to_person) + 1:03d}"
            )
            clip.person_id = person_id
            groups.setdefault(person_id, []).append(clip.clip_id)

        comparisons = [
            PairResult(
                clip_a=c["clip_a"],
                clip_b=c["clip_b"],
                similarity=c["similarity"],
                confidence=c["confidence"],
                same_speaker=c["same_speaker"],
            )
            for c in comparisons_raw
        ]

        report = AnalysisReport(
            analysis_id=analysis_id,
            created_at=datetime.now(timezone.utc),
            threshold=threshold,
            clips=clip_results,
            comparisons=comparisons,
            groups=groups,
        )
        report_path = self.settings.reports_dir / f"{analysis_id}.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return report

    # ------------------------------------------------------------------
    # Speaker identification: one clip vs. all enrolled profiles
    # ------------------------------------------------------------------

    async def identify(
        self,
        file: UploadFile,
        threshold: float | None = None,
    ) -> IdentifyResponse:
        """Identify who is speaking in a single audio clip.

        Compares the clip's embedding against all enrolled speaker centroids.
        Returns the top match with confidence and same/different decision.
        """
        threshold = threshold if threshold is not None else self.settings.speaker_similarity_threshold

        profiles = load_all_profiles(self.settings.profiles_dir)
        if not profiles:
            return IdentifyResponse(
                query_duration_seconds=0.0,
                speech_duration_seconds=0.0,
                threshold=threshold,
                minimum_margin=self.settings.identification_min_margin,
                decision="no_speakers_enrolled",
            )

        # Process the query audio
        embedding, audio_duration, speech_duration = await self._process_single_clip(
            file, self.settings.min_speech_ms
        )

        # Match against all profiles
        matches = match_against_profiles(
            embedding,
            profiles,
            threshold,
            self.settings.sigmoid_steepness,
        )

        all_matches = [MatchResult(**m) for m in matches]
        top = all_matches[0] if all_matches else None
        runner_up = all_matches[1] if len(all_matches) > 1 else None
        score_margin = (
            round(top.similarity - runner_up.similarity, 4)
            if top and runner_up
            else None
        )

        if not top or top.similarity < threshold:
            decision = "unknown"
        elif score_margin is not None and score_margin < self.settings.identification_min_margin:
            decision = "uncertain"
            top.same_person = False
        else:
            decision = "identified"

        return IdentifyResponse(
            query_duration_seconds=audio_duration,
            speech_duration_seconds=speech_duration,
            threshold=threshold,
            minimum_margin=self.settings.identification_min_margin,
            score_margin=score_margin,
            top_match=top,
            decision=decision,
            all_matches=all_matches,
            ranked_matches=all_matches,
        )

    # ------------------------------------------------------------------
    # Speaker enrollment: add voice sample to a profile
    # ------------------------------------------------------------------

    async def enroll(
        self,
        name: str,
        file: UploadFile,
        speaker_id: str | None = None,
    ) -> dict:
        """Enroll a speaker by extracting an embedding from their audio."""
        embedding, audio_duration, speech_duration = await self._process_single_clip(
            file, self.settings.enrollment_min_speech_ms
        )

        profile = enroll_speaker(
            profiles_dir=self.settings.profiles_dir,
            name=name,
            embedding=embedding,
            source_file=file.filename or "unknown",
            duration_seconds=speech_duration,
            speaker_id=speaker_id,
        )

        return {
            "speaker_id": profile.speaker_id,
            "name": profile.name,
            "enrollment_count": profile.enrollment_count,
            "message": f"Enrolled {speech_duration:.1f}s of speech",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_single_clip(
        self, file: UploadFile, min_speech_ms: int
    ) -> tuple[np.ndarray, float, float]:
        """Process a single uploaded file → (embedding, audio_duration, speech_duration)."""
        temp_id = str(uuid4())[:8]
        safe_name = Path(file.filename or "clip.audio").name
        upload_dir = self.settings.uploads_dir / "temp"
        processed_dir = self.settings.processed_dir / "temp"
        upload_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        source_path = upload_dir / f"{temp_id}_{safe_name}"
        normalized_path = processed_dir / f"{temp_id}.wav"

        try:
            with source_path.open("wb") as dst:
                shutil.copyfileobj(file.file, dst)

            normalize_audio(source_path, normalized_path)
            audio, sample_rate = load_audio(normalized_path)

            speech, timestamps, segment_arrays = extract_speech(
                audio,
                sample_rate,
                threshold=self.settings.vad_threshold,
                min_speech_ms=min_speech_ms,
                min_segment_ms=self.settings.vad_min_segment_ms,
                pad_ms=self.settings.vad_pad_ms,
                merge_gap_ms=self.settings.vad_merge_gap_ms,
            )

            if speech.size == 0:
                raise ValueError(
                    f"{safe_name}: not enough usable speech — at least "
                    f"{min_speech_ms / 1000:.1f}s of clear speech is required"
                )

            # Multi-segment embedding
            if segment_arrays:
                embedding = self.models.embed_segments(
                    segment_arrays, sample_rate, self.settings.vad_min_segment_ms
                )
            else:
                embedding = self.models.embed(speech, sample_rate)

            audio_duration = round(len(audio) / sample_rate, 3)
            speech_duration = round(len(speech) / sample_rate, 3)
            return embedding, audio_duration, speech_duration
        finally:
            for path in (source_path, normalized_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _cluster(similarities: np.ndarray, threshold: float) -> np.ndarray:
        """Agglomerative clustering using precomputed cosine distances."""
        if len(similarities) == 1:
            return np.array([0])
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
        return model.fit_predict(distances)
