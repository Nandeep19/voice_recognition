from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Voice Recognition Pipeline"
    data_dir: Path = Path("data")

    # --- Speaker embedding model ---
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    # --- Silero VAD ---
    vad_threshold: float = 0.35  # speech probability threshold (lower = more permissive)
    vad_min_segment_ms: int = 500  # drop speech segments shorter than this before embedding
    vad_pad_ms: int = 100  # pad each speech segment by this many ms on each side
    vad_merge_gap_ms: int = 300  # merge segments closer than this

    # --- Scoring ---
    speaker_similarity_threshold: float = 0.65  # ECAPA-TDNN needs ~0.75 for reliable decisions
    identification_min_margin: float = 0.05  # required gap between top-1 and top-2 scores
    sigmoid_steepness: float = 15.0  # controls confidence curve sharpness around threshold
    min_speech_ms: int = 250  # absolute minimum speech to accept a clip

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    def create_directories(self) -> None:
        for directory in (self.uploads_dir, self.processed_dir, self.reports_dir, self.profiles_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
