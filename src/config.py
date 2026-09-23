import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


FRONTEND_PATH = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "models"

# PostgreSQL with the pgvector extension, e.g.
# postgresql://vr_app:password@localhost:5432/vr_dopams
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# The models themselves are pinned in src/model_files.py, not configured here.
# ECAPA-TDNN (VoxCeleb) produces 192-d embeddings; fixes the VECTOR column size.
EMBEDDING_DIM = 192

# All scores are the combined per-speaker score from src/scoring.py, not the
# raw similarity to one clip. Measured on real recordings: known speakers
# 0.69-0.90, an unenrolled speaker 0.47 (see README, "Choosing thresholds").
# Re-measure as the gallery grows.
VOICE_THRESHOLD = float(os.getenv("VOICE_THRESHOLD", 0.55))
PERSON_MATCH_THRESHOLD = float(os.getenv("PERSON_MATCH_THRESHOLD", 0.55))
SEARCH_THRESHOLD = float(os.getenv("SEARCH_THRESHOLD", 0.45))

# 1:N identify only returns a match when the best speaker beats the next-best
# *different* speaker by at least this much; otherwise the result is ambiguous.
MATCH_MARGIN = float(os.getenv("MATCH_MARGIN", 0.05))
# Enrollment is refused when a clip already scores this high against someone
# else, unless explicitly overridden.
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.60))
# "Add clip to speaker" requires the clip to match that speaker this well.
CONFIRM_THRESHOLD = float(os.getenv("CONFIRM_THRESHOLD", 0.55))
# Every clip of one enrollment must sound like the others at least this much,
# so a wrong file cannot slip into someone's record.
ENROLL_CONSISTENCY_THRESHOLD = float(os.getenv("ENROLL_CONSISTENCY_THRESHOLD", 0.53))
# Stored clips (enroll, add clip) get a warning when a part of the clip, of at
# least MULTI_VOICE_MIN_SEGMENT_S, sounds unlike the rest. Measured on real
# clips: genuine single-speaker clips >= 0.497; two-speaker mixes 0.27-0.52.
# The ranges overlap, so this warns and never refuses; it caught 4 of 6 mixes.
MULTI_VOICE_THRESHOLD = float(os.getenv("MULTI_VOICE_THRESHOLD", 0.45))
MULTI_VOICE_MIN_SEGMENT_S = float(os.getenv("MULTI_VOICE_MIN_SEGMENT_S", 2.0))
# Sharpness of the confidence curve around the threshold.
SIGMOID_STEEPNESS = float(os.getenv("SIGMOID_STEEPNESS", 15.0))

# Silero VAD.
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", 0.35))
VAD_MIN_SEGMENT_MS = int(os.getenv("VAD_MIN_SEGMENT_MS", 500))
VAD_PAD_MS = int(os.getenv("VAD_PAD_MS", 100))
VAD_MERGE_GAP_MS = int(os.getenv("VAD_MERGE_GAP_MS", 300))

# Minimum usable speech. ECAPA-TDNN embeddings are unstable below a few seconds.
# Enrollment is stricter because a weak clip is stored permanently and pulls
# every later match against that speaker.
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", 1000))
ENROLLMENT_MIN_SPEECH_MS = int(os.getenv("ENROLLMENT_MIN_SPEECH_MS", 3000))

MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 25))
# Stored clips are 16 kHz mono 16-bit WAV (~1.9 MB per minute).
MAX_CLIP_SECONDS = float(os.getenv("MAX_CLIP_SECONDS", 180))
MAX_ENROLL_CLIPS = int(os.getenv("MAX_ENROLL_CLIPS", 5))

# Person IDs appear in URLs, so they are restricted to a safe character set.
PERSON_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
