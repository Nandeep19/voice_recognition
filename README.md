# Voice Recognition Pipeline

Speaker recognition API that determines whether audio recordings contain the
same speaker or different speakers.  Built with ECAPA-TDNN embeddings, Silero
VAD, and cosine similarity scoring.

## Pipeline

```text
Audio File
  → FFmpeg (mono 16 kHz normalization)
  → Silero VAD (neural voice activity detection)
  → Minimum-speech gate (3 s to enroll, 1 s to identify)
  → ECAPA-TDNN / SpeechBrain (speaker embedding)
  → Multi-segment weighted averaging
  → JSON Profile Storage (centroid + individual embeddings)
  → Cosine Similarity
  → Sigmoid-calibrated Confidence Score
  → Same Person / Different Person
```

## Libraries

| Stage               | Library / Model                        |
| ------------------- | -------------------------------------- |
| Audio Processing    | FFmpeg                                 |
| VAD                 | Silero VAD (via `torch.hub`)           |
| Speaker Recognition | SpeechBrain ECAPA-TDNN                 |
| Similarity          | NumPy, Scikit-learn                    |
| Storage             | JSON files on disk                     |
| API                 | FastAPI + Uvicorn                      |
| Output              | Match Result + Confidence              |

## Setup

### Requirements

- **Python 3.11 or 3.12** (3.14 is not yet supported by ML libraries)
- **FFmpeg** installed and on PATH — confirm with `ffmpeg -version`

### Install

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Optional: install dev/test dependencies
pip install -r requirements-dev.txt
```

### Configure (Optional)

Copy `.env.example` to `.env` and adjust settings:

```powershell
copy .env.example .env
```

Key settings:

| Variable                       | Default | Description                              |
| ------------------------------ | ------- | ---------------------------------------- |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.75`  | Cosine threshold for same-speaker        |
| `IDENTIFICATION_MIN_MARGIN`    | `0.05`  | Required gap between top two candidates  |
| `VAD_THRESHOLD`                | `0.35`  | Silero speech probability threshold      |
| `SIGMOID_STEEPNESS`            | `15.0`  | Confidence curve sharpness               |
| `MIN_SPEECH_MS`                | `1000`  | Minimum speech to identify a clip        |
| `ENROLLMENT_MIN_SPEECH_MS`     | `3000`  | Minimum speech to accept an enrollment   |

## Run the Server

```powershell
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger documentation, or
**http://127.0.0.1:8000/ui** for the browser test console (enroll speakers,
identify clips, and run an accuracy bench against your own recordings).

> **First request** downloads the ECAPA-TDNN model (~80 MB) and Silero VAD
> (~2 MB).  This takes 1–3 minutes.  Subsequent requests are fast.

## API Endpoints

### Speaker Enrollment

```powershell
# Enroll a new speaker
curl -X POST http://127.0.0.1:8000/speakers -F "name=Alice" -F "file=@alice.wav"

# Add more samples to improve accuracy
curl -X POST http://127.0.0.1:8000/speakers/spk_alice/add-embedding -F "file=@alice2.wav"

# List all enrolled speakers
curl http://127.0.0.1:8000/speakers

# Delete a speaker
curl -X DELETE http://127.0.0.1:8000/speakers/spk_alice
```

### Speaker Identification

```powershell
# Identify who is speaking in a clip
curl -X POST http://127.0.0.1:8000/identify -F "file=@unknown.wav"

# Use a custom threshold
curl -X POST "http://127.0.0.1:8000/identify?threshold=0.80" -F "file=@unknown.wav"

```

### Batch Analysis (Compare Multiple Clips)

```powershell
# Compare two or more clips against each other
curl -X POST http://127.0.0.1:8000/analyses \
  -F "files=@clip1.wav" \
  -F "files=@clip2.wav" \
  -F "files=@clip3.wav"

# Retrieve a previous analysis
curl http://127.0.0.1:8000/analyses/{analysis_id}

# Download report as JSON file
curl -O http://127.0.0.1:8000/analyses/{analysis_id}/download
```

### Health Check

```powershell
curl http://127.0.0.1:8000/health
```

## Testing

```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_scoring.py -v
python -m pytest tests/test_storage.py -v
python -m pytest tests/test_audio.py -v
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

## Quick Test Workflow

1. **Start the server:**
   ```powershell
   uvicorn app.main:app --reload
   ```

2. **Enroll two speakers** (use any voice recordings you have):
   ```powershell
   curl -X POST http://127.0.0.1:8000/speakers -F "name=Speaker A" -F "file=@speaker_a.wav"
   curl -X POST http://127.0.0.1:8000/speakers -F "name=Speaker B" -F "file=@speaker_b.wav"
   ```

3. **Identify a clip:**
   ```powershell
   curl -X POST http://127.0.0.1:8000/identify -F "file=@test_clip.wav"
   ```

4. **Check the response:**
   ```json
   {
     "decision": "identified",
     "top_match": {
       "rank": 1,
       "speaker_id": "spk_speaker_a",
       "name": "Speaker A",
       "similarity": 0.8723,
       "match_percentage": 87.23,
       "confidence": 0.9641,
       "confidence_percentage": 96.41,
       "same_person": true
     },
     "ranked_matches": [
       {
         "rank": 1,
         "speaker_id": "spk_speaker_a",
         "name": "Speaker A",
         "similarity": 0.8723,
         "match_percentage": 87.23,
         "confidence": 0.9641,
         "confidence_percentage": 96.41,
         "same_person": true
       }
     ]
   }
   ```

Identification uses a combined score: 50% profile centroid similarity, 30%
top-three enrollment average, and 20% best enrollment similarity. A candidate
must pass both the similarity threshold and the minimum top-1/top-2 margin;
otherwise the decision is returned as `uncertain` rather than guessed.
`ranked_matches` and `all_matches` are sorted from highest to lowest voice
match. `match_percentage` is the embedding similarity displayed as a 0-100
score, while `confidence_percentage` is confidence in the threshold decision.


## Project Structure

```
voicr_recogisation/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI endpoints
│   ├── pipeline.py      # Core orchestration (analyze, identify, enroll)
│   ├── vad.py           # Silero VAD integration
│   ├── audio.py         # FFmpeg normalization and WAV loading
│   ├── models.py        # ECAPA-TDNN embedding (single + multi-segment)
│   ├── scoring.py       # Cosine similarity, sigmoid confidence, top-N
│   ├── storage.py       # JSON speaker profile CRUD
│   ├── config.py        # Settings (env-based)
│   └── schemas.py       # Pydantic request/response models
├── tests/
│   ├── test_audio.py
│   ├── test_scoring.py
│   ├── test_storage.py
│   └── test_pipeline.py
├── data/                # Created at runtime
│   ├── uploads/
│   ├── processed/
│   ├── reports/
│   └── profiles/        # Enrolled speaker JSON files
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

## Accuracy Tips

- **Enroll multiple samples** per speaker (3–5 recordings from different
  sessions).  The centroid averaging removes per-session noise.
- **Use clean audio** for enrollment — quiet room, close microphone.
- **Recordings ≥ 3 seconds** of actual speech give the best embeddings.
- **Threshold tuning**: start with `0.75`.  Raise to `0.80` if you see false
  positives.  Lower to `0.70` only if you have noisy conditions.
- Same-speaker pairs typically score **0.80–0.95** similarity.
- Different-speaker pairs typically score **0.10–0.55** similarity.
