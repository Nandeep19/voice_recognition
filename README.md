# Voice Enrollment and Matching API

Speaker recognition with a FastAPI backend, PostgreSQL + pgvector storage and a
web console. Supports enrollment, 1:N identification, 1:1 verification,
candidate search and clip comparison.

The system is self-contained: its own database (`vr_dopams`), its own database
role (`vr_app`), and no references to or from any other system.

## What this system uses

- **FastAPI** for REST API endpoints
- **FFmpeg** to read any audio format and convert it to mono 16 kHz
- **Silero VAD** to keep only the speech
- **SpeechBrain ECAPA-TDNN** for 192-d speaker embeddings
- **PostgreSQL + pgvector** for everything that is stored: person records, the
  enrolled audio clips, voice embeddings and the match log, in one database
- **Plain HTML + JS** console, no framework

## Project Structure

```
voice_recognition/
├── app.py                     # FastAPI application entrypoint
│
├── api/
│   ├── enroll_api.py          # POST /enroll
│   ├── match_api.py           # /match, /search, /compare, /persons, /dashboard, ...
│   ├── media_api.py           # GET /media/{person_id}/clips/{clip_id}.wav (from the DB)
│   └── admin_api.py           # GET /admin/integrity
│
├── src/
│   ├── config.py              # Configuration from .env
│   ├── database.py            # PostgreSQL schema, sessions and queries
│   ├── audio.py               # FFmpeg decoding, WAV encoding
│   ├── vad.py                 # Silero voice activity detection
│   ├── model_files.py         # Pinned model versions and checksums
│   ├── embed.py               # ECAPA-TDNN embedding, multi-segment averaging
│   ├── scoring.py             # Combined speaker score, confidence, clustering
│   └── voice_service.py       # Core enrollment and matching service
│
├── database/
│   └── init_db.py             # Creates the pgvector extension and tables
│
├── scripts/
│   ├── maintenance.py         # Offline integrity check
│   ├── reembed.py             # Recompute stored embeddings from stored clips
│   ├── fetch_models.py        # Download and verify the pinned model files (once)
│   ├── import_json_profiles.py  # One-time import of the old data/profiles/*.json
│   └── backfill_audio.py      # Attach original recordings to imported speakers
│
├── frontend/
│   └── console.html           # Single-page console (served at /)
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. PostgreSQL with pgvector

You need PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector)
extension installed. As a superuser, create a dedicated role and database:

```sql
CREATE ROLE vr_app LOGIN PASSWORD '<strong password>';
CREATE DATABASE vr_dopams OWNER vr_app;
REVOKE ALL ON DATABASE vr_dopams FROM PUBLIC;
GRANT CONNECT, TEMPORARY ON DATABASE vr_dopams TO vr_app;
\c vr_dopams
CREATE EXTENSION vector;   -- needs a superuser once; pgvector is not a trusted extension
GRANT USAGE, CREATE ON SCHEMA public TO vr_app;
```

`vr_app` is not a superuser and can reach only this database (plus
`vr_dopams_test` for tests, set up the same way). Keep it that way: other
databases on the same server should `REVOKE CONNECT ... FROM PUBLIC` so this
role cannot open them either. The app runs `CREATE EXTENSION IF NOT EXISTS
vector` on startup, which is a no-op once the extension exists.

### 2. Install dependencies

Python 3.11 or 3.12, and FFmpeg on PATH (`ffmpeg -version`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
```

Then fetch the model files, once, on a machine with internet access:

```powershell
python -m scripts.fetch_models
```

Both models are pinned in `src/model_files.py`: ECAPA-TDNN to one Hugging Face
revision, Silero VAD to v5.1.2, each file to a SHA-256 checksum. The server
loads them only from `models/`, verifies every checksum at startup, and never
downloads anything or runs downloaded code at runtime. A missing or altered
file stops the server at startup with a message saying what to run. To deploy
to a machine without internet, copy the `models/` folder.

### 3. Configure environment

```powershell
Copy-Item .env.example .env
```

```env
DATABASE_URL=postgresql://vr_app:PASSWORD@localhost:5432/vr_dopams
VOICE_THRESHOLD=0.75
```

URL-encode special characters in the password. Tables are created on first
start (or run `python -m database.init_db`).

### 4. Import the old JSON profiles (once)

Speakers enrolled before the move to PostgreSQL live in `data/profiles/`:

```powershell
python -m scripts.import_json_profiles --dry-run
python -m scripts.import_json_profiles
```

They keep their IDs (`spk_...`), names and enrollment dates. The old storage
never kept audio, so they arrive with embeddings only. Re-running the import
skips IDs already present.

Then attach the original recordings, if you still have them. The JSON files
name the file behind each embedding:

```powershell
python -m scripts.backfill_audio --audio-dir C:/path/to/recordings --dry-run
python -m scripts.backfill_audio --audio-dir C:/path/to/recordings
```

Each recording is re-embedded and must match its imported embedding (≥ 0.95)
before it is stored; that imported embedding is then replaced, so nothing is
counted twice. Missing recordings leave their imported embedding in place.

## Running the Application

```powershell
uvicorn app:app --reload
```

The console is at http://127.0.0.1:8000/ and the API docs at `/docs`. Both
models are loaded before the server accepts requests, so the first request is
as fast as any other, and `/health` reports the loaded model. All state lives
in PostgreSQL, so several workers are safe:

```powershell
uvicorn app:app --workers 4
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/enroll` | New person from 1–5 clips of the same speaker |
| POST | `/match` | 1:N: who is speaking? |
| POST | `/match/person` | 1:1: is this clip the given person? |
| POST | `/search` | Top 5 candidates above `SEARCH_THRESHOLD`, for review |
| POST | `/match/confirm` | Store another clip for an existing person |
| POST | `/compare` | Group 2–10 clips by speaker. Nothing stored |
| GET | `/persons`, `/persons/{id}` | List, or one person with clips and lookup history |
| DELETE | `/persons/{id}` | Delete a person with their clips and embeddings |
| GET | `/media/{id}/clips/{clip_id}.wav` | Play a stored clip |
| GET | `/dashboard/summary`, `/match/logs` | Counts and recent lookups |
| GET | `/admin/integrity` | Incomplete records |

Every response has the same envelope: `success`, `message`, `data`, `errors`,
plus the result fields at the top level.

### POST /enroll

**Request (form-data):**
- `name` (required); optional `dob`, `gender`, `father_name`, `crime_type`,
  `crime_details`, `police_station`, `jail_name`, `sentence`, `status`
- `clips`: 1–5 audio files of the same person, each with ≥ 3 s of speech. Any
  format FFmpeg reads: WAV, MP3, M4A, OGG/Opus, AMR, FLAC, WebM, video files.
- `person_id` (optional): letters, digits, `_` and `-`. Generated as
  `VOICE000001`, ... when omitted.
- `allow_duplicate` (optional): enroll even if the voice is already on record

The person, every clip and every embedding are written in one transaction. If
anything fails, nothing is stored. Enrollment is refused when:
- a clip has too little speech, is longer than `MAX_CLIP_SECONDS` or larger
  than `MAX_UPLOAD_MB`;
- the clips do not sound like one speaker (`ENROLL_CONSISTENCY_THRESHOLD`);
- the voice already matches someone at ≥ `DUPLICATE_THRESHOLD`, unless
  `allow_duplicate=true`.

```json
{ "success": true, "person_id": "VOICE000001", "message": "Enrolled 2 clip(s), 19.3s of speech.", "warnings": [] }
```

`warnings` flags a clip where part of the recording sounds like a different
voice (see "Two voices in one clip" below). The clip is still stored.

### POST /match

**Request (form-data):** `file`: one audio clip (≥ 1 s of speech)

| `status` | Meaning |
|---|---|
| `match` | Best person ≥ `VOICE_THRESHOLD` **and** ahead of the next *different* person by ≥ `MATCH_MARGIN` |
| `ambiguous` | Above threshold, but another person scores within `MATCH_MARGIN`. No identity is asserted; `candidates` lists both |
| `no_match` | Best person below `VOICE_THRESHOLD` |

Every response includes `similarity`, `confidence`, `threshold`, `margin`,
`speech_seconds` and `ranked_matches` (top 5 with names).

### POST /match/confirm

Stores the clip and its embedding on an existing person. The clip must already
match that person at ≥ `CONFIRM_THRESHOLD`, so a stranger's voice can never be
attached to a record. Returns `warnings` like `/enroll`.

## How a score is computed

1. FFmpeg converts the upload to mono 16 kHz.
2. Silero VAD keeps only speech. Too little speech is refused rather than scored.
3. Each speech segment is embedded by ECAPA-TDNN. The segments are averaged,
   weighted by √duration.
4. For each enrolled person, pgvector computes in one SQL query the cosine
   similarity to their **centroid** (`avg(embedding)`) and to **each clip**.
5. These combine into one score: `0.5·centroid + 0.3·mean(top 3 clips) +
   0.2·best clip`, or `0.7·centroid + 0.3·best clip` with fewer than three clips.
6. `confidence` is a sigmoid centred on the threshold: ~50% at the boundary,
   near 100% far from it on either side.

### Choosing thresholds

Measured on 8 real recordings of 3 speakers (raw clip-to-clip similarity):

| | Similarity |
|---|---|
| Same speaker, different recording | 0.59 – 0.75 |
| Different speakers | 0.23 – 0.48 |

Combined scores for known speakers were 0.69–0.90; for the unenrolled
speaker, 0.47. `ENROLL_CONSISTENCY_THRESHOLD=0.53` sits in the raw gap. The
`.env.example` defaults (0.75) are stricter than this data needs; the working
`.env` uses 0.55. That is 3 speakers only: re-measure as the gallery grows,
with recordings not used for enrollment. `/compare` shows raw pairwise
similarities for any set of clips.

### Two voices in one clip

A recording of a call or a conversation blends two people into one record.
Clips that will be stored (enrollment, add clip) are split into speech pieces
of ≥ `MULTI_VOICE_MIN_SEGMENT_S` (2 s), and each piece is compared with the rest
of the clip. Below `MULTI_VOICE_THRESHOLD` (0.45) the response carries a
warning naming the time range.

This **warns and never refuses**, because on real recordings the two cases
overlap: genuine single-speaker clips scored ≥ 0.497, two-person mixes
0.27–0.52. It caught 4 of 6 test mixes with no false alarms; a second voice
that sounds similar can pass unnoticed. Reliable detection needs a speaker
diarization model.

### Model versions

Every embedding stores the model that produced it (`voice_embeddings.model`),
and search only compares embeddings from the current pinned model. Embeddings
from different models live in different vector spaces: comparing them gives
scores that look normal and mean nothing. After a model change, stale
embeddings are ignored and flagged by the integrity check until
`python -m scripts.reembed` recomputes them from the stored clips.

## Database Schema

| Table | Contents |
|---|---|
| `persons` | `person_id` (PK), personal and record fields, `created_at` |
| `voice_clips` | `id`, `person_id` → persons, `audio_data BYTEA` (16 kHz mono WAV), `filename`, `duration_seconds`, `speech_seconds`, `source` (`enroll`/`confirm`) |
| `voice_embeddings` | `id`, `person_id` → persons, `clip_id` → voice_clips (NULL only for JSON imports), `embedding VECTOR(192)` (L2-normalised), `model` (the model that produced it) |
| `match_logs` | `person_id` (not a FK, so logs outlive deletions), `name`, `similarity`, `matched`, `status`, `timestamp` |
| `person_id_seq` | Sequence behind generated `VOICE` IDs |

Clips and embeddings are separate tables, so similarity scans never read audio
bytes. Clips are stored as the normalised 16 kHz WAV, not the original upload:
one playable format, container metadata (device, location, encoder) dropped,
and exactly the audio the embedding came from. That costs ~1.9 MB per minute.

Probe clips (match, verify, search, compare) are **not** stored; the match log
records only that a lookup happened and its result.

### Similarity

Cosine similarity is computed in SQL as `1 - (embedding <=> :query)`. Search is
an **exact** scan, so no match is ever missed. Scores combine in Python, so
every person is scored on each request. That is fine into the tens of
thousands of speakers. Beyond that, pre-filter with an HNSW index on
`voice_embeddings.embedding` and score only the top few hundred persons.

## Security and Operations

### Authentication

**There is none.** Every endpoint is open to anyone who can reach the port.
Before exposing it beyond this machine, put it behind authentication (reverse
proxy with access control, network ACL, or application-level auth).

### Audio access

`GET /media/{person_id}/clips/{clip_id}.wav` serves clips from `voice_clips`.
`person_id` is restricted to `[A-Za-z0-9_-]{1,64}` and `clip_id` to digits; a
clip is only served under the person it belongs to. Anything invalid or missing
returns 404. Responses are `Cache-Control: private`.

### Backups

Audio lives in the database, so `pg_dump` captures everything. Dumps grow by
~1.9 MB per minute of enrolled audio.

### Integrity check

```powershell
python -m scripts.maintenance integrity   # exits non-zero if problems are found
```

Flags persons with no current-model embedding (unsearchable), clips missing
their embedding, and embeddings from another model. Also at
`GET /admin/integrity`.

### Re-embedding

After changing the pinned model (`src/model_files.py`) or VAD settings,
recompute every embedding from the stored clips. Each is tagged with the
current model:

```powershell
python -m scripts.reembed --dry-run   # prints how far each embedding moves
python -m scripts.reembed
```

### Concurrency

Requests are processed in parallel threads. The VAD model is recurrent (it
carries state between audio frames), so each thread has its own copy; a single
shared copy crashed the whole process when two uploads arrived together.

### Tests

```powershell
python -m pytest
```

Offline tests (validation, scoring, audio, media access) always run. The
PostgreSQL integration tests (atomic enrollment, cascading delete, centroid and
similarity in SQL, match decisions, JSON import) run when `TEST_DATABASE_URL`
points at a database whose name contains `test`:

```powershell
$env:TEST_DATABASE_URL = "postgresql://vr_app:PASSWORD@localhost:5432/vr_dopams_test"
python -m pytest
```

That database is truncated on every test. Never point it at real data.
