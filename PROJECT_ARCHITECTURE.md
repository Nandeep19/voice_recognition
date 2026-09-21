# Voice Recognition Pipeline: Architecture and Data Flow

## 1. Application Purpose

This project is a FastAPI service for speaker recognition. It supports three
main use cases:

1. Enroll a named speaker from an audio sample.
2. Identify one audio clip against enrolled speaker profiles.
3. Compare multiple clips and group clips that likely contain the same speaker.

The system recognizes speaker identity from voice characteristics. It does not
transcribe speech and does not currently perform speaker diarization inside a
recording containing multiple people.

## 2. High-Level Application Flow

```text
Client / Swagger UI / curl
          |
          v
FastAPI routes (app/main.py)
          |
          v
VoicePipeline orchestration (app/pipeline.py)
          |
          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
Audio normalization       Silero VAD          Profile storage
(app/audio.py)             (app/vad.py)        (app/storage.py)
          |                    |                    |
          +----------+---------+                    |
                     v                              |
          ECAPA-TDNN embeddings                     |
              (app/models.py)                       |
                     |                              |
                     v                              |
          Similarity and confidence <---------------+
              (app/scoring.py)
                     |
                     v
          Pydantic response models
              (app/schemas.py)
                     |
                     v
              JSON API response
```

## 3. Architecture Style

The application uses a small layered architecture.

### API Layer

File: `app/main.py`

- Creates the FastAPI application.
- Defines HTTP endpoints and upload parameters.
- Converts pipeline errors into HTTP 422 responses.
- Returns Pydantic response models or report files.
- Creates one shared `Settings`, `ModelRegistry`, and `VoicePipeline` instance.

### Orchestration Layer

File: `app/pipeline.py`

- Coordinates the complete audio-processing workflows.
- Saves uploaded files and normalized audio.
- Calls VAD and embedding generation.
- Invokes profile matching or pairwise comparison.
- Applies identification and clustering decisions.
- Writes batch-analysis reports.

### Signal and Machine-Learning Layer

Files: `app/audio.py`, `app/vad.py`, `app/models.py`

- FFmpeg converts input media to mono, 16 kHz, 16-bit WAV.
- SoundFile loads WAV data as NumPy `float32` arrays.
- Silero VAD detects speech regions and removes silence/non-speech.
- SpeechBrain ECAPA-TDNN creates L2-normalized speaker embeddings.
- Multiple VAD segments are embedded separately and combined using
  square-root-duration weighting.

The ECAPA model receives normalized VAD speech directly. ECAPA-TDNN was trained
on raw 16 kHz audio, so no extra filtering is applied: additional bandpass or
pre-emphasis moves the input away from the training distribution and tends to
degrade the embedding.

Clips must contain a minimum amount of detected speech before an embedding is
accepted. Enrollment requires more than identification, because a weak
enrollment is averaged into the centroid permanently and degrades every later
match against that speaker.

```text
ENROLLMENT_MIN_SPEECH_MS = 3000   enrollment
MIN_SPEECH_MS            = 1000   identification and batch analysis
```

### Scoring Layer

File: `app/scoring.py`

- Calculates cosine similarity between normalized embeddings.
- Produces pairwise comparison results for batch analysis.
- Matches a query against profile centroids and individual enrollments.
- Converts distance from the threshold into sigmoid-based decision confidence.

The identification score for each profile is:

```text
combined score =
    0.50 * centroid similarity
  + 0.30 * average of top 3 enrollment similarities
  + 0.20 * best enrollment similarity
```

For profiles containing only one enrollment, the centroid and enrollment
vectors are normally identical, so all three terms carry the same score.

### Persistence Layer

File: `app/storage.py`

- Stores one JSON file per speaker in `data/profiles/`.
- Stores every enrollment embedding and its metadata.
- Recomputes an L2-normalized centroid after additional enrollment.
- Provides create, read, list, and delete operations.
- Uses a process-local lock around profile writes and deletes.

This is filesystem storage, not a database. It is appropriate for a prototype
or one application process, but it is not designed for multiple server
processes writing the same profiles concurrently.

### Contract and Configuration Layer

Files: `app/schemas.py`, `app/config.py`

- Pydantic models define the JSON response contracts.
- `pydantic-settings` loads defaults and optional `.env` overrides.
- Data directories are derived from `DATA_DIR` and created at startup.

## 4. Frameworks and Libraries

| Area | Framework or library | Responsibility |
| --- | --- | --- |
| Web API | FastAPI | Routing, uploads, validation, Swagger/OpenAPI |
| Server | Uvicorn | ASGI application server |
| Schemas | Pydantic | Response validation and serialization |
| Configuration | pydantic-settings | Environment-based settings |
| Media conversion | FFmpeg | Decode formats and normalize audio |
| Audio loading | SoundFile | WAV-to-NumPy conversion |
| Numeric processing | NumPy | Waveforms, embeddings, vector operations |
| VAD | Silero VAD through `torch.hub` | Speech/non-speech segmentation |
| Speaker model | SpeechBrain ECAPA-TDNN | 192-dimensional speaker embeddings |
| ML runtime | PyTorch / Torchaudio | Model inference and tensors |
| Clustering | Scikit-learn | Agglomerative speaker grouping |
| Testing | Pytest | Unit and behavioral tests |

## 5. Project Modules

| File | Main responsibility |
| --- | --- |
| `app/main.py` | HTTP API and application initialization |
| `app/pipeline.py` | Enrollment, identification, and analysis workflows |
| `app/audio.py` | FFmpeg normalization and WAV loading |
| `app/vad.py` | Silero model loading and speech extraction |
| `app/models.py` | ECAPA model loading and embedding aggregation |
| `app/scoring.py` | Cosine scoring, combined matching, confidence |
| `app/storage.py` | Speaker JSON files and centroid management |
| `app/schemas.py` | API response models |
| `app/config.py` | Thresholds, model selection, and paths |
| `tests/` | Audio, scoring, storage, clustering, and margin tests |

## 6. Speaker Enrollment Data Flow

Endpoints:

```text
POST /speakers
POST /speakers/{speaker_id}/enroll
```

Flow:

```text
Multipart audio + speaker name
        |
        v
Save temporary upload
        |
        v
FFmpeg: mono 16 kHz WAV
        |
        v
Load waveform as float32
        |
        v
Silero VAD detects and extracts speech segments
        |
        v
ECAPA embedding per usable segment
        |
        v
Duration-weighted average + L2 normalization
        |
        v
Create or update speaker JSON profile
        |
        v
Recompute profile centroid
        |
        v
Return speaker ID and enrollment count
```

New speaker IDs are generated from the name, for example `Alice` becomes
`spk_alice`. A numeric suffix is added when that ID already exists.

## 7. Speaker Identification Data Flow

Endpoint:

```text
POST /identify?threshold=0.75
```

Flow:

```text
Query audio
    |
    v
Normalize -> VAD -> ECAPA query embedding
    |
    v
Load all JSON speaker profiles
    |
    v
For every profile:
  - compare query to centroid
  - compare query to individual enrollments
  - calculate combined score
    |
    v
Sort candidates by combined score
    |
    v
Apply score threshold and top-1/top-2 margin
    |
    v
Return top_match plus high-to-low ranked_matches
    |
    v
identified / uncertain / unknown
```

Current decision rules:

```text
If no profiles exist:
    decision = no_speakers_enrolled

If top score < 0.75:
    decision = unknown
    same_person = false

If top score >= 0.75 but top score - second score < 0.05:
    decision = uncertain
    top_match.same_person = false

Otherwise:
    decision = identified
    top_match.same_person = true
```

Both values can be configured:

```env
SPEAKER_SIMILARITY_THRESHOLD=0.75
IDENTIFICATION_MIN_MARGIN=0.05
```

The `confidence` field is confidence in the threshold decision, calculated by
a sigmoid centered on the threshold. It is not a statistically calibrated
probability that the speaker identity is correct.

The identification response also includes percentage display fields for each
match. `match_percentage` is the combined embedding similarity shown as a
0-100 score and sorted from highest to lowest. `confidence_percentage` is the
same threshold-decision confidence as `confidence`, shown as a percentage.
The raw `similarity`, `confidence`, and existing `all_matches` fields remain
available for compatibility.

## 8. Batch Analysis Data Flow

Endpoint:

```text
POST /analyses
```

Flow:

```text
Multiple uploaded clips
        |
        v
Process every clip independently
        |
        v
One embedding per uploaded clip
        |
        v
N x N cosine similarity matrix
        |
        +-----------------------------+
        |                             |
        v                             v
Pairwise comparison list      Agglomerative clustering
                                      |
                                      v
                             person_001, person_002, ...
        |                             |
        +--------------+--------------+
                       v
              Saved analysis JSON report
```

The report is stored as `data/reports/{analysis_id}.json`. Original batch
uploads and processed speech WAV files are retained under analysis-specific
directories for inspection.

Although the pipeline implementation can technically process one clip and
assign it to `person_001`, this endpoint is intended and documented for
comparing two or more clips. A single clip should use `/identify` when the goal
is matching an enrolled person.

## 9. Speaker Profile Format

Location:

```text
data/profiles/spk_<name>.json
```

Conceptual structure:

```json
{
  "speaker_id": "spk_alice",
  "name": "Alice",
  "created_at": "UTC ISO timestamp",
  "updated_at": "UTC ISO timestamp",
  "embeddings": [
    {
      "embedding": ["192 floating-point values"],
      "source_file": "alice.wav",
      "duration_seconds": 5.2,
      "enrolled_at": "UTC ISO timestamp"
    }
  ],
  "centroid": ["192 floating-point values"],
  "enrollment_count": 1
}
```

The centroid is the normalized mean of all enrollment embeddings. Profile
files contain biometric representations and should be treated as sensitive
data even though they do not contain raw audio.

## 10. Runtime File Flow

```text
data/
  uploads/
    temp/                  Temporary identify/enroll uploads
    {analysis_id}/         Original batch-analysis uploads
  processed/
    temp/                  Temporary normalized identify/enroll WAV files
    {analysis_id}/         Normalized and extracted batch speech WAV files
  profiles/                Persistent speaker profile JSON files
  reports/                 Persistent batch-analysis reports
```

Identify and enrollment temporary source/normalized files are deleted after a
successful processing flow. Batch files are retained. If processing raises an
exception before cleanup, temporary files can remain.

## 11. Model Lifecycle and Concurrency

- FastAPI constructs a single `ModelRegistry` at module import time.
- ECAPA-TDNN is lazy-loaded on the first embedding request.
- Silero VAD is also lazy-loaded on first use.
- Thread locks prevent duplicate model initialization in one process.
- The SpeechBrain model is cached in `models/speaker/`.
- `torch.hub` may need network access for the first Silero load.

The route functions are asynchronous, but FFmpeg, VAD, model inference, file
I/O, and scoring are mostly synchronous operations. Heavy concurrent traffic
can therefore block application workers. A production deployment would use a
controlled worker strategy or a background inference queue.

## 12. API Summary

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Service name and documentation path |
| GET | `/ui` | Browser test console (enroll, identify, accuracy bench) |
| GET | `/health` | Model-load state and enrolled-speaker count |
| POST | `/speakers` | Create a speaker profile |
| POST | `/speakers/{id}/enroll` | Add an enrollment to a profile |
| GET | `/speakers` | List profile metadata |
| DELETE | `/speakers/{id}` | Delete a profile |
| POST | `/identify` | Match one clip against profiles |
| POST | `/analyses` | Compare and cluster uploaded clips |
| GET | `/analyses/{id}` | Read a saved report |
| GET | `/analyses/{id}/download` | Download a report JSON file |

## 13. Error and Validation Flow

- FastAPI validates query ranges such as `threshold` between 0 and 1.
- Invalid or undecodable media produces HTTP 422.
- Missing FFmpeg produces HTTP 422 with a runtime error message.
- Audio with no accepted VAD speech produces HTTP 422.
- Missing speakers and reports produce HTTP 404.
- Analysis IDs receive basic path-safety validation.
- Corrupted profile JSON files are skipped when listing all profiles.

## 14. Testing Architecture

The test suite covers:

- Audio loading: mono downmix and resampling to 16 kHz.
- Cosine similarity and similarity matrices.
- Sigmoid confidence behavior.
- Combined centroid/enrollment scoring.
- Top-1/top-2 uncertain decisions.
- Speaker profile creation, update, centroid, loading, and deletion.
- Agglomerative clustering behavior.

The tests mainly exercise units with synthetic vectors and audio. They do not
currently provide an end-to-end accuracy benchmark using a labeled voice
dataset, real uploads, FFmpeg, Silero, and ECAPA inference together.

## 15. Current Strengths

- Modules have clear, focused responsibilities.
- Models are lazy-loaded rather than loaded for every request.
- Embeddings and centroids are L2-normalized consistently.
- Identification uses both stable centroids and individual enrollment matches.
- The top-candidate margin prevents ambiguous high scores from being accepted.
- Multiple enrollment samples improve a profile without replacing history.
- Pydantic response models keep API contracts explicit.
- Path sanitization reduces traversal risk for profiles and reports.

## 16. Current Limitations and Risks

1. **No speaker diarization:** one clip containing multiple speakers can create
   a mixed embedding and an unreliable result.
2. **Thresholds are heuristic:** `0.75` and `0.05` are not calibrated on a
   project-specific validation dataset.
3. **Confidence is not probability:** sigmoid confidence can look precise even
   when the underlying model has not been calibrated for this environment.
4. **JSON storage has limited concurrency:** the lock protects one process only
   and writes are not atomic across processes.
5. **No authentication or authorization:** anyone with API access can enroll,
   inspect, or delete speakers.
6. **No upload limits:** very large files can consume disk, CPU, and memory.
7. **No explicit audio-quality gate:** clipping, low SNR, background voices,
   and music are not scored before enrollment.
8. **Synchronous inference:** expensive operations can block API workers.
9. **Minimum-speech gates are heuristic:** 3000 ms and 1000 ms are reasonable
   defaults for ECAPA-TDNN but have not been tuned on project data.

## 17. Recommended Next Architecture Improvements

### Accuracy

- Build a labeled validation set from actual users, devices, and rooms.
- Tune similarity and margin thresholds using false-accept and false-reject
  measurements.
- Enroll 3-5 samples per speaker from different sessions.
- Add speech-quality checks before accepting enrollment embeddings.
- Reject or diarize recordings containing multiple speakers.

### Reliability

- Put temporary-file cleanup in `finally` blocks.
- Validate upload MIME types and enforce file-size/duration limits.
- Add startup checks for FFmpeg and local model availability.
- Replace plain JSON writes with atomic temporary-file replacement.

### Production Scale

- Add authentication and speaker-level authorization.
- Move profiles to a database or vector store with transaction support.
- Run inference in a bounded worker pool or task queue.
- Add structured logging, request IDs, metrics, and model latency tracking.
- Version profiles with the model and preprocessing configuration so embeddings
  from incompatible pipelines cannot be mixed.

## 18. Recommended Operational Flow

```text
1. Start API
2. Enroll each person with multiple clean recordings
3. Confirm profile enrollment counts
4. Identify new single-speaker clips with /identify
5. Use /analyses only to compare whole clips against one another
6. Review unknown and uncertain cases instead of forcing an identity
7. Periodically evaluate thresholds against labeled recordings
8. Back up and protect data/profiles as biometric data
```

## 19. Overall Assessment

The project is a well-structured speaker-recognition prototype with a clean
separation between API handling, audio processing, inference, scoring, and
storage. Its strongest current workflow is closed-set identification of a
single dominant speaker against a small set of enrolled profiles. The next
major accuracy improvement should come from measured threshold calibration,
multiple diverse enrollments, audio-quality controls, and diarization or
multi-speaker rejection rather than from making the confidence formula more
aggressive.
