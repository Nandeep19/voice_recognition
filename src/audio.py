"""Audio preprocessing — format normalization, loading and WAV encoding.

Every upload is converted to mono 16 kHz by FFmpeg, in memory, and that is
the form stored in the database, the only place audio is kept: one playable
format for every clip, container
metadata (device, location, encoder tags) dropped, and exactly the waveform
the embedding was computed from, so embeddings can be recomputed later.

ECAPA-TDNN was trained on raw 16 kHz audio, so the waveform is handed to the
model as-is. Extra filtering moves the input away from the training
distribution and tends to degrade the embedding rather than improve it.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


def _ffmpeg_to_pcm(source: str, stdin: bytes | None) -> bytes:
    command = [
        "ffmpeg", "-v", "error",
        *(["-nostdin"] if stdin is None else []),
        "-i", source,
        "-vn", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ]
    try:
        result = subprocess.run(command, input=stdin, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.decode("utf-8", "replace").strip().splitlines()
        message = details[-1] if details else "unknown FFmpeg error"
        raise ValueError(f"Could not decode audio: {message}") from exc
    return result.stdout


def decode_audio(data: bytes, filename: str = "clip") -> np.ndarray:
    """Decode any FFmpeg-readable audio to mono float32 at TARGET_SAMPLE_RATE.

    The upload is streamed to FFmpeg in memory; nothing is written to disk.
    The one exception is an MP4/M4A/MOV whose index sits at the end of the
    file, which FFmpeg can only reach by seeking: that is retried through a
    temporary file, deleted as soon as FFmpeg has read it.
    """
    try:
        pcm_bytes = _ffmpeg_to_pcm("pipe:0", data)
    except ValueError:
        if Path(filename).suffix.lower() not in _SEEKABLE_ONLY:
            raise
        pcm_bytes = _decode_via_temp_file(data, Path(filename).suffix.lower())

    pcm = np.frombuffer(pcm_bytes, dtype="<i2")
    if pcm.size == 0:
        raise ValueError("The file contains no audio.")
    return (pcm.astype(np.float32) / 32768.0)


# Containers that may need seeking to decode (index at the end of the file).
_SEEKABLE_ONLY = {".mp4", ".m4a", ".m4b", ".mov", ".3gp", ".3g2"}


def _decode_via_temp_file(data: bytes, suffix: str) -> bytes:
    fd, source = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return _ffmpeg_to_pcm(source, None)
    finally:
        try:
            os.unlink(source)
        except OSError:
            pass


def encode_wav(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """16-bit PCM WAV bytes, the stored form of every clip."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def load_audio(source: Path | BinaryIO | bytes) -> tuple[np.ndarray, int]:
    """Load a WAV (path, file object or bytes) as mono float32 at TARGET_SAMPLE_RATE."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        # Resample using linear interpolation (lightweight, good enough after FFmpeg)
        target_length = round(len(audio) * TARGET_SAMPLE_RATE / sample_rate)
        indices = np.linspace(0, len(audio) - 1, target_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    return audio, sample_rate
