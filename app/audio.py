"""Audio preprocessing — format normalization and loading.

ECAPA-TDNN was trained on raw 16 kHz audio, so the waveform is handed to the
model as-is. Extra filtering moves the input away from the training
distribution and tends to degrade the embedding rather than improve it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000


def normalize_audio(source: Path, destination: Path) -> None:
    """Convert any audio format to mono 16 kHz 16-bit WAV via FFmpeg."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip().splitlines()
        message = details[-1] if details else "unknown FFmpeg error"
        raise ValueError(f"Could not decode audio: {message}") from exc


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 at TARGET_SAMPLE_RATE."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        # Resample using linear interpolation (lightweight, good enough after FFmpeg)
        target_length = round(len(audio) * TARGET_SAMPLE_RATE / sample_rate)
        indices = np.linspace(0, len(audio) - 1, target_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sample_rate = TARGET_SAMPLE_RATE
    return audio, sample_rate
