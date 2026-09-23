"""Silero VAD — neural voice activity detection.

Silero VAD is a compact neural network trained on thousands of hours of speech
and non-speech audio.  It vastly outperforms energy-based (RMS) detectors on
soft speech, background music, HVAC rumble, and varying SNR conditions.

The model is recurrent: it carries hidden state from one 32 ms frame to the
next. One shared instance used by two requests at once interleaves that state
and, in practice, crashes the process natively. Each thread therefore gets its
own instance (~2 MB), loaded from the pinned, checksum-verified file
models/silero_vad.jit (see src/model_files.py). Nothing is downloaded and no
downloaded code is run.
"""

from __future__ import annotations

import threading

import numpy as np
import torch

from src.model_files import VAD_FILE, verify_vad_model

_verify_lock = threading.Lock()
_verified = False
_local = threading.local()


def _get_vad():
    """This thread's Silero VAD model, loaded on first use in the thread."""
    global _verified
    model = getattr(_local, "model", None)
    if model is None:
        if not _verified:
            with _verify_lock:
                if not _verified:
                    verify_vad_model()
                    _verified = True
        model = torch.jit.load(str(VAD_FILE), map_location="cpu")
        model.eval()
        _local.model = model
    return model


def warm_up() -> None:
    """Verify and load the model now rather than on the first request."""
    _get_vad()


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.35,
    min_segment_ms: int = 500,
    pad_ms: int = 100,
    merge_gap_ms: int = 300,
) -> list[tuple[int, int]]:
    """Detect speech segments using Silero VAD.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 waveform, values in [-1, 1].
    sample_rate : int
        Must be 16000 (Silero requirement).
    threshold : float
        Speech probability threshold.  Lower values are more permissive and
        capture soft / quiet speech at the cost of more false positives.
    min_segment_ms : int
        Discard speech segments shorter than this (milliseconds).
    pad_ms : int
        Pad each detected segment on both sides to preserve onset/offset
        transients that carry speaker identity.
    merge_gap_ms : int
        Merge segments closer than this to avoid fragmenting utterances.

    Returns
    -------
    list[tuple[int, int]]
        List of (start_sample, end_sample) ranges.
    """
    if audio.size == 0:
        return []

    if sample_rate != 16_000:
        raise ValueError(f"Silero VAD requires 16 kHz audio, got {sample_rate} Hz")
    model = _get_vad()

    # Reset model state between calls
    model.reset_states()

    # Silero works on 512-sample frames (32 ms at 16 kHz)
    window_size = 512
    tensor = torch.from_numpy(audio).float()

    # Compute speech probability for each frame
    speech_probs: list[float] = []
    for i in range(0, len(tensor) - window_size + 1, window_size):
        chunk = tensor[i : i + window_size]
        prob = model(chunk, sample_rate).item()
        speech_probs.append(prob)

    if not speech_probs:
        return []

    # Threshold to binary voiced/unvoiced
    segments: list[tuple[int, int]] = []
    start_frame: int | None = None
    for i, prob in enumerate(speech_probs):
        if prob >= threshold and start_frame is None:
            start_frame = i
        elif prob < threshold and start_frame is not None:
            start_sample = start_frame * window_size
            end_sample = min(len(audio), i * window_size)
            segments.append((start_sample, end_sample))
            start_frame = None

    # Close any open segment
    if start_frame is not None:
        segments.append((start_frame * window_size, len(audio)))

    if not segments:
        return []

    # --- Merge segments closer than merge_gap_ms ---
    merge_gap_samples = int(sample_rate * merge_gap_ms / 1000)
    merged: list[tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # --- Pad segments ---
    pad_samples = int(sample_rate * pad_ms / 1000)
    padded: list[tuple[int, int]] = []
    for start, end in merged:
        padded_start = max(0, start - pad_samples)
        padded_end = min(len(audio), end + pad_samples)
        padded.append((padded_start, padded_end))

    # --- Re-merge after padding (padding may cause overlap) ---
    final: list[tuple[int, int]] = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = final[-1]
        if start <= prev_end:
            final[-1] = (prev_start, max(prev_end, end))
        else:
            final.append((start, end))

    # --- Filter by minimum duration ---
    min_samples = int(sample_rate * min_segment_ms / 1000)
    final = [(s, e) for s, e in final if (e - s) >= min_samples]

    return final


def extract_speech(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.35,
    min_speech_ms: int = 250,
    min_segment_ms: int = 500,
    pad_ms: int = 100,
    merge_gap_ms: int = 300,
) -> tuple[np.ndarray, list[tuple[float, float]], list[np.ndarray]]:
    """Extract speech regions and return both concatenated and individual segments.

    Returns
    -------
    tuple
        - concatenated speech waveform (np.ndarray)
        - list of (start_seconds, end_seconds) timestamps
        - list of individual speech segment arrays (for multi-segment embedding)
    """
    segments = detect_speech_segments(
        audio,
        sample_rate,
        threshold=threshold,
        min_segment_ms=min_segment_ms,
        pad_ms=pad_ms,
        merge_gap_ms=merge_gap_ms,
    )

    if not segments:
        return np.array([], dtype=np.float32), [], []

    # Extract individual segment arrays
    segment_arrays: list[np.ndarray] = []
    timestamps: list[tuple[float, float]] = []
    for start, end in segments:
        segment_arrays.append(audio[start:end])
        timestamps.append((start / sample_rate, end / sample_rate))

    # Check total speech duration
    total_speech_samples = sum(len(s) for s in segment_arrays)
    if total_speech_samples * 1000 / sample_rate < min_speech_ms:
        return np.array([], dtype=np.float32), [], []

    concatenated = np.concatenate(segment_arrays)
    return concatenated, timestamps, segment_arrays
