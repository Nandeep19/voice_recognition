"""ECAPA-TDNN speaker embedding model via SpeechBrain.

Provides single-segment and multi-segment embedding with length-weighted
averaging for robust speaker representations.

The model (~80 MB) is downloaded on first use and loaded once per process.
"""

from __future__ import annotations

import math
from threading import Lock

import numpy as np

from src.config import MODEL_DIR, SPEAKER_MODEL

_speaker_model = None
_speaker_lock = Lock()


def model_loaded() -> bool:
    return _speaker_model is not None


def _get_speaker_model():
    global _speaker_model
    if _speaker_model is None:
        with _speaker_lock:
            if _speaker_model is None:
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy

                _speaker_model = EncoderClassifier.from_hparams(
                    source=SPEAKER_MODEL,
                    savedir=str(MODEL_DIR / "speaker"),
                    local_strategy=LocalStrategy.COPY,
                )
    return _speaker_model


def embed(waveform: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Compute L2-normalized embedding for a single waveform.

    Audio normalization and speech extraction happen before this function.
    The ECAPA model receives the resulting waveform directly so its input
    remains close to the distribution used during model training.
    """
    import torch

    model = _get_speaker_model()
    tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        embedding = model.encode_batch(tensor).squeeze().cpu().numpy()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 1e-8 else embedding


def embed_segments(
    segments: list[np.ndarray],
    sample_rate: int = 16_000,
    min_segment_ms: int = 500,
) -> np.ndarray:
    """Compute a length-weighted average embedding from multiple speech segments.

    Instead of embedding the full concatenated speech, each segment is
    embedded independently and combined with sqrt(duration) weighting.
    This prevents one noisy or long segment from dominating the result.

    Segments shorter than *min_segment_ms* are merged with their nearest
    neighbor before embedding.
    """
    if not segments:
        raise ValueError("No speech segments to embed")

    min_samples = int(sample_rate * min_segment_ms / 1000)

    # Merge short segments with their nearest neighbor
    usable_segments = _merge_short_segments(segments, min_samples)

    if not usable_segments:
        # Fall back to concatenating everything
        usable_segments = [np.concatenate(segments)]

    # Embed each segment and weight by sqrt(duration)
    embeddings: list[np.ndarray] = []
    weights: list[float] = []
    for seg in usable_segments:
        embeddings.append(embed(seg, sample_rate))
        weights.append(math.sqrt(len(seg) / sample_rate))

    # If only one segment, just return its embedding (already normalized)
    if len(embeddings) == 1:
        return embeddings[0]

    # Weighted average
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    averaged = sum(w * e for w, e in zip(weights_arr, embeddings))

    # L2 normalize the result
    norm = np.linalg.norm(averaged)
    return averaged / norm if norm > 1e-8 else averaged


def _merge_short_segments(
    segments: list[np.ndarray],
    min_samples: int,
) -> list[np.ndarray]:
    """Merge segments shorter than min_samples with their neighbors.

    Strategy: iterate and accumulate short segments into a buffer.
    When a long-enough segment is found, flush the buffer by prepending
    it to the current segment.
    """
    if not segments:
        return []

    result: list[np.ndarray] = []
    buffer: list[np.ndarray] = []

    for seg in segments:
        if len(seg) >= min_samples:
            if buffer:
                # Prepend buffered short segments to this one
                buffer.append(seg)
                result.append(np.concatenate(buffer))
                buffer = []
            else:
                result.append(seg)
        else:
            buffer.append(seg)

    # Handle remaining buffered segments
    if buffer:
        if result:
            # Append to the last usable segment
            result[-1] = np.concatenate([result[-1]] + buffer)
        else:
            # All segments were short — concatenate everything
            result.append(np.concatenate(buffer))

    return result
