"""Tests for audio loading — mono downmix and resampling to the model's rate."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio import TARGET_SAMPLE_RATE, load_audio


@pytest.fixture
def wav_path(tmp_path: Path):
    def _write(audio: np.ndarray, sample_rate: int) -> Path:
        path = tmp_path / f"clip_{sample_rate}.wav"
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        return path

    return _write


class TestLoadAudio:
    def test_returns_float32_mono_at_target_rate(self, wav_path) -> None:
        tone = 0.3 * np.sin(np.linspace(0, 200, TARGET_SAMPLE_RATE, dtype=np.float32))
        audio, sample_rate = load_audio(wav_path(tone, TARGET_SAMPLE_RATE))

        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert sample_rate == TARGET_SAMPLE_RATE

    def test_stereo_is_downmixed_to_mono(self, wav_path) -> None:
        left = np.full(TARGET_SAMPLE_RATE, 0.5, dtype=np.float32)
        right = np.full(TARGET_SAMPLE_RATE, -0.1, dtype=np.float32)
        stereo = np.stack([left, right], axis=1)

        audio, _ = load_audio(wav_path(stereo, TARGET_SAMPLE_RATE))

        assert audio.ndim == 1
        # Mean of the two channels, within 16-bit quantization error.
        assert audio.mean() == pytest.approx(0.2, abs=1e-3)

    def test_resamples_to_target_rate(self, wav_path) -> None:
        source_rate = 8_000
        tone = 0.3 * np.sin(np.linspace(0, 100, source_rate, dtype=np.float32))

        audio, sample_rate = load_audio(wav_path(tone, source_rate))

        assert sample_rate == TARGET_SAMPLE_RATE
        # One second in, one second out — now at the higher rate.
        assert len(audio) == pytest.approx(TARGET_SAMPLE_RATE, rel=0.01)

    def test_already_at_target_rate_is_not_resampled(self, wav_path) -> None:
        tone = 0.3 * np.sin(np.linspace(0, 200, 4_000, dtype=np.float32))

        audio, sample_rate = load_audio(wav_path(tone, TARGET_SAMPLE_RATE))

        assert sample_rate == TARGET_SAMPLE_RATE
        assert len(audio) == 4_000
