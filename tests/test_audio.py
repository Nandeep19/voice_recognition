"""Tests for audio preprocessing — pre-emphasis, bandpass, and pipeline."""

import numpy as np

from app.audio import apply_bandpass_filter, apply_pre_emphasis, preprocess_for_embedding


class TestPreEmphasis:
    def test_first_sample_unchanged(self) -> None:
        audio = np.array([0.5, 0.3, 0.1, -0.2], dtype=np.float32)
        result = apply_pre_emphasis(audio)
        assert result[0] == audio[0]

    def test_boosts_high_frequency(self) -> None:
        """Pre-emphasis should increase energy in high-frequency components."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        # Mix of low (200 Hz) and high (4000 Hz) frequency
        audio = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)
        emphasized = apply_pre_emphasis(audio.astype(np.float32))

        # Compare high-freq energy ratio before and after
        spectrum_before = np.abs(np.fft.rfft(audio))
        spectrum_after = np.abs(np.fft.rfft(emphasized))

        # High-freq bins (above 2000 Hz)
        freq_bins = np.fft.rfftfreq(len(audio), d=1.0 / sr)
        high_mask = freq_bins > 2000

        ratio_before = spectrum_before[high_mask].sum() / (spectrum_before.sum() + 1e-10)
        ratio_after = spectrum_after[high_mask].sum() / (spectrum_after.sum() + 1e-10)
        assert ratio_after > ratio_before

    def test_empty_input(self) -> None:
        result = apply_pre_emphasis(np.array([], dtype=np.float32))
        assert result.size == 0

    def test_single_sample(self) -> None:
        result = apply_pre_emphasis(np.array([0.5], dtype=np.float32))
        assert result[0] == 0.5


class TestBandpassFilter:
    def test_removes_dc_offset(self) -> None:
        """Bandpass should remove DC (0 Hz) component."""
        sr = 16000
        audio = np.ones(sr, dtype=np.float32) * 0.5  # pure DC
        filtered = apply_bandpass_filter(audio, sr, low_hz=80.0)
        assert abs(filtered.mean()) < 0.05  # DC should be mostly gone

    def test_passes_speech_frequencies(self) -> None:
        """Frequencies within 80-7500 Hz should pass through."""
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)  # 1 kHz = speech band
        filtered = apply_bandpass_filter(audio, sr)
        # Energy should be preserved
        assert np.abs(filtered).mean() > 0.3

    def test_short_signal_returns_unchanged(self) -> None:
        audio = np.array([0.1, 0.2], dtype=np.float32)
        result = apply_bandpass_filter(audio, 16000)
        np.testing.assert_array_equal(result, audio)


class TestPreprocessForEmbedding:
    def test_output_is_normalized(self) -> None:
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
        result = preprocess_for_embedding(audio, sr)
        assert np.abs(result).max() <= 1.0 + 1e-6

    def test_empty_returns_empty(self) -> None:
        result = preprocess_for_embedding(np.array([], dtype=np.float32), 16000)
        assert result.size == 0
