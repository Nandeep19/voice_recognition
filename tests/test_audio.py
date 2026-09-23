"""Tests for audio loading — mono downmix and resampling to the model's rate."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import TARGET_SAMPLE_RATE, decode_audio, encode_wav, load_audio


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


class TestStoredForm:
    def test_wav_bytes_round_trip(self) -> None:
        tone = 0.3 * np.sin(np.linspace(0, 200, 8_000, dtype=np.float32))
        data = encode_wav(tone)

        assert data[:4] == b"RIFF", "stored clips are WAV"
        audio, sample_rate = load_audio(data)
        assert sample_rate == TARGET_SAMPLE_RATE
        np.testing.assert_allclose(audio, tone, atol=1e-4)

    def test_ffmpeg_normalises_to_mono_16k(self, tmp_path: Path) -> None:
        """Needs FFmpeg on PATH, like the app itself."""
        import shutil

        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg not installed")
        stereo = np.stack([np.full(44_100, 0.5), np.full(44_100, -0.1)], axis=1).astype(np.float32)
        path = tmp_path / "clip.wav"
        sf.write(path, stereo, 44_100, subtype="PCM_16")

        audio = decode_audio(path.read_bytes(), "clip.wav")

        assert audio.ndim == 1
        assert len(audio) == pytest.approx(TARGET_SAMPLE_RATE, rel=0.01)
        assert audio.mean() == pytest.approx(0.2, abs=0.01)

    def test_undecodable_bytes_are_refused(self) -> None:
        import shutil

        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg not installed")
        with pytest.raises(ValueError, match="Could not decode"):
            decode_audio(b"this is not audio", "x.wav")


class TestNothingOnDisk:
    """Uploads are decoded in memory; audio is stored only in the database."""

    def _encode(self, tmp_path: Path, name: str, *ffmpeg_args: str) -> bytes:
        import shutil
        import subprocess

        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg not installed")
        out = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=1", *ffmpeg_args, str(out)],
            check=True,
        )
        return out.read_bytes()

    @pytest.mark.parametrize("name", ["clip.wav", "clip.ogg", "clip.mp3", "clip.flac"])
    def test_common_formats_never_touch_disk(self, tmp_path, monkeypatch, name) -> None:
        import tempfile

        data = self._encode(tmp_path, name)

        def no_disk(*args, **kwargs):
            raise AssertionError("decode_audio wrote a temporary file")

        monkeypatch.setattr(tempfile, "mkstemp", no_disk)
        audio = decode_audio(data, name)
        assert len(audio) == pytest.approx(TARGET_SAMPLE_RATE, rel=0.05)

    def test_m4a_with_index_at_end_still_decodes(self, tmp_path) -> None:
        """Needs seeking, so it goes through a temporary file that is removed."""
        import tempfile

        data = self._encode(tmp_path, "clip.m4a", "-c:a", "aac")
        before = set(Path(tempfile.gettempdir()).glob("*.m4a"))

        audio = decode_audio(data, "clip.m4a")

        assert len(audio) == pytest.approx(TARGET_SAMPLE_RATE, rel=0.05)
        assert set(Path(tempfile.gettempdir()).glob("*.m4a")) == before, "temporary file removed"


class TestVadThreadSafety:
    """Silero VAD is recurrent. One shared instance used by concurrent requests
    crashed the whole process, so every thread must get its own."""

    def _vad(self):
        try:
            from src import vad

            vad._get_vad()
        except Exception as exc:  # no network and no local torch hub cache
            pytest.skip(f"Silero VAD unavailable: {exc}")
        return vad

    def test_each_thread_gets_its_own_model(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        import threading

        vad = self._vad()
        barrier = threading.Barrier(3)  # forces three distinct threads

        def model_id(_):
            barrier.wait()
            return id(vad._get_vad())

        with ThreadPoolExecutor(3) as ex:
            ids = set(ex.map(model_id, range(3)))
        assert len(ids) == 3

    def test_concurrent_detection_matches_sequential(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        vad = self._vad()
        rng = np.random.default_rng(0)
        t = np.arange(TARGET_SAMPLE_RATE * 3) / TARGET_SAMPLE_RATE
        # Bursts of a voiced-like harmonic signal separated by silence.
        clips = []
        for seed in range(4):
            gate = (np.sin(2 * np.pi * (0.5 + seed * 0.2) * t) > 0).astype(np.float32)
            voiced = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), start=1))
            clips.append((0.3 * gate * voiced + 0.01 * rng.standard_normal(t.size)).astype(np.float32))

        sequential = [vad.detect_speech_segments(c, TARGET_SAMPLE_RATE) for c in clips]
        with ThreadPoolExecutor(4) as ex:
            concurrent = list(ex.map(lambda c: vad.detect_speech_segments(c, TARGET_SAMPLE_RATE), clips * 3))
        assert concurrent == sequential * 3

    def test_wrong_sample_rate_is_an_error_not_an_assert(self) -> None:
        from src import vad

        with pytest.raises(ValueError, match="16 kHz"):
            vad.detect_speech_segments(np.zeros(8000, dtype=np.float32), 8000)
