"""Pinned model files.

The server loads both models only from local files whose SHA-256 matches
these pins; it never downloads anything or runs downloaded code at runtime.
Provision the files once, on a machine with internet access:

    python -m scripts.fetch_models

Changing a model means changing these pins in code, then re-embedding every
stored clip (python -m scripts.reembed). Embeddings from different models live
in different vector spaces and must never be compared.
"""

import hashlib
from pathlib import Path

from src.config import MODEL_DIR

# ECAPA-TDNN speaker embeddings (SpeechBrain, VoxCeleb), pinned to one revision.
SPEAKER_REPO = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
SPEAKER_DIR = MODEL_DIR / "speaker"
# Local file name -> SHA-256. label_encoder.ckpt is the repo's label_encoder.txt,
# renamed the way SpeechBrain stores it.
SPEAKER_FILES = {
    "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    "label_encoder.ckpt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
}

# Silero VAD v5.1.2, TorchScript build.
VAD_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.jit"
VAD_FILE = MODEL_DIR / "silero_vad.jit"
VAD_SHA256 = "85c48e1f0ecb604e5d2a268f3ccfb912d4f7e935acdc86af5a3fc5b0aea7b29a"

# Stored with every embedding. Only embeddings with the current ID are searched.
EMBEDDING_MODEL_ID = f"{SPEAKER_REPO}@{SPEAKER_REVISION[:12]}"


class ModelFilesError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ModelFilesError(f"Model file {path} is missing. Run: python -m scripts.fetch_models")
    actual = sha256(path)
    if actual != expected:
        raise ModelFilesError(
            f"Model file {path} does not match its pinned checksum (got {actual[:12]}…, "
            f"expected {expected[:12]}…). Delete it and run: python -m scripts.fetch_models"
        )


def verify_speaker_model() -> None:
    for name, expected in SPEAKER_FILES.items():
        verify(SPEAKER_DIR / name, expected)


def verify_vad_model() -> None:
    verify(VAD_FILE, VAD_SHA256)
