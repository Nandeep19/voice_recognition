"""Download the pinned model files into models/ and verify them.

    python -m scripts.fetch_models

The only step that needs internet access. Files already present with the
right checksum are left alone, so re-running is cheap. Every file is checked
against the SHA-256 pinned in src/model_files.py; a mismatch is deleted and
reported, never used.
"""

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from src import model_files as mf

# Local name -> file name in the Hugging Face repo.
_SPEAKER_SOURCES = {name: name for name in mf.SPEAKER_FILES}
_SPEAKER_SOURCES["label_encoder.ckpt"] = "label_encoder.txt"


def _ok(path: Path, expected: str) -> bool:
    return path.is_file() and mf.sha256(path) == expected


def _install(downloaded: Path, target: Path, expected: str) -> bool:
    actual = mf.sha256(downloaded)
    if actual != expected:
        downloaded.unlink(missing_ok=True)
        print(f"  {target.name}: checksum mismatch ({actual[:12]}…), discarded")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded), target)
    print(f"  {target.name}: downloaded and verified")
    return True


def fetch_speaker_model() -> bool:
    from huggingface_hub import hf_hub_download

    print(f"Speaker model {mf.SPEAKER_REPO}@{mf.SPEAKER_REVISION[:12]}")
    ok = True
    for local, remote in _SPEAKER_SOURCES.items():
        target = mf.SPEAKER_DIR / local
        expected = mf.SPEAKER_FILES[local]
        if _ok(target, expected):
            print(f"  {local}: present")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(hf_hub_download(mf.SPEAKER_REPO, remote, revision=mf.SPEAKER_REVISION, local_dir=tmp))
            ok &= _install(path, target, expected)
    return ok


def fetch_vad_model() -> bool:
    print("Silero VAD v5.1.2")
    if _ok(mf.VAD_FILE, mf.VAD_SHA256):
        print(f"  {mf.VAD_FILE.name}: present")
        return True
    fd, tmp = tempfile.mkstemp(suffix=".jit")
    with open(fd, "wb") as handle, urllib.request.urlopen(mf.VAD_URL, timeout=60) as response:
        shutil.copyfileobj(response, handle)
    return _install(Path(tmp), mf.VAD_FILE, mf.VAD_SHA256)


def main() -> int:
    ok = fetch_speaker_model() & fetch_vad_model()
    print("All model files verified." if ok else "Some model files could not be verified.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
