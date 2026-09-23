"""Media endpoint: serving clips from the database, and refusal of malformed
or mismatched IDs before any audio is returned."""

import pytest

WAV = b"RIFF\x24\x00\x00\x00WAVEstored-wav-bytes"


@pytest.fixture
def stored_clip(monkeypatch):
    """Serve WAV for clip 7 of VOICE000001 only; record every lookup."""
    import src.database as db

    lookups = []

    def fake_get(person_id, clip_id):
        lookups.append((person_id, clip_id))
        if (person_id, clip_id) == ("VOICE000001", 7):
            return WAV, "audio/wav"
        return None

    monkeypatch.setattr(db, "get_clip_audio", fake_get)
    return lookups


class TestClipServing:
    def test_serves_an_enrolled_clip(self, client, stored_clip):
        res = client.get("/media/VOICE000001/clips/7.wav")
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.content == WAV

    def test_clip_of_another_person_is_404(self, client, stored_clip):
        """Clip 7 exists, but not under this person."""
        assert client.get("/media/VOICE000002/clips/7.wav").status_code == 404

    def test_missing_clip_is_404(self, client, stored_clip):
        assert client.get("/media/VOICE000001/clips/8.wav").status_code == 404

    def test_response_is_not_publicly_cacheable(self, client, stored_clip):
        """Personal records must not land in shared caches."""
        cache = client.get("/media/VOICE000001/clips/7.wav").headers.get("cache-control", "")
        assert "private" in cache.lower()
        assert "public" not in cache.lower()


class TestMalformedIds:
    """Malformed IDs are refused before the database is queried."""

    @pytest.mark.parametrize("clip_id", ["abc", "-1", "7.0", "1e3", "..", "9" * 19, ""])
    def test_non_numeric_clip_ids_refused(self, client, stored_clip, clip_id):
        assert client.get(f"/media/VOICE000001/clips/{clip_id}.wav").status_code == 404
        assert stored_clip == []

    @pytest.mark.parametrize(
        "person_id",
        ["..", "../..", "..%2f..%2fsrc", "%2e%2e%2f%2e%2e", "C:", "\\\\server\\share"],
    )
    def test_traversal_style_ids_refused(self, client, stored_clip, person_id):
        assert client.get(f"/media/{person_id}/clips/7.wav").status_code in (404, 400)
        assert stored_clip == []

    def test_person_id_charset_enforced(self, client, stored_clip):
        for bad in ["a.b", "a b", "a;b", "a$b", "a" * 65]:
            assert client.get(f"/media/{bad}/clips/7.wav").status_code in (404, 400)
        assert stored_clip == []
