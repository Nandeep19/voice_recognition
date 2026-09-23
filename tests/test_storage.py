"""Storage and matching behaviour.

Offline tests cover validation and helpers. Tests taking the ``pg`` fixture
run against a real PostgreSQL+pgvector database (TEST_DATABASE_URL) and cover
what the storage exists for: atomic enrollment with the audio stored,
cascading delete, and correct similarity scores from pgvector.
"""

import json
import sys

import numpy as np
import pytest

from tests.conftest import EMBEDDING_DIM, fake_analyse, voice_clip


def _clips(*datas: bytes) -> list:
    return [(f"clip{i}.wav", data) for i, data in enumerate(datas, start=1)]


def _person(**overrides) -> dict:
    data = {"person_id": None, "name": "Test Person", "gender": "M", "crime_type": "Theft", "status": "Active"}
    data.update(overrides)
    return data


# ------------------------------------------------------------------ offline --

class TestNormalise:
    def test_output_is_unit_length(self):
        from src.database import normalise

        assert np.isclose(np.linalg.norm(normalise(np.arange(1, EMBEDDING_DIM + 1))), 1.0)

    def test_wrong_dimension_refused(self):
        from src.database import normalise

        with pytest.raises(ValueError, match=str(EMBEDDING_DIM)):
            normalise([1.0] * 512)

    def test_zero_vector_refused(self):
        from src.database import normalise

        with pytest.raises(ValueError):
            normalise([0.0] * EMBEDDING_DIM)


class TestEnrollValidation:
    def test_person_id_with_unsafe_characters_refused(self, fake_voices):
        with pytest.raises(ValueError, match="Person ID"):
            fake_voices.enroll_person(_person(person_id="../etc"), _clips(voice_clip(1)))

    def test_name_required(self, fake_voices):
        with pytest.raises(ValueError, match="Name is required"):
            fake_voices.enroll_person(_person(name="   "), _clips(voice_clip(1)))

    def test_no_clips_refused(self, fake_voices):
        with pytest.raises(ValueError, match="At least one"):
            fake_voices.enroll_person(_person(), [])

    def test_too_many_clips_refused(self, fake_voices):
        with pytest.raises(ValueError, match="At most 5"):
            fake_voices.enroll_person(_person(), _clips(*[voice_clip(1, t) for t in range(6)]))

    def test_clips_of_one_speaker_are_consistent(self, fake_voices):
        clips = [fake_analyse(voice_clip(1, t), f"{t}.wav", 0) for t in range(3)]
        fake_voices._check_enrollment_consistency(clips)

    def test_clip_of_another_speaker_refused(self, fake_voices):
        """A wrong file must not slip into someone's record."""
        clips = [fake_analyse(d, f"{i}.wav", 0) for i, d in enumerate([voice_clip(1, 1), voice_clip(1, 2), voice_clip(2)])]
        with pytest.raises(ValueError, match="Clip 3"):
            fake_voices._check_enrollment_consistency(clips)


class TestAudioGates:
    def test_oversized_upload_refused(self, monkeypatch):
        import src.voice_service as vs

        monkeypatch.setattr(vs, "_MAX_UPLOAD_BYTES", 10)
        with pytest.raises(ValueError, match="larger than"):
            vs._decode_clip(b"x" * 11, "a.wav")

    def test_empty_upload_refused(self):
        import src.voice_service as vs

        with pytest.raises(ValueError, match="Empty"):
            vs._decode_clip(b"", "a.wav")

    def test_overlong_clip_refused(self, monkeypatch):
        import src.voice_service as vs

        monkeypatch.setattr(vs, "decode_audio", lambda data, name: np.zeros(16_000 * 200, dtype=np.float32))
        with pytest.raises(ValueError, match="limit is"):
            vs._decode_clip(b"audio", "a.wav")

    def test_too_little_speech_refused(self, monkeypatch):
        import src.voice_service as vs

        monkeypatch.setattr(vs, "extract_speech", lambda *a, **k: (np.array([], dtype=np.float32), [], []))
        with pytest.raises(ValueError, match="Not enough clear speech"):
            vs._embed_audio(np.zeros(16_000, dtype=np.float32), 3000)


class TestCompare:
    def test_groups_clips_by_speaker(self, fake_voices):
        result = fake_voices.compare_clips(_clips(voice_clip(1, 1), voice_clip(1, 2), voice_clip(2)))

        speakers = [clip["speaker"] for clip in result["clips"]]
        assert speakers[0] == speakers[1] != speakers[2]
        assert len(result["comparisons"]) == 3
        assert len(result["groups"]) == 2

    def test_single_clip_refused(self, fake_voices):
        with pytest.raises(ValueError, match="at least two"):
            fake_voices.compare_clips(_clips(voice_clip(1)))


class TestDeleteEndpoint:
    def test_unknown_person_is_404(self, client, monkeypatch):
        import src.database as db

        def missing(pid):
            raise ValueError(f"Person {pid} does not exist.")

        monkeypatch.setattr(db, "delete_person", missing)
        assert client.delete("/persons/VOICE999999").status_code == 404


# -------------------------------------------------------------- PostgreSQL --

class TestEnrollment:
    def test_enroll_stores_person_clips_and_embeddings(self, pg, fake_voices):
        result = fake_voices.enroll_person(_person(), _clips(voice_clip(10, 1), voice_clip(10, 2)))

        assert result["person_id"] == "VOICE000001"
        person = pg.find_person_by_id("VOICE000001")
        assert [c["url"] for c in person["clips"]] == [
            f"media/VOICE000001/clips/{cid}.wav" for cid in result["clip_ids"]
        ]
        assert person["clips"][0]["filename"] == "clip1.wav"
        assert person["embedding_count"] == 2
        data, content_type = pg.get_clip_audio("VOICE000001", result["clip_ids"][1])
        assert content_type == "audio/wav" and data[:4] == b"RIFF", "stored as WAV"

    def test_clip_is_only_served_under_its_owner(self, pg, fake_voices):
        first = fake_voices.enroll_person(_person(), _clips(voice_clip(1)))
        fake_voices.enroll_person(_person(), _clips(voice_clip(2)))
        assert pg.get_clip_audio("VOICE000002", first["clip_ids"][0]) is None

    def test_failure_after_insert_leaves_nothing(self, pg, fake_voices, monkeypatch):
        """No partial enrollments: person, clips and embeddings go together."""
        real_add = pg.add_person_with_clips

        def add_then_fail(session, *args, **kwargs):
            real_add(session, *args, **kwargs)
            session.flush()  # rows really reach the server...
            raise RuntimeError("simulated crash mid-enrollment")

        monkeypatch.setattr(pg, "add_person_with_clips", add_then_fail)
        with pytest.raises(RuntimeError):
            fake_voices.enroll_person(_person(person_id="VOICE_CRASH"), _clips(voice_clip(20)))

        # ...and are all rolled back.
        assert pg.find_person_by_id("VOICE_CRASH") is None
        assert pg.get_storage_stats()["clips"] == 0
        assert pg.count_embeddings() == 0

    def test_duplicate_person_id_refused(self, pg, fake_voices):
        fake_voices.enroll_person(_person(person_id="VOICE_DUP"), _clips(voice_clip(1)))
        with pytest.raises(ValueError, match="already exists"):
            fake_voices.enroll_person(_person(person_id="VOICE_DUP"), _clips(voice_clip(4)))
        assert pg.count_embeddings() == 1

    def test_generated_id_skips_manually_taken_id(self, pg, fake_voices):
        fake_voices.enroll_person(_person(person_id="VOICE000001"), _clips(voice_clip(1)))
        assert fake_voices.enroll_person(_person(), _clips(voice_clip(4)))["person_id"] == "VOICE000002"

    def test_mixed_speakers_refused_before_storing(self, pg, fake_voices):
        with pytest.raises(ValueError, match="same speaker"):
            fake_voices.enroll_person(_person(), _clips(voice_clip(1), voice_clip(2)))
        assert pg.get_storage_stats()["persons"] == 0


class TestSimilarity:
    def test_exact_match_scores_one_not_minus_one(self, pg, fake_voices):
        """Guards the pgvector sign trap: `<#>` is the NEGATIVE inner product."""
        fake_voices.enroll_person(_person(), _clips(voice_clip(30)))
        result = fake_voices.match_clip(("q.wav", voice_clip(30)))

        assert result["matched"] is True
        assert result["similarity"] == pytest.approx(1.0, abs=1e-4)
        assert result["person"]["person_id"] == "VOICE000001"

    def test_new_take_of_enrolled_speaker_matches(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(30, 1), voice_clip(30, 2), voice_clip(30, 3)))
        result = fake_voices.match_clip(("q.wav", voice_clip(30, 9)))
        assert result["status"] == "match"
        assert 0.85 < result["similarity"] < 1.0

    def test_unrelated_voice_is_unknown(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(30)))
        result = fake_voices.match_clip(("q.wav", voice_clip(99)))
        assert result["matched"] is False
        assert result["status"] == "no_match"
        assert result["similarity"] < 0.5

    def test_centroid_is_computed_in_sql(self, pg, fake_voices):
        """The centroid score must equal cosine to the mean of the stored vectors."""
        from src.database import normalise

        fake_voices.enroll_person(_person(), _clips(voice_clip(5, 1), voice_clip(5, 2)))
        query = np.random.default_rng(1).standard_normal(EMBEDDING_DIM)
        [row] = pg.speaker_similarities(query)

        stored = np.vstack([normalise(fake_analyse(voice_clip(5, t), "", 0)["embedding"]) for t in (1, 2)])
        centroid = stored.mean(axis=0)
        expected = float(normalise(query) @ centroid / np.linalg.norm(centroid))
        assert row["centroid_similarity"] == pytest.approx(expected, abs=1e-5)
        assert sorted(row["similarities"]) == pytest.approx(sorted((stored @ normalise(query)).tolist()), abs=1e-5)

    def test_search_returns_one_row_per_person(self, pg, fake_voices):
        """Several clips of one person must not fill several of the five slots."""
        fake_voices.enroll_person(_person(name="A"), _clips(voice_clip(40, 1), voice_clip(40, 2), voice_clip(40, 3)))
        fake_voices.enroll_person(_person(name="B"), _clips(voice_clip(41)))

        results = fake_voices.search_persons(("q.wav", voice_clip(40)))["results"]
        assert [r["person_id"] for r in results] == ["VOICE000001"]
        assert results[0]["rank"] == 1
        assert results[0]["name"] == "A"

    def test_match_against_person(self, pg, fake_voices):
        fake_voices.enroll_person(_person(name="A"), _clips(voice_clip(50)))
        fake_voices.enroll_person(_person(name="B"), _clips(voice_clip(51)))

        assert fake_voices.match_against_person("VOICE000001", ("q.wav", voice_clip(50, 1)))["matched"] is True
        assert fake_voices.match_against_person("VOICE000002", ("q.wav", voice_clip(50, 1)))["matched"] is False

    def test_confirm_match_stores_clip_for_matching_voice(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(60)))
        result = fake_voices.confirm_match("VOICE000001", ("new.wav", voice_clip(60, 3)))

        assert result["similarity"] > 0.9
        person = pg.find_person_by_id("VOICE000001")
        assert [c["source"] for c in person["clips"]] == ["enroll", "confirm"]
        assert person["embedding_count"] == 2
        assert pg.get_clip_audio("VOICE000001", result["clip_id"]) is not None

    def test_confirm_match_refuses_a_different_voice(self, pg, fake_voices):
        """A stranger's voice must never be attached to someone's record."""
        fake_voices.enroll_person(_person(), _clips(voice_clip(60)))
        with pytest.raises(ValueError, match="does not match"):
            fake_voices.confirm_match("VOICE000001", ("x.wav", voice_clip(70)))
        assert pg.count_embeddings() == 1
        assert pg.get_storage_stats()["clips"] == 1


class TestIdentifyDecision:
    def test_clear_winner_is_a_match(self, pg, fake_voices):
        fake_voices.enroll_person(_person(name="A"), _clips(voice_clip(30)))
        fake_voices.enroll_person(_person(name="B"), _clips(voice_clip(80)))
        result = fake_voices.match_clip(("q.wav", voice_clip(30)))
        assert result["status"] == "match"
        assert result["person"]["person_id"] == "VOICE000001"
        assert result["margin"] > result["required_margin"]
        assert [m["person_id"] for m in result["ranked_matches"]] == ["VOICE000001", "VOICE000002"]

    def test_two_people_scoring_alike_is_ambiguous(self, pg, fake_voices):
        """Asserting an identity when two records fit equally well is wrong."""
        fake_voices.enroll_person(_person(name="A"), _clips(voice_clip(30)))
        fake_voices.enroll_person(_person(name="B"), _clips(voice_clip(30)), allow_duplicate=True)
        result = fake_voices.match_clip(("q.wav", voice_clip(30)))
        assert result["status"] == "ambiguous"
        assert result["matched"] is False
        assert "person" not in result
        assert {c["person_id"] for c in result["candidates"]} == {"VOICE000001", "VOICE000002"}

    def test_nothing_enrolled_is_refused(self, pg, fake_voices):
        with pytest.raises(ValueError, match="No voices are enrolled"):
            fake_voices.match_clip(("q.wav", voice_clip(1)))


class TestDuplicateEnrollment:
    def test_same_voice_twice_is_refused(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(1)))
        with pytest.raises(ValueError, match="already enrolled as VOICE000001"):
            fake_voices.enroll_person(_person(), _clips(voice_clip(1, 5)))
        assert pg.count_embeddings() == 1

    def test_duplicate_can_be_explicitly_allowed(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(1)))
        result = fake_voices.enroll_person(_person(), _clips(voice_clip(1)), allow_duplicate=True)
        assert result["person_id"] == "VOICE000002"


class TestDeletion:
    def test_delete_cascades_clips_and_embeddings(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(1, 1), voice_clip(1, 2)))
        fake_voices.enroll_person(_person(), _clips(voice_clip(4)))
        pg.create_match_log("VOICE000001", "Test Person", 0.9, True)

        result = fake_voices.delete_person_cascade("VOICE000001")

        assert result["removed_persons"] == 1
        assert result["removed_clips"] == 2
        assert result["removed_embeddings"] == 2
        assert pg.get_storage_stats()["clips"] == 1, "other person untouched"
        assert pg.count_embeddings() == 1
        assert len(pg.get_match_logs_for_person("VOICE000001")) == 1, "audit log outlives the person"

    def test_delete_unknown_person_raises(self, pg, fake_voices):
        with pytest.raises(ValueError, match="does not exist"):
            fake_voices.delete_person_cascade("VOICE999999")


class TestIntegrity:
    def test_complete_enrollment_is_consistent(self, pg, fake_voices):
        fake_voices.enroll_person(_person(), _clips(voice_clip(1, 1), voice_clip(1, 2)))
        report = fake_voices.integrity_report()
        assert report["consistent"] is True
        assert report["counts"] == {"persons": 1, "clips": 2, "embeddings": 2}

    def test_person_without_embeddings_is_flagged(self, pg, fake_voices):
        from sqlalchemy import delete

        fake_voices.enroll_person(_person(), _clips(voice_clip(1)))
        with pg.session_scope() as session:
            session.execute(delete(pg.VoiceEmbedding))
        report = fake_voices.integrity_report()
        assert report["persons_without_embeddings"] == ["VOICE000001"]
        assert len(report["clips_without_embedding"]) == 1
        assert report["consistent"] is False


class TestJsonImport:
    def test_profiles_import_once_with_their_ids(self, pg, tmp_path, monkeypatch):
        from scripts import import_json_profiles

        vectors = [np.random.default_rng(s).standard_normal(EMBEDDING_DIM).tolist() for s in (1, 2)]
        (tmp_path / "spk_alice.json").write_text(
            json.dumps(
                {
                    "speaker_id": "spk_alice",
                    "name": "Alice",
                    "created_at": "2026-05-01T10:00:00+00:00",
                    "embeddings": [{"embedding": v} for v in vectors],
                    "centroid": vectors[0],
                    "enrollment_count": 2,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "argv", ["import", "--dir", str(tmp_path)])

        assert import_json_profiles.main() == 0
        assert import_json_profiles.main() == 0, "re-running skips, does not fail"

        person = pg.find_person_by_id("spk_alice")
        assert person["name"] == "Alice"
        assert person["created_at"].startswith("2026-05-01")
        assert person["clips"] == [] and person["embedding_count"] == 2
        report = pg.integrity_report()
        assert report["embeddings_without_clip"] == 2
        assert report["consistent"] is True


class TestAudioBackfill:
    def test_recordings_replace_their_imported_embeddings(self, pg, tmp_path, monkeypatch):
        from scripts import backfill_audio, import_json_profiles
        from tests.conftest import fake_analyse, fake_embedding

        profiles, audio = tmp_path / "profiles", tmp_path / "audio"
        profiles.mkdir(), audio.mkdir()
        (audio / "a1.ogg").write_bytes(voice_clip(7, 1))
        (audio / "stranger.ogg").write_bytes(voice_clip(99))  # a different recording under a listed name
        records = [
            {"embedding": fake_embedding(voice_clip(7, 1)).tolist(), "source_file": "a1.ogg"},
            {"embedding": fake_embedding(voice_clip(7, 2)).tolist(), "source_file": "stranger.ogg"},
            {"embedding": fake_embedding(voice_clip(7, 3)).tolist(), "source_file": "missing.ogg"},
        ]
        (profiles / "spk_a.json").write_text(
            json.dumps({"speaker_id": "spk_a", "name": "A", "embeddings": records, "centroid": records[0]["embedding"]}),
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "argv", ["import", "--dir", str(profiles)])
        import_json_profiles.main()

        monkeypatch.setattr(backfill_audio, "_analyse_clip", fake_analyse)
        monkeypatch.setattr(sys, "argv", ["backfill", "--profiles", str(profiles), "--audio-dir", str(audio)])
        assert backfill_audio.main() == 1, "the mismatched recording is reported"

        person = pg.find_person_by_id("spk_a")
        assert [c["filename"] for c in person["clips"]] == ["a1.ogg"]
        assert person["embedding_count"] == 3, "one replaced, two kept"
        assert pg.integrity_report()["embeddings_without_clip"] == 2

        backfill_audio.main()  # re-running adds nothing
        assert len(pg.find_person_by_id("spk_a")["clips"]) == 1


# ------------------------------------------------------------ model pinning --

class TestModelFiles:
    def test_missing_file_is_refused(self, tmp_path):
        from src.model_files import ModelFilesError, verify

        with pytest.raises(ModelFilesError, match="fetch_models"):
            verify(tmp_path / "absent.ckpt", "0" * 64)

    def test_altered_file_is_refused(self, tmp_path):
        from src.model_files import ModelFilesError, sha256, verify

        path = tmp_path / "model.ckpt"
        path.write_bytes(b"original weights")
        expected = sha256(path)
        verify(path, expected)
        path.write_bytes(b"tampered weights")
        with pytest.raises(ModelFilesError, match="checksum"):
            verify(path, expected)


class TestEmbeddingModelTracking:
    def test_new_embeddings_record_the_current_model(self, pg, fake_voices):
        from sqlalchemy import select

        from src.model_files import EMBEDDING_MODEL_ID

        fake_voices.enroll_person(_person(), _clips(voice_clip(1)))
        with pg.session_scope() as session:
            assert list(session.scalars(select(pg.VoiceEmbedding.model))) == [EMBEDDING_MODEL_ID]

    def test_embeddings_from_another_model_are_never_compared(self, pg, fake_voices):
        """Vectors from different models are not comparable; a match against
        one would be a meaningless score."""
        from sqlalchemy import update

        fake_voices.enroll_person(_person(name="Old"), _clips(voice_clip(30)))
        with pg.session_scope() as session:
            session.execute(update(pg.VoiceEmbedding).values(model="some/older-model@abc"))

        with pytest.raises(ValueError, match="No voices are enrolled"):
            fake_voices.match_clip(("q.wav", voice_clip(30)))
        report = pg.integrity_report()
        assert report["embeddings_from_other_models"] == {"some/older-model@abc": 1}
        assert report["persons_without_embeddings"] == ["VOICE000001"]
        assert report["consistent"] is False

    def test_reembedding_moves_a_clip_to_the_current_model(self, pg, fake_voices):
        from sqlalchemy import select, update

        from src.model_files import EMBEDDING_MODEL_ID

        clip_id = fake_voices.enroll_person(_person(), _clips(voice_clip(30)))["clip_ids"][0]
        with pg.session_scope() as session:
            session.execute(update(pg.VoiceEmbedding).values(model="some/older-model@abc"))
        pg.replace_clip_embedding(clip_id, np.ones(EMBEDDING_DIM))
        with pg.session_scope() as session:
            assert session.scalar(select(pg.VoiceEmbedding.model)) == EMBEDDING_MODEL_ID

    def test_migration_tags_rows_created_before_the_column(self, pg, fake_voices):
        from sqlalchemy import text

        from src.model_files import EMBEDDING_MODEL_ID

        fake_voices.enroll_person(_person(), _clips(voice_clip(1)))
        with pg.get_engine().begin() as conn:
            conn.execute(text("ALTER TABLE voice_embeddings DROP COLUMN model"))
        pg.init_db()
        with pg.get_engine().connect() as conn:
            assert conn.execute(text("SELECT model FROM voice_embeddings")).scalars().all() == [EMBEDDING_MODEL_ID]
            nullable = conn.execute(
                text("SELECT is_nullable FROM information_schema.columns WHERE table_name='voice_embeddings' AND column_name='model'")
            ).scalar()
        assert nullable == "NO"


# -------------------------------------------------------- multi-voice warning --

SR = 16_000


def _seg(seconds: float, speaker: int) -> np.ndarray:
    """Audio whose first sample encodes the speaker, for a fake embedder."""
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    audio[0] = speaker
    return audio


class TestMultiVoiceWarning:
    def _check(self, monkeypatch, parts):
        import src.voice_service as vs

        monkeypatch.setattr(
            vs, "embed", lambda audio, sr=SR: np.random.default_rng(int(audio[0])).standard_normal(EMBEDDING_DIM)
        )
        segments, times, t = [], [], 0.0
        for seconds, speaker in parts:
            segments.append(_seg(seconds, speaker))
            times.append((t, t + seconds))
            t += seconds + 0.5
        return vs._multi_voice_warning(segments, times)

    def test_one_speaker_gives_no_warning(self, monkeypatch):
        assert self._check(monkeypatch, [(3, 1), (4, 1), (2.5, 1)]) is None

    def test_second_voice_is_reported_with_its_time(self, monkeypatch):
        warning = self._check(monkeypatch, [(3, 1), (4, 1), (3, 2)])
        assert warning and "8.0-11.0s" in warning and "different voice" in warning

    def test_short_pieces_join_the_next_long_one(self):
        import src.voice_service as vs

        pieces = vs._pieces_for_voice_check(
            [_seg(0.5, 1), _seg(3, 1), _seg(0.4, 1)], [(0.0, 0.5), (1.0, 4.0), (4.5, 4.9)]
        )
        assert [(round(s, 1), round(e, 1)) for _, s, e in pieces] == [(0.0, 4.9)]

    def test_single_piece_cannot_be_checked(self, monkeypatch):
        assert self._check(monkeypatch, [(5, 1)]) is None


class TestEnrollEndpoint:
    def test_warnings_are_returned_at_the_top_level(self, client, monkeypatch):
        """The console reads fields from the top level, like every other endpoint."""
        import api.enroll_api as enroll_api

        monkeypatch.setattr(
            enroll_api,
            "enroll_person",
            lambda data, uploads, allow: {"success": True, "person_id": "VOICE000009", "warnings": ["two voices"], "message": "ok"},
        )
        body = client.post("/enroll", data={"name": "A"}, files={"clips": ("a.wav", b"x")}).json()
        assert body["warnings"] == ["two voices"]
        assert body["person_id"] == "VOICE000009"
        assert body["data"]["warnings"] == ["two voices"]
