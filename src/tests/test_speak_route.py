"""/speak gives the app Nova's own voice.

Sep 23 2026: the app could show a reply but not hear it, because the only
synthesis path in the repo ended at aplay on the Pi. A phone asking for audio
must get the same voice the room gets and must not be able to reach the room
speaker while doing it."""

import os
import tempfile
from types import SimpleNamespace

import database

# server.py calls init_db() at import. Point the database somewhere throwaway
# first, so importing it here can never touch the real ~/miles/data/miles.db.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import config                                    # noqa: E402
import server                                    # noqa: E402
import tts                                       # noqa: E402


@pytest.fixture
def client():
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


@pytest.fixture
def elevenlabs(monkeypatch):
    """Stand in for the synthesizer: record what was asked for, return bytes.

    Opening the audio device is made an error rather than left alone. That is
    the assertion the room cares about, and it holds for every test here
    without any of them having to remember to make it."""
    calls = []

    def fake_stream(**kwargs):
        calls.append(kwargs)
        # Checked here rather than after the fact: the question is whether the
        # room speaker is locked *while* the phone is being served, and once
        # the call returns an unheld lock proves nothing.
        assert not config.speak_lock.locked(), "the room speaker must stay free"
        yield b"ID3audio"
        yield b"-more"

    monkeypatch.setattr(tts, "_elevenlabs", SimpleNamespace(
        text_to_speech=SimpleNamespace(stream=fake_stream)))
    monkeypatch.setattr(tts, "_open_aplay", lambda: pytest.fail(
        "/speak must never open the audio device on the Pi"))
    monkeypatch.setattr(tts, "get_pronunciations", lambda: [])
    return calls


def test_the_reply_comes_back_as_audio(client, elevenlabs):
    response = client.post("/speak", json={"text": "Twelve credits is full time."})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content == b"ID3audio-more"


def test_the_whole_reply_is_one_request(client, elevenlabs):
    """The reason this endpoint takes a finished reply instead of sentences.

    eleven_v3 voices each request on its own, so two sentences sent separately
    come back in two slightly different deliveries and the voice appears to
    change speaker partway through. _tts_consumer solves it the same way."""
    client.post("/speak", json={"text": "Twelve credits is full time. You have nine left."})
    assert len(elevenlabs) == 1
    assert "nine left" in elevenlabs[0]["text"]


def test_the_app_gets_mp3_and_the_room_still_gets_pcm(client, elevenlabs):
    """Raw PCM is fine down a local pipe and wasteful down a tunnel. The room's
    format is asserted alongside it because play() opens aplay with S16_LE at
    22050 hardcoded, so changing the default here would silently produce noise."""
    client.post("/speak", json={"text": "Hello."})
    assert elevenlabs[0]["output_format"] == config.TTS_APP_OUTPUT_FORMAT
    assert elevenlabs[0]["output_format"].startswith("mp3")

    tts.start_synthesis("Hello.")
    assert elevenlabs[1]["output_format"] == config.TTS_OUTPUT_FORMAT


def test_it_is_the_same_voice_the_room_hears(client, elevenlabs):
    """The whole point of the endpoint. A second voice id anywhere would make
    Nova sound like one person in the room and another on the phone."""
    client.post("/speak", json={"text": "Hello."})
    assert elevenlabs[0]["voice_id"] == config.TTS_VOICE_ID
    assert elevenlabs[0]["model_id"] == config.DEFAULT_TTS_MODEL


def test_the_pronunciation_table_is_applied(client, monkeypatch, elevenlabs):
    """The app displays the real spelling and hears the respelling. Without
    this, his name is said wrong on the phone and right in the room."""
    monkeypatch.setattr(tts, "get_pronunciations", lambda: [("Lethanial", "Luhthanyul", None)])
    monkeypatch.setattr(config, "TTS_PHONEME_TAGS", False)
    client.post("/speak", json={"text": "Morning, Lethanial."})
    assert "Luhthanyul" in elevenlabs[0]["text"]


def test_nothing_to_say_is_rejected(client, elevenlabs):
    assert client.post("/speak", json={"text": "   "}).status_code == 400
    assert elevenlabs == [], "no empty request should reach the synthesizer"


def test_a_screenful_is_rejected_before_it_is_billed(client, elevenlabs):
    response = client.post("/speak", json={"text": "a" * (server._SPEAK_MAX_CHARS + 1)})
    assert response.status_code == 413
    assert elevenlabs == []


def test_a_synthesis_failure_is_a_status_code_not_a_truncated_file(client, monkeypatch):
    """The first chunk is pulled before the response starts precisely so this
    can be a 502. Once a byte is sent the status is already chosen."""
    def fails(**kwargs):
        raise RuntimeError("ElevenLabs is down")
        yield  # noqa: unreachable, makes this a generator

    monkeypatch.setattr(tts, "_elevenlabs", SimpleNamespace(
        text_to_speech=SimpleNamespace(stream=fails)))
    monkeypatch.setattr(tts, "get_pronunciations", lambda: [])
    response = client.post("/speak", json={"text": "Hello."})
    assert response.status_code == 502


def test_it_needs_a_token():
    """Every other route is authenticated and this one carries his voice."""
    with TestClient(server.app) as anonymous:
        # 401, not 403: get_current_user turns a missing or bad token into the
        # same fail closed 401 every other route returns.
        assert anonymous.post("/speak", json={"text": "Hello."}).status_code == 401
