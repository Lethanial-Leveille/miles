"""Deepgram is an upgrade with whisper underneath it, never a replacement.

Every test here is about the fallback holding. A remote engine that is down,
slow, rate limited or answering strangely must cost no more than today's
behaviour, because whisper is what makes the offline path work at all and it is
still there.

The one distinction worth its own test: None and the empty string mean different
things. None is "ask whisper". An empty string is Deepgram saying it heard no
words, which on a false wake is the right answer and better than whisper's
invented sentence, and 45 percent of wakes are false.
"""

import wave

import pytest
import requests

import deepgram_stt as dg


def _frames(n=8):
    return [b"\x00\x01" * 640 for _ in range(n)]


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setattr(dg, "DEEPGRAM_STT", True)
    monkeypatch.setattr(dg, "DEEPGRAM_API_KEY", "test-key-not-a-real-one")


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _ok(text):
    return _Resp({"results": {"channels": [{"alternatives":
                                            [{"transcript": text}]}]}})


def test_the_transcript_comes_back(monkeypatch):
    monkeypatch.setattr(dg.requests, "post",
                        lambda *a, **k: _ok("Set a timer for one minute."))
    assert dg.transcribe_bytes(b"wav") == "Set a timer for one minute."


def test_no_words_is_an_empty_string_and_not_a_failure(monkeypatch):
    """The false wake case. Returning None here would send it to whisper, which
    invents words in 5 of 20 non speech clips, and lose the whole gain."""
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _ok(""))
    assert dg.transcribe_bytes(b"wav") == ""


def test_a_network_failure_asks_whisper(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no route to host")
    monkeypatch.setattr(dg.requests, "post", boom)
    assert dg.transcribe_bytes(b"wav") is None


def test_an_http_error_asks_whisper(monkeypatch):
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _Resp({}, 429))
    assert dg.transcribe_bytes(b"wav") is None


def test_a_response_shaped_wrong_asks_whisper(monkeypatch):
    """A changed API or an error body must not raise into the wake loop."""
    monkeypatch.setattr(dg.requests, "post",
                        lambda *a, **k: _Resp({"unexpected": True}))
    assert dg.transcribe_bytes(b"wav") is None


def test_no_alternatives_is_no_words_rather_than_a_failure(monkeypatch):
    monkeypatch.setattr(dg.requests, "post",
                        lambda *a, **k: _Resp({"results": {"channels": [{"alternatives": []}]}}))
    assert dg.transcribe_bytes(b"wav") == ""


def test_nothing_is_sent_without_a_key(monkeypatch):
    sent = []
    monkeypatch.setattr(dg, "DEEPGRAM_API_KEY", None)
    monkeypatch.setattr(dg.requests, "post",
                        lambda *a, **k: sent.append(1) or _ok("x"))
    assert dg.transcribe_bytes(b"wav") is None
    assert dg.start(_frames()) is None
    assert sent == [], "a request went out with no key configured"


def test_the_flag_turns_it_off_completely(monkeypatch):
    """One switch back to fully local, with nothing else to change."""
    sent = []
    monkeypatch.setattr(dg, "DEEPGRAM_STT", False)
    monkeypatch.setattr(dg.requests, "post",
                        lambda *a, **k: sent.append(1) or _ok("x"))
    assert dg.available() is False
    assert dg.transcribe_bytes(b"wav") is None
    assert dg.start(_frames()) is None
    assert sent == []


def test_smart_format_is_not_requested():
    """It rewrites numbers as digits, and the eval truths and Nova's spoken
    replies are both written in words. The live params must match what was
    measured or the measurement stops describing what runs."""
    assert "smart_format" not in dg._PARAMS
    assert dg._PARAMS["punctuate"] == "true"


def test_frames_become_a_readable_wav_in_memory():
    """In memory because the speculative WAV on disk is overwritten by the next
    speculation while this request may still be reading it."""
    import io
    data = dg._wav_bytes(_frames(4))
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == dg.RATE
        assert w.getnframes() == 4 * 640


def test_a_cancelled_run_reports_nothing(monkeypatch):
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _ok("too late"))
    run = dg.Run(b"wav")
    run.cancel()
    assert run.result(timeout=2) is None


def test_a_run_that_never_answers_gives_up(monkeypatch):
    """Second line of defence behind the request timeout, for a thread wedged
    somewhere requests did not time out."""
    import threading
    release = threading.Event()

    def hang(*a, **k):
        release.wait(5)
        return _ok("eventually")
    monkeypatch.setattr(dg.requests, "post", hang)
    run = dg.Run(b"wav")
    assert run.result(timeout=0.2) is None
    release.set()


def test_an_unreadable_file_asks_whisper():
    assert dg.transcribe_file("/nonexistent/path/to/clip.wav") is None
