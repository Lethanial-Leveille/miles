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


class _FakeSession:
    """Routes the shared connection through requests.post, which each test
    patches, so nothing here can reach the network. Records opens and warms."""
    opened = 0
    warms = []

    def __init__(self):
        _FakeSession.opened += 1

    def post(self, *a, **k):
        return requests.post(*a, **k)

    def get(self, *a, **k):
        _FakeSession.warms.append(a[0])
        return _Resp({})

    def close(self):
        pass


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setattr(dg, "DEEPGRAM_STT", True)
    monkeypatch.setattr(dg, "DEEPGRAM_API_KEY", "test-key-not-a-real-one")
    monkeypatch.setattr(dg.requests, "Session", _FakeSession)
    monkeypatch.setattr(dg, "_session", None)
    monkeypatch.setattr(dg, "_last_answer", 0.0)
    monkeypatch.setattr(dg, "_warming", False)
    _FakeSession.opened = 0
    _FakeSession.warms = []


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


# ── the connection, kept warm while he talks (Sep 25 2026) ──

import time as _time


def _wait_for(condition, seconds=1.0):
    deadline = _time.monotonic() + seconds
    while not condition() and _time.monotonic() < deadline:
        _time.sleep(0.01)
    return condition()


def test_a_connection_answered_recently_is_reused(monkeypatch):
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _ok("one"))
    assert dg.transcribe_bytes(b"x") == "one"
    assert dg.transcribe_bytes(b"x") == "one"
    assert _FakeSession.opened == 1


def test_a_connection_idle_past_the_limit_is_replaced_not_tried(monkeypatch):
    """Deepgram closes one idle for about 5s; trying it costs a failed request."""
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _ok("one"))
    dg.transcribe_bytes(b"x")
    monkeypatch.setattr(dg, "_last_answer", _time.monotonic() - dg._REUSE_WITHIN_S)
    dg.transcribe_bytes(b"x")
    assert _FakeSession.opened == 2


def test_a_dropped_connection_is_retried_once_on_a_new_one(monkeypatch):
    attempts = []
    def post(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.ConnectionError("RemoteDisconnected")
        return _ok("second try")
    monkeypatch.setattr(dg.requests, "post", post)
    assert dg.transcribe_bytes(b"x") == "second try"
    assert len(attempts) == 2


def test_a_timeout_is_not_retried(monkeypatch):
    """A retry after a timeout would double the wait before whisper takes over."""
    attempts = []
    def post(*a, **k):
        attempts.append(1)
        raise requests.ConnectTimeout("slow")
    monkeypatch.setattr(dg.requests, "post", post)
    assert dg.transcribe_bytes(b"x") is None
    assert len(attempts) == 1


def test_keeping_warm_sends_no_audio_and_is_rate_limited():
    dg.keep_warm()
    assert _wait_for(lambda: len(_FakeSession.warms) == 1)
    dg.keep_warm()
    dg.keep_warm()
    _time.sleep(0.05)
    assert _FakeSession.warms == [dg._WARM_URL], "one warm inside the window"
    assert dg._WARM_URL != dg._ENDPOINT


def test_nothing_is_warmed_without_a_key(monkeypatch):
    monkeypatch.setattr(dg, "DEEPGRAM_API_KEY", None)
    dg.keep_warm()
    _time.sleep(0.05)
    assert _FakeSession.warms == []


def test_the_completion_hook_gets_the_transcript(monkeypatch):
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _ok("hook"))
    heard = []
    run = dg.start(_frames(), on_done=heard.append)
    assert run.result() == "hook"
    assert _wait_for(lambda: heard == ["hook"])


def test_a_cancelled_run_does_not_call_the_hook(monkeypatch):
    def slow(*a, **k):
        _time.sleep(0.1)
        return _ok("late")
    monkeypatch.setattr(dg.requests, "post", slow)
    heard = []
    run = dg.start(_frames(), on_done=heard.append)
    run.cancel()
    _time.sleep(0.2)
    assert heard == []


def test_a_failing_hook_does_not_lose_the_transcript(monkeypatch):
    monkeypatch.setattr(dg.requests, "post", lambda *a, **k: _ok("kept"))
    def broken(text):
        raise RuntimeError("hook bug")
    run = dg.start(_frames(), on_done=broken)
    assert run.result() == "kept"
