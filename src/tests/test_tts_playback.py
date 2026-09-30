"""What reaches the speaker, and what the journal hears about it."""

import tts


def test_only_underruns_reach_the_journal(capsys):
    """aplay prints a banner on every start; only a dry buffer is news."""
    tts._report_underruns([
        b"Playing raw data 'stdin' : Signed 16 bit Little Endian, Rate 22050 Hz, Mono\n",
        b"underrun!!! (at least 412.337 ms long)\n",
    ])
    out = capsys.readouterr().out
    assert "speaker ran dry: underrun!!! (at least 412.337 ms long)" in out
    assert "Playing raw data" not in out


def test_aplay_errors_are_read_not_discarded():
    import inspect
    source = inspect.getsource(tts._open_aplay)
    assert "stderr=subprocess.PIPE" in source
    assert "_report_underruns" in source


# ── resending a request that has not answered (Sep 25 2026) ──

import threading
import time
from types import SimpleNamespace

import pytest


def _fake_elevenlabs(monkeypatch, behaviours):
    """Each request takes the next behaviour: (delay before first chunk, chunks,
    error). Records how many requests were made."""
    calls = []
    lock = threading.Lock()

    def stream(**request):
        with lock:
            index = len(calls)
            calls.append(request)
        delay, chunks, error = behaviours[index]
        time.sleep(delay)
        if error:
            raise error
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(tts, "_elevenlabs",
                        SimpleNamespace(text_to_speech=SimpleNamespace(stream=stream)))
    monkeypatch.setattr(tts, "TTS_HEDGE_AFTER_MS", 50)
    return calls


def _play(synthesis):
    return b"".join(synthesis.chunks()), synthesis.error


def test_a_prompt_answer_is_never_sent_twice(monkeypatch):
    calls = _fake_elevenlabs(monkeypatch, [(0, [b"a", b"b"], None)])
    audio, error = _play(tts.Synthesis("Hello."))
    time.sleep(0.1)
    assert (audio, error, len(calls)) == (b"ab", None, 1)


def test_a_stalled_request_is_sent_again_and_the_second_is_played(monkeypatch):
    calls = _fake_elevenlabs(monkeypatch, [(0.5, [b"slow"], None), (0, [b"fast"], None)])
    audio, error = _play(tts.Synthesis("Hello."))
    assert (audio, error, len(calls)) == (b"fast", None, 2)
    assert calls[0] == calls[1], "the same request, voice and all"


def test_the_first_is_kept_when_it_answers_before_the_resend(monkeypatch):
    calls = _fake_elevenlabs(monkeypatch, [(0.08, [b"first"], None), (0.5, [b"second"], None)])
    audio, _ = _play(tts.Synthesis("Hello."))
    assert audio == b"first", "one delivery, never the two interleaved"


def test_a_quick_failure_is_reported_without_a_resend(monkeypatch):
    calls = _fake_elevenlabs(monkeypatch, [(0, [], RuntimeError("401"))])
    audio, error = _play(tts.Synthesis("Hello."))
    time.sleep(0.1)
    assert audio == b"" and isinstance(error, RuntimeError) and len(calls) == 1


def test_when_both_fail_the_failure_is_reported(monkeypatch):
    _fake_elevenlabs(monkeypatch, [(0.2, [], RuntimeError("timeout")),
                                   (0, [], RuntimeError("429"))])
    audio, error = _play(tts.Synthesis("Hello."))
    assert audio == b"" and isinstance(error, RuntimeError)


def test_a_loser_failing_later_does_not_mark_the_winner_failed(monkeypatch):
    _fake_elevenlabs(monkeypatch, [(0.3, [], RuntimeError("late")), (0, [b"ok"], None)])
    synthesis = tts.Synthesis("Hello.")
    audio, _ = _play(synthesis)
    time.sleep(0.4)
    assert audio == b"ok" and synthesis.error is None


def test_resending_can_be_turned_off(monkeypatch):
    calls = _fake_elevenlabs(monkeypatch, [(0.2, [b"only"], None), (0, [b"never"], None)])
    monkeypatch.setattr(tts, "TTS_HEDGE_AFTER_MS", None)
    audio, _ = _play(tts.Synthesis("Hello."))
    assert (audio, len(calls)) == (b"only", 1)


class _FakeAplay:
    """aplay, draining for `drain_s` after stdin closes unless killed."""

    def __init__(self, drain_s):
        self.written = []
        self.killed = False
        self._drain_s = drain_s
        self._closed_at = None
        self.stdin = SimpleNamespace(write=self.written.append, flush=lambda: None,
                                     close=self._close)

    def _close(self):
        self._closed_at = time.monotonic()

    def poll(self):
        if self.killed:
            return -9
        if self._closed_at and time.monotonic() - self._closed_at >= self._drain_s:
            return 0
        return None

    def kill(self):
        self.killed = True

    def wait(self):
        while self.poll() is None:
            time.sleep(0.005)


def _synthesis(chunks, on_chunk=None):
    def gen():
        for i, c in enumerate(chunks):
            if on_chunk:
                on_chunk(i)
            yield c
    return SimpleNamespace(chunks=gen, requested_at=time.monotonic(), error=None)


def _barge_in_setup(monkeypatch, drain_s):
    fake = _FakeAplay(drain_s)
    monkeypatch.setattr(tts, "_open_aplay", lambda: fake)
    monkeypatch.setattr(tts.timing, "note_tts", lambda *a: None)
    return fake


def test_a_wake_word_mid_sentence_stops_the_writing(monkeypatch):
    fake = _barge_in_setup(monkeypatch, drain_s=5.0)
    interrupt = threading.Event()
    synthesis = _synthesis([b"a", b"b", b"c", b"d"],
                           on_chunk=lambda i: i == 2 and interrupt.set())
    tts.play(synthesis, interrupt=interrupt)
    assert fake.written == [b"a", b"b"]
    assert fake.killed


def test_a_wake_word_while_the_speaker_drains_cuts_her_off(monkeypatch):
    """The case the single check missed: everything written, audio still playing."""
    fake = _barge_in_setup(monkeypatch, drain_s=5.0)
    interrupt = threading.Event()
    threading.Timer(0.1, interrupt.set).start()
    started = time.monotonic()
    tts.play(_synthesis([b"a", b"b"]), interrupt=interrupt)
    assert fake.killed
    assert time.monotonic() - started < 1.0


def test_no_wake_word_plays_to_the_end(monkeypatch):
    fake = _barge_in_setup(monkeypatch, drain_s=0.1)
    tts.play(_synthesis([b"a", b"b"]), interrupt=threading.Event())
    assert fake.written == [b"a", b"b"]
    assert not fake.killed
