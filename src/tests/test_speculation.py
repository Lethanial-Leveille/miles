"""Both capture loops must speculate, and both must give up on resumed speech.

This file exists because speculative transcription was written into
record_command and never into listen_for_followup, and nothing noticed for as
long as the feature had existed. Phase 2 of listen_for_followup is a deliberate
copy of record_command's loop, so the two drift apart silently by construction:
the copy still worked, still transcribed, still returned the right text, and
just paid the full Whisper cost every time.

It was not visible in the aggregate either. The median transcribe time across
both turn types sat in between the two, looking like one unremarkable number
rather than a bimodal distribution with follow ups pinned to the slow mode.
Splitting timing_log by turn_type is what exposed it: 671ms on initial turns
against 1187ms on follow ups, same window, same config.

Two layers here, because audio.py cannot be imported while miles-voice is
running. It opens the mic, loads the wake model and the encoder, and takes an
exclusive lock at import, all deliberately. So:

  1. The AST tests parse audio.py as text and always run, including on the Pi
     with the service live. They pin the structural invariant that actually
     broke: both loops call the shared helper.
  2. The behavioural tests import audio and drive the real loops against a fake
     stream. They are the stronger check and they skip when the mic is locked,
     which on this Pi is most of the time. Stop miles-voice to run them.

The AST layer is not a substitute for the behavioural one. It cannot tell that
the call is reached, only that it is written. It is here because a test that
skips on the machine that matters is not protection.
"""

import ast
import os
import pathlib

import pytest

AUDIO_PY = pathlib.Path(__file__).resolve().parent.parent / "audio.py"

CAPTURE_LOOPS = ["record_command", "listen_for_followup"]


def _function(name):
    tree = ast.parse(AUDIO_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from audio.py")


def _calls(name):
    return {n.func.id for n in ast.walk(_function(name))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


# ── structural: runs everywhere, including against a live service ──

@pytest.mark.parametrize("loop", CAPTURE_LOOPS)
def test_capture_loop_starts_speculation(loop):
    assert "maybe_speculate" in _calls(loop), (
        f"{loop} never starts a speculative transcription, so every turn it "
        f"captures pays the full Whisper cost after endpointing rather than "
        f"during it.")


@pytest.mark.parametrize("loop", CAPTURE_LOOPS)
def test_capture_loop_cancels_speculation(loop):
    assert "cancel_speculation" in _calls(loop), (
        f"{loop} never discards a speculation, so a run started mid sentence "
        f"can be collected as if it covered the whole turn.")


def test_neither_loop_inlines_the_speculation(loop=None):
    """The helper exists because inlining is what let the two drift apart.

    An inlined copy would pass both tests above only by also calling the
    helper, so this pins the other direction: no loop constructs a _Speculation
    directly."""
    for name in CAPTURE_LOOPS:
        node = _function(name)
        inlined = {n.func.id for n in ast.walk(node)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "_Speculation"}
        assert not inlined, (
            f"{name} builds a _Speculation itself. Route it through "
            f"maybe_speculate so both loops keep the same rules.")


# ── behavioural: the real loops, when the mic is free ──

def _mic_is_locked():
    """True when miles-voice (or anything else) holds the capture lock."""
    import fcntl
    path = os.path.expanduser("~/miles/build/mic.lock")
    try:
        fd = open(path, "w")
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        fd.close()


live = pytest.mark.skipif(
    _mic_is_locked(),
    reason="miles-voice holds the mic lock, so audio.py cannot be imported. "
           "Stop the service to run these.")


class FakeStream:
    """Yields a scripted speech pattern one VAD frame at a time.

    The pattern is a string: 's' for a frame of speech, '.' for silence. Frames
    are opaque, because _is_speech is patched to read the script rather than the
    audio. That keeps these tests about loop control flow rather than about
    webrtcvad, which has its own coverage."""

    def __init__(self, script):
        self.script = script
        self.index = 0

    def read(self, *args, **kwargs):
        # Runs off the end as silence, so a loop that never breaks still
        # terminates on its own max_chunks guard instead of hanging the suite.
        frame = self.script[self.index] if self.index < len(self.script) else '.'
        self.index += 1
        return frame.encode()


# One second of speech, then silence long enough to endpoint. SILENCE_LIMIT is
# 1.2s and speculation fires at 450ms, so the pause is long enough for both.
SPEAKS_THEN_STOPS = 's' * 34 + '.' * 60

# Speech, a pause past the speculation point, more speech, then a real stop.
# The first speculation covers only half the turn and has to be discarded.
SPEAKS_PAUSES_RESUMES = 's' * 34 + '.' * 20 + 's' * 20 + '.' * 60


@pytest.fixture
def spy(monkeypatch):
    """Patch out the device and the Whisper subprocess, recording every
    speculation started and cancelled."""
    import audio

    started, cancelled = [], []

    class FakeSpeculation:
        def __init__(self, frames):
            self.stale = False
            self.frames = frames
            started.append(frames)

        def cancel(self):
            self.stale = True
            cancelled.append(self.frames)

        def result(self):
            return "fake transcript"

    monkeypatch.setattr(audio, "_Speculation", FakeSpeculation)
    monkeypatch.setattr(audio, "_pending_speculation", None)
    monkeypatch.setattr(audio, "_is_speech", lambda data: data == b's')
    monkeypatch.setattr(audio, "_write_wav", lambda frames, path=None: "/fake.wav")
    monkeypatch.setattr(audio, "SPECULATIVE_TRANSCRIBE", True)
    monkeypatch.setattr(audio.timing, "note_speech_end", lambda *a: None)
    monkeypatch.setattr(audio.timing, "mark", lambda *a: None)

    def install(script):
        monkeypatch.setattr(audio, "stream", FakeStream(script))

    return audio, install, started, cancelled


@live
@pytest.mark.parametrize("loop", CAPTURE_LOOPS)
def test_loop_speculates_on_a_clean_stop(spy, loop):
    audio, install, started, _ = spy
    install(SPEAKS_THEN_STOPS)
    getattr(audio, loop)()
    assert len(started) == 1


@live
@pytest.mark.parametrize("loop", CAPTURE_LOOPS)
def test_speaking_through_a_pause_discards_and_retries(spy, loop):
    """A speculation started mid sentence transcribes half a turn.

    Both loops have to cancel it and let the next pause start a fresh one, or
    the turn gets answered on a truncated transcript, which is far worse than
    paying full price for the transcription."""
    audio, install, started, cancelled = spy
    install(SPEAKS_PAUSES_RESUMES)
    getattr(audio, loop)()

    assert len(started) == 2, "expected a discarded attempt and a real one"
    assert len(cancelled) == 1
    # The kept one saw strictly more audio than the discarded one.
    assert len(started[1]) > len(started[0])


@live
@pytest.mark.parametrize("loop", CAPTURE_LOOPS)
def test_speculation_is_off_when_disabled(spy, monkeypatch, loop):
    audio, install, started, _ = spy
    monkeypatch.setattr(audio, "SPECULATIVE_TRANSCRIBE", False)
    install(SPEAKS_THEN_STOPS)
    getattr(audio, loop)()
    assert started == []
