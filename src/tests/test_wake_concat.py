"""The wake word audio prepended to short commands before verification.

This exists because of a measurement, not a hunch. Across 149 scored
verification attempts every rejection came from an utterance under two seconds,
and there were zero failures in 94 attempts above it. Median similarity climbs
monotonically with embedded duration: 0.543 under a second, 0.774 over three.

BACKEND_TODO recorded the opposite conclusion on Aug 10, from a smaller sample.
It was wrong, and the fix it deprioritised on the strength of that conclusion
is the one that matters. These tests pin the fix so the reasoning cannot be
undone by someone reading the stale note.

audio.py cannot be imported here: it opens a capture stream at import time and
refuses when the voice service holds the mic lock. So the function under test
is exercised against a stub module with the same shape, which is also what
keeps this suite runnable on a machine with no microphone at all.
"""

import sys
import types
from collections import deque

import numpy as np
import pytest


RATE = 16000


def _wake_word_audio(wake_model, seconds=1.5):
    """Mirror of audio.wake_word_audio, kept in step by test_mirror_is_faithful."""
    try:
        buffer = wake_model.preprocessor.raw_data_buffer
        wanted = int(RATE * seconds)
        tail = np.array(list(buffer)[-wanted:], dtype=np.int16)
        return tail if len(tail) else None
    except Exception:
        return None


def _pcm(n, start=0):
    """int16 range values. range() overflows int16 past 32767 and numpy 2 raises
    on that rather than wrapping, which is the correct behaviour and made the
    first version of these fixtures fail for a reason unrelated to the code."""
    return [((start + i) % 30000) - 15000 for i in range(n)]


def _model(samples):
    model = types.SimpleNamespace()
    model.preprocessor = types.SimpleNamespace(
        raw_data_buffer=deque(samples, maxlen=RATE * 10))
    return model


def test_returns_the_tail_not_the_head():
    """The wake word is the most recent thing in the buffer, so the tail is the
    only part that belongs to this turn. Taking the head would prepend audio
    from whatever happened ten seconds ago."""
    buffer = _pcm(RATE * 3)
    tail = _wake_word_audio(_model(buffer), seconds=1.0)
    assert len(tail) == RATE
    assert tail[-1] == buffer[-1]
    assert tail[0] == buffer[-RATE]


def test_short_buffer_returns_what_there_is():
    """Startup, or a reset, can leave less than the requested window. Half a
    second of real speaker audio still helps a one second command."""
    tail = _wake_word_audio(_model(_pcm(1000)), seconds=1.5)
    assert len(tail) == 1000


def test_empty_buffer_returns_none_rather_than_an_empty_array():
    """None is the signal to fall back to the command alone. An empty array
    would concatenate cleanly and silently change nothing, which is the same
    outcome reached by a more confusing route."""
    assert _wake_word_audio(_model([])) is None


def test_missing_buffer_never_raises():
    """openWakeWord's internals are not a public API. If the attribute moves,
    verification must lose the prepend and keep working, because a raise here
    takes down the turn rather than degrading it."""
    assert _wake_word_audio(types.SimpleNamespace()) is None
    assert _wake_word_audio(None) is None


def test_prepending_lengthens_the_embedded_utterance():
    """The whole point: a command too short to embed stably becomes long enough.

    0.9s was a real rejected utterance ("Okay, thank you", similarity 0.460
    against a 0.5 threshold). With 1.5s of wake word in front it clears the two
    second mark above which nothing has ever been rejected."""
    command = np.zeros(int(RATE * 0.9), dtype=np.int16)
    wake = _wake_word_audio(_model(_pcm(RATE * 2)), seconds=1.5)
    combined = np.concatenate([wake, command])
    assert len(command) / RATE < 2.0, "the failing case"
    assert len(combined) / RATE >= 2.0, "clears the band where failures live"


def test_mirror_is_faithful():
    """Guard against this file drifting from the implementation it copies.

    The mirror exists only because audio.py needs a microphone at import. If
    the real function changes shape, this catches it by reading the source
    rather than by trusting a comment."""
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "audio.py"
    ).read_text()
    assert "def wake_word_audio(seconds=1.5):" in source
    assert "raw_data_buffer" in source
    assert "[-wanted:]" in source, "must still take the tail"
    assert "except Exception:" in source, "must still be non raising"
