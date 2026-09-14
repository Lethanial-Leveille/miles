"""The wake word restarts a turn wherever it is heard. Nothing here opens the
microphone or loads the wake model."""

import numpy as np

from parsing import split_after_wake_phrase
from wake_listener import WakeListener


def test_only_the_wake_word_after_the_room():
    """Sep 13 2026: this exact transcript went to Claude and was ignored."""
    assert split_after_wake_phrase("No way. Hey Nova. Hey Nova.") == (True, "")


def test_the_command_after_the_last_wake_word_is_kept():
    assert split_after_wake_phrase("we were just talking hey nova, what's on my schedule") == \
        (True, "what's on my schedule")


def test_no_wake_word_leaves_the_transcript_alone():
    assert split_after_wake_phrase("what's on my schedule") == (False, "what's on my schedule")


def test_a_bare_nova_is_not_the_wake_word():
    """Saying her name to someone else in the room is not addressing her."""
    assert split_after_wake_phrase("tell Nova I said hi") == (False, "tell Nova I said hi")
    assert split_after_wake_phrase("they nova whatever") == (False, "they nova whatever")


def _frame(samples):
    return np.zeros(samples, dtype=np.int16).tobytes()


def test_capture_frames_become_whole_80ms_chunks():
    seen = []
    listener = WakeListener(lambda chunk: seen.append(len(chunk)) or {"hey_nova": 0.0}, 0.4)
    heard = [listener.feed(_frame(480)) for _ in range(8)]   # 3840 samples
    assert seen == [1280, 1280, 1280]
    assert not any(heard)


def test_a_detection_is_reported_once_and_the_buffer_starts_clean():
    scores = iter([0.1, 0.9, 0.1])
    listener = WakeListener(lambda chunk: {"hey_nova": next(scores)}, 0.4)
    assert listener.feed(_frame(1280)) is False
    assert listener.feed(_frame(1280)) is True
    assert listener.feed(_frame(1280)) is False


def test_the_threshold_is_exclusive_like_the_main_loop():
    """voice_main treats a score equal to WAKE_THRESHOLD as a miss."""
    listener = WakeListener(lambda chunk: {"hey_nova": 0.4}, 0.4)
    assert listener.feed(_frame(1280)) is False
