"""Sentences are synthesized ahead and played in order. Nothing here reaches
ElevenLabs, the microphone or the speaker."""

import asyncio
import sys
import threading
import time
import types

import brain


def _run(monkeypatch, sentences, play, start=None):
    monkeypatch.setattr(brain, "start_synthesis", start or (lambda sentence: sentence))
    monkeypatch.setattr(brain, "_play_with_barge_in", play)

    async def go():
        queue = asyncio.Queue()
        for sentence in sentences:
            queue.put_nowait(sentence)
        queue.put_nowait(None)
        parts = []
        await asyncio.wait_for(brain._tts_consumer(queue, parts, set()), timeout=5)
        return parts
    return asyncio.run(go())


def test_the_next_sentence_is_synthesizing_before_the_current_one_ends(monkeypatch):
    """The whole point. Sep 13 2026: on eleven_v3 each later sentence opened
    with a median 647ms of silence, because it was not requested until the one
    before it had finished playing."""
    second_started = threading.Event()
    seen_while_first_played = []

    def start(sentence):
        if sentence == "Two.":
            second_started.set()
        return sentence

    def play(synthesis):
        if synthesis == "One.":
            seen_while_first_played.append(second_started.wait(timeout=2))
        return False, 0.0

    assert _run(monkeypatch, ["One.", "Two."], play, start) == ["One.", "Two."]
    assert seen_while_first_played == [True]


def test_never_more_than_the_limit_in_flight(monkeypatch):
    lock, active, peak = threading.Lock(), [0], [0]

    def start(sentence):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        return sentence

    def play(synthesis):
        time.sleep(0.05)  # long enough for the feeder to try to run ahead
        with lock:
            active[0] -= 1
        return False, 0.0

    parts = _run(monkeypatch, [f"Sentence {i}." for i in range(6)], play, start)
    assert len(parts) == 6
    assert peak[0] <= brain._SYNTHESIS_AHEAD


def test_sentences_play_in_the_order_they_were_written(monkeypatch):
    def start(text):
        # The first request is the slowest, so a race would reorder the reply.
        time.sleep(0.05 if text == "One." else 0)
        return text
    order = []
    parts = _run(monkeypatch, ["One.", "Two.", "Three."],
                 lambda s: (order.append(s), (False, 0.0))[1], start)
    assert order == ["One.", "Two... Three."]
    assert parts == ["One.", "Two.", "Three."], "history keeps what she actually wrote"


def test_the_first_sentence_alone_and_the_rest_as_one_request(monkeypatch):
    """Separate requests per sentence were heard as different deliveries."""
    requested = []
    _run(monkeypatch, ["Sure.", "It's due Friday.", "Want a reminder?", "I can set one."],
         lambda s: (False, 0.0), lambda text: requested.append(text) or text)
    assert requested == ["Sure.", "It's due Friday... Want a reminder? I can set one."]


def test_an_interruption_keeps_only_what_was_said(monkeypatch):
    assert _run(monkeypatch, ["One.", "Two.", "Three."], lambda s: (True, 0.0)) == ["One."]


def test_a_failure_starting_synthesis_cannot_hang_the_turn(monkeypatch):
    def broken(sentence):
        raise RuntimeError("no network")
    assert _run(monkeypatch, ["One."], lambda s: (False, 0.0), broken) == []


def test_barge_in_reports_an_interruption_only_when_one_happened(monkeypatch):
    """The wrapper used to read its own stop signal back as an interruption,
    so every sentence counted as interrupted once barge in was on."""
    monkeypatch.setattr(brain, "BARGE_IN", True)
    monkeypatch.setattr(brain, "play", lambda synthesis, interrupt=None: (interrupt.wait(0.2), 12.0)[1])

    def quiet(stop_event):
        stop_event.wait(2)

    def hears_the_wake_word(stop_event):
        stop_event.set()

    for watcher, expected in ((quiet, False), (hears_the_wake_word, True)):
        monkeypatch.setitem(sys.modules, "audio", types.SimpleNamespace(watch_for_interrupt=watcher))
        assert brain._play_with_barge_in("One.") == (expected, 12.0)
