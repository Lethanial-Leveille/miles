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
    monkeypatch.setattr(brain, "play", lambda synthesis, interrupt=None, her=None: (interrupt.wait(0.2), 12.0)[1])

    def quiet(stop_event, her=None):
        stop_event.wait(2)

    def hears_the_wake_word(stop_event, her=None):
        stop_event.set()

    for watcher, expected in ((quiet, False), (hears_the_wake_word, True)):
        monkeypatch.setitem(sys.modules, "audio", types.SimpleNamespace(watch_for_interrupt=watcher))
        assert brain._play_with_barge_in("One.") == (expected, 12.0)


# ── what reached the room, for her history (Oct 2 2026) ──

class _Synth:
    """A synthesis that was cut heard_s into audio_s of audio."""

    def __init__(self, text, heard_s=None, audio_s=None):
        self.text, self.heard_s, self.audio_s = text, heard_s, audio_s


def test_a_cut_reply_keeps_what_played_and_where_it_stopped(monkeypatch):
    """The first sentence played whole; the rest was cut a quarter of the way in."""
    synths = {}

    def start(text):
        synths[text] = _Synth(text, heard_s=1.0 if text != "Step count alone matters little." else None,
                              audio_s=4.0)
        return synths[text]

    def play(synthesis):
        return synthesis.heard_s is not None, 0.0

    monkeypatch.setattr(brain, "start_synthesis", start)
    monkeypatch.setattr(brain, "_play_with_barge_in", play)
    monkeypatch.setattr(brain, "join_for_speech", lambda sentences: " ".join(sentences))
    heard = brain._Heard()

    async def go():
        queue = asyncio.Queue()
        for sentence in ["Step count alone matters little.", "You lift four days a week.",
                         "Recovery matters more than steps."]:
            queue.put_nowait(sentence)
        queue.put_nowait(None)
        await asyncio.wait_for(brain._tts_consumer(queue, [], set(), heard), timeout=5)
    asyncio.run(go())

    assert heard.cut
    # The rest is 11 words, a quarter of its audio played: round(2.75) is 3.
    assert heard.text() == f"Step count alone matters little. You lift four {brain.CUT_OFF_MARK}"


def test_a_reply_played_through_is_not_cut(monkeypatch):
    heard = brain._Heard()
    heard.played(["One.", "Two."])
    assert not heard.cut and heard.text().startswith("One. Two.")


# ── what she had left to say (Oct 30 2026) ──

def test_nothing_is_left_when_the_reply_played_through():
    heard = brain._Heard()
    heard.played(["One.", "Two."])
    assert heard.remainder(["One.", "Two."]) == ""


def test_the_rest_of_the_cut_sentence_comes_first_then_what_never_played():
    heard = brain._Heard()
    heard.played(["Step count alone matters little."])
    heard.cut_during(["You lift four days a week.", "Recovery matters more than steps."],
                     _Synth("x", heard_s=1.0, audio_s=4.0))
    # Eleven words, a quarter played: three heard, eight left.
    assert heard.remainder(["Step count alone matters little.", "You lift four days a week.",
                            "Recovery matters more than steps.", "So sleep first."]) == (
        "days a week. Recovery matters more than steps. So sleep first.")


def test_a_cut_with_no_known_position_leaves_the_sentences_after_it():
    """Without heard_s the cut piece counts as heard, as history already does."""
    heard = brain._Heard()
    heard.cut_during(["One."], _Synth("x"))
    assert heard.remainder(["One.", "Two.", "Three."]) == "Two. Three."


def test_the_remainder_carries_no_emotion_cue():
    heard = brain._Heard()
    heard.cut_during(["One."], _Synth("x"))
    assert heard.remainder(["One.", "[calmly] Two."]) == "Two."


def test_the_remainder_is_empty_when_he_cut_her_at_the_last_word():
    heard = brain._Heard()
    heard.cut_during(["One two."], _Synth("x", heard_s=60.0))
    assert heard.remainder(["One two."]) == ""


def test_what_is_written_after_the_cut_is_still_left(monkeypatch):
    """The consumer is done the moment he cuts in, but Claude is not: the
    sentences it writes afterwards go on the router's record and nowhere else."""
    from stream_router import StreamRouter

    monkeypatch.setattr(brain, "start_synthesis",
                        lambda text: _Synth(text, heard_s=1.0, audio_s=4.0))
    monkeypatch.setattr(brain, "_play_with_barge_in", lambda synthesis: (True, 0.0))
    monkeypatch.setattr(brain, "join_for_speech", lambda sentences: " ".join(sentences))
    heard, written = brain._Heard(), []

    async def go():
        queue = asyncio.Queue()
        router = StreamRouter(queue, record=written)
        consumer = asyncio.create_task(brain._tts_consumer(queue, [], set(), heard))
        await router.feed("Step count alone matters little. ")
        await asyncio.wait_for(consumer, timeout=5)      # cut during the first sentence
        await router.feed("You lift four days a week. Recovery matters more.")
        await router.finalize()
    asyncio.run(go())

    assert heard.text() == f"Step {brain.CUT_OFF_MARK}"
    assert heard.remainder(written) == ("count alone matters little. "
                                        "You lift four days a week. Recovery matters more.")


def test_the_cut_mark_survives_the_history_trim():
    long_reply = " ".join(["word"] * 60) + " " + brain.CUT_OFF_MARK
    trimmed = brain._trim_history([{"role": "assistant", "content": long_reply}])
    assert trimmed[0]["content"].endswith(brain.CUT_OFF_MARK)
    assert len(trimmed[0]["content"].split()) < 40


def test_her_closing_question_survives_the_history_trim():
    """Oct 9 2026: the trim cut "When do they said they'd get back to you?" off
    a 49 word reply, and she asked it again two turns later."""
    reply = " ".join(["word"] * 40) + ". When do you expect to hear back?"
    trimmed = brain._trim_history([{"role": "assistant", "content": reply}])
    assert trimmed[0]["content"].endswith("... When do you expect to hear back?")


def test_a_question_inside_the_head_is_not_said_twice():
    reply = " ".join(["word"] * 27) + ". When do you expect to hear back?"
    trimmed = brain._trim_history([{"role": "assistant", "content": reply}])
    assert trimmed[0]["content"].count("When do") == 1


def test_a_closing_statement_is_still_trimmed():
    reply = " ".join(["word"] * 40) + ". That is all."
    trimmed = brain._trim_history([{"role": "assistant", "content": reply}])
    assert trimmed[0]["content"].endswith("...")


def test_a_cut_reply_keeps_no_question_he_never_heard():
    reply = " ".join(["word"] * 40) + ". Did it go well? " + brain.CUT_OFF_MARK
    trimmed = brain._trim_history([{"role": "assistant", "content": reply}])
    assert "Did it go well?" not in trimmed[0]["content"]


def test_once_cut_the_rest_of_a_tool_turn_is_never_played():
    """A second player would start a second watcher reading the mic while the
    carrier holds his interruption, and answer what he had stopped."""
    import inspect
    source = inspect.getsource(brain.ask_nova_async)
    assert "_collect_text if heard.cut else _consumer_for(channel, device)" in source
    assert "None if heard.cut else bridge_key" in source
    assert 'heard.text() if heard.cut else final_text' in source
