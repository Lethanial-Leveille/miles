"""What she had left to say when he cut her off is held for a go on.

Nothing here reaches Claude, ElevenLabs, the microphone or the speaker."""

import asyncio
from types import SimpleNamespace

import pytest

import brain
import resume
from tools import registry


@pytest.fixture(autouse=True)
def nothing_held():
    resume.clear()
    yield
    resume.clear()


# ── the store ──

def test_what_was_kept_is_taken_once():
    resume.keep("and then the rest.", now=100.0)
    assert resume.take(now=101.0) == "and then the rest."
    assert resume.take(now=102.0) is None, "one go on must never say it twice"


def test_nothing_kept_means_nothing_to_take():
    assert resume.take(now=0.0) is None


def test_a_blank_remainder_is_no_remainder():
    resume.keep("   ", now=0.0)
    assert resume.take(now=1.0) is None
    resume.keep("", now=0.0)
    assert resume.take(now=1.0) is None


def test_it_goes_stale_after_the_window_and_stays_gone():
    resume.keep("the rest.", now=0.0)
    assert resume.take(now=resume.RESUME_WINDOW_S + 1) is None
    assert resume.take(now=resume.RESUME_WINDOW_S + 2) is None


def test_the_window_includes_its_last_second():
    resume.keep("the rest.", now=0.0)
    assert resume.take(now=resume.RESUME_WINDOW_S) == "the rest."


def test_a_newer_cut_replaces_the_older_one():
    resume.keep("old.", now=0.0)
    resume.keep("new.", now=5.0)
    assert resume.take(now=6.0) == "new."


def test_keeping_nothing_drops_what_was_there():
    resume.keep("old.", now=0.0)
    resume.keep("", now=5.0)
    assert resume.take(now=6.0) is None


def test_clear_lets_go_of_it():
    resume.keep("old.", now=0.0)
    resume.clear()
    assert resume.take(now=1.0) is None


# ── a whole turn ──

class _FakeStream:
    """Stands in for claude.messages.stream: the text it was given, then a final
    message holding whatever tool calls it was given."""

    def __init__(self, texts, tool_uses=()):
        self._texts = texts
        self._final = SimpleNamespace(
            content=list(tool_uses),
            usage=SimpleNamespace(cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for text in self._texts:
            yield SimpleNamespace(type="content_block_delta",
                                  delta=SimpleNamespace(type="text_delta", text=text))
        for block in self._final.content:
            yield SimpleNamespace(type="content_block_start", content_block=block)

    async def get_final_message(self):
        return self._final


@pytest.fixture
def turn(monkeypatch):
    """Runs one whole turn on fakes. cut_first says whether the room speaker is
    interrupted during the first thing she says, a quarter of the way in."""
    monkeypatch.setattr(brain, "speak", lambda text, *a, **k: None)
    monkeypatch.setattr(brain.phrasebank, "play", lambda key, **k: key)
    monkeypatch.setattr(brain, "save_message", lambda *a, **k: 7)
    monkeypatch.setattr(brain, "last_reply_at", lambda: None)
    monkeypatch.setattr(brain, "delete_message", lambda message_id: None)
    monkeypatch.setattr(brain, "get_recent_messages",
                        lambda limit=20: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr(brain, "get_seed_memories", lambda: [])
    monkeypatch.setattr(brain, "get_episodic_memories", lambda: [])
    monkeypatch.setattr(brain, "memory_manifest", lambda: [])
    monkeypatch.setattr(brain, "build_enhanced_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(brain, "search_memories", lambda *a, **k: [])
    monkeypatch.setattr(brain, "log_tool_call", lambda *a, **k: None)
    monkeypatch.setattr(brain.alerts, "take_for_fold", lambda: [])
    monkeypatch.setattr(registry, "call", lambda name, args: "ran")
    monkeypatch.setattr(brain.memory_pass, "notice_later", lambda *a, **k: None)

    def run(*streams, channel="voice", device="pi", cut_first=False):
        spoken = []

        def play(synthesis):
            spoken.append(synthesis.text)
            return cut_first and len(spoken) == 1, 0.0

        monkeypatch.setattr(brain, "start_synthesis",
                            lambda text, *a, **k: SimpleNamespace(
                                text=text, heard_s=1.0, audio_s=4.0))
        monkeypatch.setattr(brain, "_play_with_barge_in", play)
        remaining = list(streams)
        monkeypatch.setattr(brain, "claude", SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: remaining.pop(0))))
        return asyncio.run(brain.ask_nova_async("hi", device=device, channel=channel))
    return run


THREE = ["Step count alone matters little. You lift four days a week. ",
         "Recovery matters more."]


def test_a_cut_reply_keeps_the_rest_of_the_sentence_and_everything_after(turn):
    turn(_FakeStream(THREE), cut_first=True)
    # Five words, a quarter of the audio: "Step" was heard.
    assert resume.take() == ("count alone matters little. You lift four days a week. "
                             "Recovery matters more.")


def test_a_reply_that_played_through_keeps_nothing(turn):
    turn(_FakeStream(THREE))
    assert resume.take() is None


def test_a_reply_that_played_through_ends_the_chance_to_go_on(turn):
    resume.keep("the old reply's ending.")
    turn(_FakeStream(["Fine."]))
    assert resume.take() is None


def test_a_typed_turn_leaves_it_alone(turn):
    resume.keep("the old reply's ending.")
    turn(_FakeStream(["Fine."]), channel="text")
    assert resume.take() == "the old reply's ending."


def test_a_turn_from_the_phone_leaves_it_alone(turn):
    resume.keep("the old reply's ending.")
    turn(_FakeStream(["Fine."]), device="phone")
    assert resume.take() == "the old reply's ending."


def test_room_talk_she_stays_out_of_leaves_it_alone(turn):
    """It may be what cut her, and it is not an answer to what she was saying."""
    resume.keep("the old reply's ending.")
    ignore = SimpleNamespace(type="tool_use", name="ignore", id="toolu_1", input={})
    turn(_FakeStream([], [ignore]))
    assert resume.take() == "the old reply's ending."


def test_a_cut_during_the_lead_in_keeps_the_answer_he_never_heard(turn):
    """On a tool turn the follow up is collected, never played, once he has cut in."""
    weather = SimpleNamespace(type="tool_use", name="get_weather", id="toolu_1",
                              input={"location": "Gainesville"})
    turn(_FakeStream(["Let me check."], [weather]),
         _FakeStream(["It is sunny. Bring sunglasses."]), cut_first=True)
    assert resume.take() == "me check. It is sunny. Bring sunglasses."
