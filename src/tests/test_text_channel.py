"""A typed turn is read on a screen and never played through the room speaker.

Sep 14 2026: every message typed in the app was spoken aloud by miles-server,
and /chat returned only after playback, so the app showed Nova thinking while
she was already talking. Nothing here reaches Claude, ElevenLabs or the speaker."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

import brain
from tools import registry


class _FakeStream:
    """Stands in for claude.messages.stream: yields the text it was given, then
    a final message holding whatever tool calls it was given."""

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

    async def get_final_message(self):
        return self._final


@pytest.fixture
def turn(monkeypatch):
    """Runs one whole turn against fakes and records everything played."""
    played = []
    monkeypatch.setattr(brain, "speak", lambda text, *a, **k: played.append(text))
    monkeypatch.setattr(brain, "start_synthesis", lambda text, *a, **k: text)
    monkeypatch.setattr(brain, "_play_with_barge_in",
                        lambda synthesis: (played.append(synthesis), (False, 0.0))[1])
    monkeypatch.setattr(brain, "save_message", lambda *a, **k: None)
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

    def run(channel, *streams, on_text=None):
        remaining = list(streams)
        monkeypatch.setattr(brain, "claude", SimpleNamespace(
            messages=SimpleNamespace(
                stream=lambda **kwargs: (run.calls.append(kwargs), remaining.pop(0))[1])))
        result = asyncio.run(brain.ask_nova_async("hi", device="app", channel=channel,
                                                  on_text=on_text))
        return result, played
    run.calls = []
    return run


def test_a_typed_reply_is_returned_and_never_played(turn):
    result, played = turn("text", _FakeStream(["Twelve credits", " is full time."]))
    assert result.text == "Twelve credits is full time."
    assert played == []


def test_the_same_reply_on_voice_is_played(turn):
    """The control. Without it the test above would pass just as well if the
    fakes had stopped seeing speech at all."""
    result, played = turn("voice", _FakeStream(["Twelve credits", " is full time."]))
    assert result.text == "Twelve credits is full time."
    assert played == ["Twelve credits is full time."]


def test_a_silent_tool_turn_shows_done_without_saying_it(turn):
    timer = SimpleNamespace(type="tool_use", name="set_timer", id="toolu_1",
                            input={"minutes": 5})
    result, played = turn("text", _FakeStream([], [timer]))
    assert result.text == "Done."
    assert played == []


def test_code_written_lines_all_go_through_the_channel_check():
    """A later direct speak call in the turn would bring the bug straight back."""
    source = inspect.getsource(brain.ask_nova_async)
    assert "run_in_executor(None, speak" not in source
    assert "_tts_consumer(" not in source


def test_a_typed_turn_is_told_to_write_numerals_after_the_clock():
    """The history is mostly spoken replies with numbers in words, and it beat
    the numerals rule in the system prompt."""
    from prompts import TEXT_TURN_NOTE
    messages = [{"role": "user", "content": "how was my sleep"}]
    noted = brain._with_text_note(brain._with_current_time(messages), "text")
    assert noted[-1]["content"].endswith(TEXT_TURN_NOTE)
    assert "[Current date and time:" in noted[-1]["content"]
    assert messages[-1]["content"] == "how was my sleep", "the stored message stays raw"


def test_a_voice_turn_gets_no_note():
    messages = [{"role": "user", "content": "how was my sleep"}]
    assert brain._with_text_note(messages, "voice") == messages


def test_the_note_is_used_in_the_turn():
    assert "_with_text_note(recent, channel)" in inspect.getsource(brain.ask_nova_async)


def test_the_reply_is_streamed_as_it_is_written(turn):
    events = []
    result, _ = turn("text", _FakeStream(["Twelve credits", " is full time."]),
                     on_text=lambda kind, text: events.append((kind, text)))
    assert events == [("delta", "Twelve credits"), ("delta", " is full time.")]
    assert "".join(text for _, text in events) == result.text


def test_a_tool_turn_tells_the_reader_to_drop_the_bridge_sentence(turn):
    """The bridge sentence is not part of the answer the tool result produces,
    and it is not what gets saved either."""
    weather = SimpleNamespace(type="tool_use", name="get_weather", id="toolu_1", input={})
    events = []
    result, _ = turn("text",
                     _FakeStream(["Let me check."], [weather]),
                     _FakeStream(["It's 75 degrees."]),
                     on_text=lambda kind, text: events.append((kind, text)))
    assert ("reset", "") in events
    assert events.index(("reset", "")) < events.index(("delta", "It's 75 degrees."))
    assert result.text == "It's 75 degrees."


def test_a_voice_turn_streams_nothing_and_still_speaks(turn):
    """The voice loop passes no callback, so nothing here changes for it."""
    result, played = turn("voice", _FakeStream(["Twelve credits."]))
    assert played == ["Twelve credits."]
    assert result.text == "Twelve credits."


def test_a_turn_that_only_stores_something_still_answers(turn):
    """Sep 15 2026: "my last day is September 25" was answered "Done.", because
    Nova called remember and wrote nothing alongside it."""
    stored = SimpleNamespace(type="tool_use", name="remember", id="toolu_1",
                             input={"content": "Last day is September 25."})
    result, played = turn("text",
                          _FakeStream([], [stored]),
                          _FakeStream(["So you have nine shifts left."]))
    assert result.text == "So you have nine shifts left."
    assert played == []
    assert "tools" not in turn.calls[1], "nothing left to look up, so no tools"


def test_a_follow_up_after_a_lookup_still_offers_tools(turn):
    weather = SimpleNamespace(type="tool_use", name="get_weather", id="toolu_1", input={})
    turn("text", _FakeStream([], [weather]), _FakeStream(["It's 75 degrees."]))
    assert "tools" in turn.calls[1]
