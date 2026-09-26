"""A speculative turn buffers a Claude call and does nothing else.

It runs before the endpoint, on a transcript he may keep adding to, so the one
thing it must never do is act: no saved message, no consumed alert, no tool, no
sound. These drive it against a fake client from the worker thread and replay it
on another loop, the way the real turn will.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import brain
import speculative_turn as st


def _event(text):
    return SimpleNamespace(type="content_block_delta",
                           delta=SimpleNamespace(type="text_delta", text=text))


class _FakeStream:
    def __init__(self, texts, delay, fail=None, tools=()):
        self.texts, self.delay, self.fail, self.tools = texts, delay, fail, list(tools)
        self.closed = threading.Event()
        self.final = SimpleNamespace(content=self.tools, usage=None, texts=texts)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed.set()
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for text in self.texts:
            await asyncio.sleep(self.delay)
            yield _event(text)
        for block in self.tools:
            yield SimpleNamespace(type="content_block_start", content_block=block)
        if self.fail:
            raise self.fail

    async def get_final_message(self):
        return self.final


@pytest.fixture
def claude(monkeypatch):
    """The worker's client, replaced. Records every call's arguments."""
    state = SimpleNamespace(calls=[], streams=[], texts=["One.", " Two."], delay=0.0,
                            fail=None, tools=[], warmed=0)

    def stream(**kwargs):
        state.calls.append(kwargs)
        fake = _FakeStream(state.texts, state.delay, state.fail, state.tools)
        state.streams.append(fake)
        return fake

    monkeypatch.setattr(st, "_client", SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    def warm():
        state.warmed += 1
    monkeypatch.setattr(st.tts, "warm", warm)
    monkeypatch.setattr(brain.registry, "api_schemas", lambda: [{"name": "fake_tool"}])
    return state


def _request(fingerprint="fp"):
    return brain.TurnRequest(model="claude-haiku-4-5", system="system",
                             messages=[{"role": "user", "content": "hi"}],
                             fingerprint=fingerprint)


def _wait(condition, seconds=2.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


def _consume(replay):
    async def go():
        texts = []
        async with replay as stream:
            async for event in stream:
                texts.append(event.delta.text)
            final = await stream.get_final_message()
        return texts, final
    return asyncio.run(go())


def test_a_finished_speculation_replays_every_event_and_the_final(claude):
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: turn._buffer.done)
    texts, final = _consume(turn.claim("fp"))
    assert texts == ["One.", " Two."]
    assert final.texts == ["One.", " Two."]


def test_claimed_midway_it_replays_the_buffer_then_follows_live(claude):
    claude.texts = ["A.", " B.", " C.", " D."]
    claude.delay = 0.05
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: len(turn._buffer.events) >= 1)
    texts, _ = _consume(turn.claim("fp"))
    assert texts == ["A.", " B.", " C.", " D."]


def test_the_call_is_exactly_the_real_turns_first_call(claude):
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: turn._buffer.done)
    assert claude.calls == [brain.first_call_kwargs(_request())]


def test_elevenlabs_is_warmed_when_the_call_goes_out(claude):
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: turn._buffer.done)
    assert claude.warmed == 1


def test_cancelling_midway_closes_the_stream(claude):
    claude.texts = ["A."] * 50
    claude.delay = 0.02
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: len(turn._buffer.events) >= 1)
    turn.cancel()
    assert claude.streams[0].closed.wait(1.0), "Anthropic must stop generating"
    assert _wait(lambda: turn._buffer.done)
    assert isinstance(turn._buffer.error, st.Cancelled)
    assert turn.claim("fp") is None


def test_cancelled_before_the_request_is_built_nothing_is_sent(claude):
    gate = threading.Event()

    def slow_request():
        gate.wait(1.0)
        return _request()

    turn = st.SpeculativeTurn(slow_request)
    turn.cancel()
    gate.set()
    time.sleep(0.1)
    assert claude.calls == []


def test_a_changed_input_is_not_claimed_and_is_dropped(claude):
    claude.texts = ["A."] * 50
    claude.delay = 0.02
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: turn.fingerprint == "fp")
    assert turn.claim("something changed") is None
    assert claude.streams[0].closed.wait(1.0)


def test_a_request_not_built_yet_is_not_claimed(claude):
    gate = threading.Event()
    turn = st.SpeculativeTurn(lambda: (gate.wait(1.0), _request())[1])
    assert turn.claim("fp") is None
    gate.set()


def test_a_speculation_that_already_failed_is_not_claimed(claude):
    claude.fail = RuntimeError("overloaded")
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: turn._buffer.done)
    assert turn.claim("fp") is None


def test_a_failure_after_the_claim_reaches_the_turn_like_a_live_one(claude):
    claude.texts = ["A.", " B."]
    claude.delay = 0.05
    claude.fail = RuntimeError("connection reset")
    turn = st.SpeculativeTurn(_request)
    replay = turn.claim("fp") if _wait(lambda: turn.fingerprint == "fp") else None
    with pytest.raises(RuntimeError, match="connection reset"):
        _consume(replay)


def test_it_acts_on_nothing_even_when_the_model_calls_a_tool(claude, monkeypatch):
    """A tool_use block is only an event in the buffer until the real turn."""
    def forbidden(*a, **k):
        raise AssertionError("a speculation acted")
    for target, name in [(brain, "save_message"), (brain, "delete_message"),
                         (brain.alerts, "take_for_fold"), (brain.pending_action, "begin_turn"),
                         (brain.registry, "call"), (brain, "speak"), (brain, "start_synthesis"),
                         (brain.timing, "mark")]:
        monkeypatch.setattr(target, name, forbidden)
    tool = SimpleNamespace(type="tool_use", name="create_calendar_event", id="toolu_1", input={})
    claude.texts = ["Adding it."]
    claude.tools = [tool]
    turn = st.SpeculativeTurn(_request)
    assert _wait(lambda: turn._buffer.done)
    assert turn._buffer.error is None, "nothing forbidden was reached"
    assert [e.type for e in turn._buffer.events] == ["content_block_delta", "content_block_start"]
    assert turn._buffer.final.content == [tool], "held for the real turn to run"
    turn.cancel()
