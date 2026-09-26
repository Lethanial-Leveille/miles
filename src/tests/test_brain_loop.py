"""ask_nova keeps one event loop and one client per thread.

A pooled connection belongs to the loop that opened it. When ask_nova called
asyncio.run, every turn got a new loop, the client reused a connection from the
dead one, and the SDK retried after a 0.4 to 0.5s backoff: about 590ms per call
against 75ms on a kept loop, measured Sep 25 2026. These pin the reuse, the
separation between threads, and the cleanup asyncio.run used to do for us.
"""

import asyncio
import threading

import brain


def _capture(monkeypatch):
    seen = []

    async def fake_turn(user_text, **kwargs):
        seen.append((asyncio.get_running_loop(), brain._claude()))
        return brain.TurnResult(text="ok")

    monkeypatch.setattr(brain, "ask_nova_async", fake_turn)
    monkeypatch.setattr(brain, "effective_tier", lambda: "hokage")
    return seen


def _on_fresh_thread(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join()
    return out[0]


def test_turns_on_one_thread_share_loop_and_client(monkeypatch):
    seen = _capture(monkeypatch)

    def two_turns():
        brain.ask_nova("a")
        brain.ask_nova("b")

    _on_fresh_thread(two_turns)
    (loop1, client1), (loop2, client2) = seen
    assert loop1 is loop2
    assert client1 is client2
    assert client1 is not brain.claude


def test_threads_never_share_a_loop_or_client(monkeypatch):
    seen = _capture(monkeypatch)
    _on_fresh_thread(lambda: brain.ask_nova("a"))
    _on_fresh_thread(lambda: brain.ask_nova("b"))
    (loop1, client1), (loop2, client2) = seen
    assert loop1 is not loop2
    assert client1 is not client2


def test_a_loop_of_your_own_gets_the_module_client(monkeypatch):
    """The tests replace brain.claude, and a caller driving ask_nova_async from
    its own loop cannot use connections tied to another loop."""
    sentinel = object()
    monkeypatch.setattr(brain, "claude", sentinel)

    async def which():
        return brain._claude()

    def after_a_turn():
        _capture(monkeypatch)
        brain.ask_nova("a")
        return asyncio.run(which())

    assert _on_fresh_thread(after_a_turn) is sentinel


def test_leftover_tasks_are_cancelled_like_asyncio_run(monkeypatch):
    """A turn that raises can leave a consumer waiting on its queue. It must not
    wake up inside the next turn."""
    left = {}

    async def failing_turn(user_text, **kwargs):
        left["task"] = asyncio.create_task(asyncio.Queue().get())
        raise RuntimeError("stream failed")

    monkeypatch.setattr(brain, "ask_nova_async", failing_turn)
    monkeypatch.setattr(brain, "effective_tier", lambda: "hokage")

    def turn():
        try:
            brain.ask_nova("a")
        except RuntimeError:
            return left["task"].cancelled()

    assert _on_fresh_thread(turn) is True


def test_the_thread_client_keeps_idle_connections_between_turns():
    """httpx drops an idle connection after 5s by default and turns are a median
    37s apart, so without this every turn reconnected anyway. It must also stay
    under Anthropic's own idle close, measured between 300 and 420s."""
    def client_limits():
        brain._thread_loop()
        pool = brain._thread_state.claude._client._transport._pool
        return pool._keepalive_expiry

    expiry = _on_fresh_thread(client_limits)
    assert expiry == brain._KEEPALIVE_S
    assert 30 <= brain._KEEPALIVE_S < 300
