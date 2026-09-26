"""A turn's first Claude call, opened before the endpoint and held for the real turn.

The endpoint waits out 1.2s of silence to be sure he has finished, and the
Deepgram transcript exists about 0.5 to 0.7s into that wait. A speculative turn
spends the rest of it on the Claude call the real turn is about to make. At the
endpoint the real turn claims it: if what the request was built from is
unchanged, it reads the buffered events and carries on live, and if anything
differs, or he spoke again, the speculation is cancelled and the turn makes its
own call exactly as before.

**It only buffers.** No message is saved, no alert consumed, no confirmation
clock advanced, no tool run, no timing marked, nothing played or shown. Every one
of those belongs to the real turn, after the claim, so a discarded speculation
leaves nothing behind but the request itself. Cancelling closes the HTTP stream,
so Anthropic stops generating.

**One worker thread, one event loop, run forever.** The same reason as the loop
per thread in brain.py: a pooled connection belongs to the loop that opened it,
and a loop that lives on keeps its connection warm between speculations.

**Started only when the real turn would reach Claude with this exact text.**
consider() follows voice_main step by step: the wake phrase split and the noise
check in the order each path applies them, local intent, and verification of the
early embedding by the same rule verify_voice uses. A voice that does not verify
as his gets no speculation, so his personal prompt is never sent for it.
"""

import asyncio
import threading
import time

import alerts
import brain
import local_intent
import tts
import verification
from config import MODEL_A, MODEL_AB_TEST, SPECULATIVE_CLAUDE
from database import get_recent_messages
from parsing import is_noise_transcript, split_after_wake_phrase

_worker_lock = threading.Lock()
_loop = None
_client = None


def _worker_loop():
    """Start the worker on first use and return its loop."""
    global _loop
    with _worker_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run():
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            threading.Thread(target=run, name="speculative_turn", daemon=True).start()
            ready.wait()
            _loop = loop
    return _loop


def _claude():
    """The worker's own client, made on the worker's loop the first time."""
    global _client
    if _client is None:
        _client = brain._new_client()
    return _client


class Cancelled(Exception):
    """The speculation was dropped before it finished."""


class _Buffer:
    """Events from the worker, read by the real turn on another thread and loop.

    The reader registers an asyncio.Event on its own loop; the writer sets it with
    call_soon_threadsafe after appending, so a reader waiting for the next event
    wakes on its own loop rather than polling."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events = []
        self.final = None
        self.error = None
        self.done = False
        self._waiter = None   # (loop, asyncio.Event)

    def _wake(self):
        waiter = self._waiter
        if waiter is not None:
            loop, event = waiter
            loop.call_soon_threadsafe(event.set)

    def push(self, event):
        with self._lock:
            self.events.append(event)
        self._wake()

    def finish(self, final=None, error=None):
        with self._lock:
            if self.done:
                return
            self.final, self.error, self.done = final, error, True
        self._wake()

    def attach(self):
        event = asyncio.Event()
        with self._lock:
            self._waiter = (asyncio.get_running_loop(), event)
        return event

    def read(self, index):
        """(event or None, finished) at this position, cleared for waiting."""
        with self._lock:
            if index < len(self.events):
                return self.events[index], False
            return None, self.done


class _Replay:
    """Stands in for the SDK's stream: async with, async for, get_final_message.

    Yields what was buffered, then waits for the rest as it arrives."""

    def __init__(self, turn):
        self._turn = turn
        self._buffer = turn._buffer
        self._index = 0
        self._ready = None

    async def __aenter__(self):
        self._ready = self._buffer.attach()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # The real turn failed partway: nothing more of this is wanted.
        if exc_type is not None:
            self._turn.cancel()
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            self._ready.clear()
            event, finished = self._buffer.read(self._index)
            if event is not None:
                self._index += 1
                return event
            if finished:
                if self._buffer.error is not None:
                    raise self._buffer.error
                raise StopAsyncIteration
            await self._ready.wait()

    async def get_final_message(self):
        while True:
            self._ready.clear()
            if self._buffer.done:
                if self._buffer.error is not None:
                    raise self._buffer.error
                return self._buffer.final
            await self._ready.wait()


class SpeculativeTurn:
    """Build a request and open its Claude stream on the worker, buffering events.

    make_request runs on the worker, off the capture loop, and must build the
    request with no side effects: brain.build_request with log_recall=False and
    peeked alerts. It is a callable rather than a request so that recall, about
    87ms, is spent here and not on the thread reading the microphone."""

    def __init__(self, make_request):
        self._buffer = _Buffer()
        self._cancelled = threading.Event()
        self.fingerprint = None
        self.request = None      # the TurnRequest, for the turn that claims it
        self.started_at = time.monotonic()
        self.opened_at = None    # when the Claude request went out
        self._future = asyncio.run_coroutine_threadsafe(
            self._run(make_request), _worker_loop())

    async def _run(self, make_request):
        try:
            request = await asyncio.to_thread(make_request)
            if self._cancelled.is_set():
                raise Cancelled()
            self.request = request
            self.fingerprint = request.fingerprint
            # The first sentence will want ElevenLabs about 1.5s from now.
            tts.warm()
            self.opened_at = time.monotonic()
            async with _claude().messages.stream(**brain.first_call_kwargs(request)) as stream:
                async for event in stream:
                    self._buffer.push(event)
                final = await stream.get_final_message()
            self._buffer.finish(final=final)
        except (asyncio.CancelledError, Cancelled):
            self._buffer.finish(error=Cancelled())
        except Exception as exc:
            self._buffer.finish(error=exc)

    def cancel(self):
        """Drop it. Safe from any thread, any number of times."""
        self._cancelled.set()
        self._future.cancel()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def claim(self, fingerprint):
        """The stream for the real turn, or None when this cannot stand in for it.

        None, and the speculation cancelled, when it was dropped, when its request
        is not built yet, when anything it was built from has changed, or when it
        already failed. The real turn then makes its own call, as it always did."""
        usable = (not self.cancelled
                  and self.fingerprint is not None
                  and self.fingerprint == fingerprint
                  and not (self._buffer.done and self._buffer.error is not None))
        if not usable:
            self.cancel()
            return None
        return _Replay(self)


# ── when one is allowed to start, and handing it over ──

_current_lock = threading.Lock()
_current = None

# Whether the conversation in progress is his. A follow up is trusted on the
# session only when it is: a guest's conversation stays a guest's, and a
# speculation must never send his personal prompt for a guest's words.
_session_is_him = False


def set_session(is_him):
    """Called by voice_main each time it decides whose conversation this is."""
    global _session_is_him
    _session_is_him = bool(is_him)


def user_text_for(text, turn_type):
    """(the text the real turn would send, session_trusted, named), or None when
    the turn would not reach Claude with it.

    The order matters and differs by path, exactly as in voice_main: a first
    turn splits off the wake phrase and then checks for noise; a follow up
    checks for noise first, and a wake phrase in it makes a fresh turn that is
    scored rather than trusted on the session. named follows voice_main too: a
    first turn always, a follow up only when it held the wake phrase."""
    if text is None:
        return None
    if turn_type == "initial":
        said, rest = split_after_wake_phrase(text)
        if said and not rest:
            return None
        text = rest if said else text
        return None if is_noise_transcript(text) else (text, False, True)
    if is_noise_transcript(text):
        return None
    said, rest = split_after_wake_phrase(text)
    if said:
        return (rest, False, True) if rest else None
    return text, True, False


def consider(text, turn_type, embedded_seconds, embedding, voiceprint):
    """Start a speculative turn for this transcript if the real one would call
    Claude with it, and return it; otherwise None. Called on the Deepgram
    thread the moment a speculative transcript arrives."""
    global _current
    # A/B alternation advances a counter when the model is chosen, so there is
    # no way to know the real turn's model in advance. It is off in production.
    if not SPECULATIVE_CLAUDE or MODEL_AB_TEST:
        return None
    # Before the intent examples are encoded, classify would spend seconds.
    if not local_intent.is_warm():
        return None
    derived = user_text_for(text, turn_type)
    if derived is None:
        return None
    user_text, session_trusted, named = derived
    session_trusted = session_trusted and _session_is_him
    if local_intent.classify(user_text) is not None:
        return None
    verdict = verification.decide(embedded_seconds, embedding, voiceprint, session_trusted)
    if verdict.outcome != verification.VERIFIED:
        return None

    def make_request():
        # The real turn saves his message and then reads the last twenty rows,
        # so this is the last nineteen plus his message, never saved here.
        history = get_recent_messages(19) + [{"role": "user", "content": user_text}]
        return brain.build_request(user_text, "voice", brain.effective_tier(), MODEL_A,
                                   history, alerts.peek_for_fold(), log_recall=False,
                                   named=named)

    turn = SpeculativeTurn(make_request)
    with _current_lock:
        previous, _current = _current, turn
    if previous is not None:
        previous.cancel()
    print(f"(speculating on {user_text!r})", flush=True)
    return turn


def take():
    """The pending speculation, handed to the real turn, or None."""
    global _current
    with _current_lock:
        turn, _current = _current, None
    return turn


def discard():
    """Cancel anything pending. Called whenever a new recording starts."""
    turn = take()
    if turn is not None:
        turn.cancel()
