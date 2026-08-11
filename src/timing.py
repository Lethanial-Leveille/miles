"""Per turn latency instrumentation.

The stages of a turn are spread across four modules and three execution
contexts: record_command and transcribe on the main thread, the Claude stream
in an asyncio loop, and speak() on an executor thread. Rather than thread a
context object through every signature, stages write into one module level
turn record. The voice loop handles exactly one turn at a time, so a single
record is sufficient; a lock guards it because the writers are on different
threads.

Two rules keep stray writes out:

  1. Writes are ignored entirely when no turn is active. tts.speak is also
     wired to timer and reminder alerts through actions.set_speak_fn, so an
     alert firing between turns would otherwise open a phantom record.
  2. First write wins per stage. An action turn makes two Claude calls and
     speaks several sentences; the number that matters for perceived latency
     is always the first one. The second Claude call is recorded separately
     as action_ms.

The clock is time.monotonic() throughout. Wall clock is not usable here since
it can step backward on an NTP correction mid turn.
"""

import threading
import time
from contextlib import contextmanager

from database import log_timing

_lock = threading.Lock()
_turn = None


def begin_turn(turn_type):
    """Open a new turn record, discarding any unfinished previous one."""
    global _turn
    with _lock:
        _turn = {
            "turn_type": turn_type,
            "speech_end": None,
            "action_fired": False,
            "model": None,
            "stages": {},
        }


def abandon_turn():
    """Drop the current turn without logging it.

    Used when a turn ends before producing audio (noise transcript, failed
    verification). Those turns have no perceived latency to report and would
    otherwise skew the distribution with partial rows."""
    global _turn
    with _lock:
        _turn = None


def note_speech_end(speech_end_monotonic):
    """Record when the user actually stopped talking.

    Everything the user experiences as waiting is measured from here, which is
    why this is captured inside the capture loop rather than when the loop
    returns. The gap between the two is the endpointing delay, invisible in
    any measurement that starts after recording ends."""
    with _lock:
        if _turn is None or speech_end_monotonic is None:
            return
        _turn["speech_end"] = speech_end_monotonic
        _turn["stages"].setdefault(
            "speech_end_to_endpoint_ms",
            (time.monotonic() - speech_end_monotonic) * 1000.0,
        )


def mark(stage, milliseconds):
    with _lock:
        if _turn is None:
            return
        _turn["stages"].setdefault(stage, milliseconds)


def note_model(model):
    """Record which model served this turn. Set once, at the first Claude
    call, so an action turn's second call cannot overwrite it — both calls in
    a turn use the same model."""
    with _lock:
        if _turn is None or _turn["model"] is not None:
            return
        _turn["model"] = model


def note_cache(read_tokens, creation_tokens):
    """Record prompt cache effectiveness for the turn's first Claude call.

    Zero reads across repeated turns means caching is not engaging, which the
    API never reports as an error: below the model's minimum cacheable prefix
    it simply does nothing."""
    with _lock:
        if _turn is None:
            return
        _turn["stages"].setdefault("cache_read_tokens", read_tokens)
        _turn["stages"].setdefault("cache_creation_tokens", creation_tokens)


def note_tool(tool_ms):
    """Time spent inside the tool itself, separate from the model call that
    speaks about the result.

    The old single action_ms bundled both, so a slow action turn could not be
    attributed. Weather is two HTTP round trips; a timer is a thread spawn and
    effectively free. Whether a bridge sentence is worth its verbosity depends
    on this number, so it is measured before that is decided rather than
    guessed at from which tools feel slow."""
    with _lock:
        if _turn is None:
            return
        _turn["action_fired"] = True
        _turn["stages"].setdefault("tool_ms", tool_ms)


def note_second_ttft(second_ttft_ms):
    """Time to first token on the follow up call that reads a tool result back.

    Only set for tools with returns_to_model. Fire and forget tools never make
    this call, and a null here is the signal that the turn took the cheap
    path."""
    with _lock:
        if _turn is None:
            return
        _turn["stages"].setdefault("second_ttft_ms", second_ttft_ms)


def note_action(action_ms):
    """Record the extra work an action turn does: dispatching the action and
    making the second Claude call. Flags the turn so action turns can be
    separated from plain ones during analysis, since they are structurally
    slower and would otherwise smear the distribution."""
    with _lock:
        if _turn is None:
            return
        _turn["action_fired"] = True
        _turn["stages"].setdefault("action_ms", action_ms)


@contextmanager
def stopwatch(stage):
    """Time a synchronous block and record it under `stage`."""
    started = time.monotonic()
    try:
        yield
    finally:
        mark(stage, (time.monotonic() - started) * 1000.0)


def note_tts(ttfb_ms, first_audio_ms):
    """Record the first speech synthesis of the turn and close out the
    perceived latency measurement.

    total_perceived_ms is taken here rather than at the end of the turn,
    because what the user perceives is the wait until Nova starts talking, not
    until she stops. Measured at the moment the first chunk is flushed to
    aplay, which is the closest software proxy available: ALSA adds its own
    buffer delay after that point, so the true figure at the speaker is
    slightly higher and consistently so."""
    with _lock:
        if _turn is None or "tts_ttfb_ms" in _turn["stages"]:
            return
        _turn["stages"]["tts_ttfb_ms"] = ttfb_ms
        _turn["stages"]["tts_first_audio_ms"] = first_audio_ms
        if _turn["speech_end"] is not None:
            _turn["stages"]["total_perceived_ms"] = (
                time.monotonic() - _turn["speech_end"]
            ) * 1000.0


def end_turn(transcript=None, response=None):
    """Write the turn to the timing log and close it out."""
    global _turn
    with _lock:
        if _turn is None:
            return
        turn, _turn = _turn, None

    stages = turn["stages"]
    log_timing(
        turn_type=turn["turn_type"],
        action_fired=turn["action_fired"],
        transcript=transcript,
        speech_end_to_endpoint_ms=stages.get("speech_end_to_endpoint_ms"),
        transcribe_ms=stages.get("transcribe_ms"),
        verify_ms=stages.get("verify_ms"),
        claude_ttft_ms=stages.get("claude_ttft_ms"),
        claude_total_ms=stages.get("claude_total_ms"),
        tts_ttfb_ms=stages.get("tts_ttfb_ms"),
        tts_first_audio_ms=stages.get("tts_first_audio_ms"),
        action_ms=stages.get("action_ms"),
        total_perceived_ms=stages.get("total_perceived_ms"),
        model=turn["model"],
        response=response,
        cache_read_tokens=stages.get("cache_read_tokens"),
        cache_creation_tokens=stages.get("cache_creation_tokens"),
        first_sentence_ms=stages.get("first_sentence_ms"),
        max_pause_ms=stages.get("max_pause_ms"),
        tool_ms=stages.get("tool_ms"),
        second_ttft_ms=stages.get("second_ttft_ms"),
    )

    total = stages.get("total_perceived_ms")
    if total is not None:
        print(f"  [{turn['model'] or 'model n/a'}] "
              f"perceived latency {total:.0f}ms "
              f"(endpoint {stages.get('speech_end_to_endpoint_ms', 0):.0f} + "
              f"stt {stages.get('transcribe_ms', 0):.0f} + "
              f"verify {stages.get('verify_ms', 0):.0f} + "
              f"llm {stages.get('claude_ttft_ms', 0):.0f} + "
              f"tts {stages.get('tts_first_audio_ms', 0):.0f})", flush=True)
