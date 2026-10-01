import asyncio
import hashlib
import json
import re
import threading
import time
from datetime import datetime
from typing import NamedTuple

import anthropic
import httpx
import timing
from tts import join_for_speech, play, speak, start_synthesis
from prompts import build_enhanced_prompt, TEXT_TURN_NOTE
from database import (save_message, delete_message, last_reply_at, get_seed_memories, get_episodic_memories,
                      get_recent_messages, get_messages_after, search_memories, memory_manifest,
                      get_shareable_memories, effective_tier)
from parsing import extract_memories, strip_leading_bracket_cue
from stream_router import StreamRouter
from tools import registry, Permission, permits
from database import log_tool_call
import alerts
import memory_pass
import phrasebank
import spoken_archive

# Imported for its side effect: registering the tools. Without it the registry
# is empty, the capability block is blank, and Nova silently has no
# capabilities. Kept explicit, with this comment, because an unused looking
# import is exactly what a later cleanup deletes.
import actions        # noqa: F401
import memory_tool    # noqa: F401
import system_state   # noqa: F401
import tier_tool      # noqa: F401
import calendar_tools # noqa: F401
import oura_tools     # noqa: F401
import hevy_tools     # noqa: F401
import prowl_tools    # noqa: F401
import pending_action # noqa: F401
import web_search     # noqa: F401

from config import (MODEL_AB_TEST, MODEL_A, MODEL_B, PROMPT_CACHING,
                    HISTORY_ASSISTANT_WORDS, RECALL_LIMIT, BARGE_IN,
                    BRIDGE_DELAY_MS, HISTORY_GAP_NOTE_MINUTES)

claude = anthropic.AsyncAnthropic()

# One event loop per thread, kept for the life of the process, with its own
# client. ask_nova used to call asyncio.run, which builds a loop and throws it
# away, while the module client kept its pooled connection from the loop before.
# Reusing it raised "Event loop is closed", and the SDK retried after a backoff
# of 0.4 to 0.5s, on every turn and silently, since nothing configures logging.
# Measured Sep 25 2026 with count_tokens: about 590ms per call that way against
# 75ms on a kept loop; first turns after a restart, which had no stale
# connection, ran 300 to 390ms faster to first token. Per thread rather than one
# shared loop because the server runs turns on worker threads concurrently with
# the voice loop, and a turn's synchronous work would stall the other's stream.
_thread_state = threading.local()

# How long an idle connection is kept for the next turn. httpx drops it after
# 5s by default, and turns are a median 37s apart, so every turn reconnected
# (about 140ms against 75ms warm). Anthropic kept one open through 300s idle and
# had closed it by 420s, measured Sep 25 2026; this stays under that with
# margin, because a reuse that races the server's own close costs a retry.
_KEEPALIVE_S = 240


def _new_client():
    """A client for one long lived loop, keeping idle connections _KEEPALIVE_S."""
    return anthropic.AsyncAnthropic(
        http_client=anthropic.DefaultAsyncHttpxClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20,
                                keepalive_expiry=_KEEPALIVE_S)))


def _thread_loop():
    if getattr(_thread_state, "loop", None) is None:
        _thread_state.loop   = asyncio.new_event_loop()
        _thread_state.claude = _new_client()
    return _thread_state.loop


def _claude():
    """The persistent client when running on this thread's own loop, else the
    module one. The module client is what the tests replace, and what anyone
    driving ask_nova_async from a loop of their own must get, because this
    thread's client holds connections that only its own loop can use."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None and running is getattr(_thread_state, "loop", None):
        return _thread_state.claude
    return claude


def _cancel_leftovers(loop):
    """What asyncio.run does on the way out, without closing the loop.

    A turn that raised can leave a consumer task waiting on its queue forever.
    asyncio.run used to cancel those; a loop that outlives the turn has to do
    it itself, or they would wake inside the next turn."""
    leftovers = [t for t in asyncio.all_tasks(loop) if not t.done()]
    for task in leftovers:
        task.cancel()
    if leftovers:
        loop.run_until_complete(asyncio.gather(*leftovers, return_exceptions=True))


class TurnResult(NamedTuple):
    """What one turn produced.

    `dismissed` is set when Nova recognized an intent to end the conversation
    and emitted [ACTION: dismiss]. Returning it here rather than stashing it in
    module state keeps the signal explicit at the call site, which matters
    because two different callers drive this: the voice loop, which acts on it,
    and the HTTP server, which does not."""
    text: str
    dismissed: bool = False
    # Set when Nova judged the speech was not addressed to her at all. Distinct
    # from dismissed: dismissed ends a conversation he ended, this ends one he
    # was never having with her, and the difference is whether she says anything.
    ignored: bool = False

# Turn counter for the A/B alternation. Resets on service restart, which is
# harmless: alternation stays balanced from whatever point it resumes.
_turn_index = 0


def _select_model():
    """Pick the model for this turn, alternating strictly when the A/B is on.

    Chosen once per turn and reused for an action turn's second call, so a
    single turn is never split across two models."""
    global _turn_index
    if not MODEL_AB_TEST:
        return MODEL_A
    model = MODEL_A if _turn_index % 2 == 0 else MODEL_B
    _turn_index += 1
    return model


# The sentence playing and the one after it. Enough that the next sentence is
# ready the moment the current one ends; few enough that a long reply never
# opens more ElevenLabs requests at once than the smallest plan allows. His API
# key cannot read the plan's real limit, so this assumes the floor.
_SYNTHESIS_AHEAD = 2


def _play_with_barge_in(synthesis):
    """Play one synthesized sentence, listening for him while it plays: the
    wake word, or any speech loud enough (BARGE_IN_SPEECH).

    Returns (interrupted, waited_ms).

    The watcher owns the microphone for exactly the span aplay is draining,
    which is the only window where the main loop is not reading it. Two readers
    on one stream do not raise, they interleave frames, so the exclusivity here
    is load bearing rather than tidy.

    interrupted is read before the watcher is released, not after. The watcher
    signals a heard wake word by setting the same event this function sets to
    stop it, so reading it afterwards returned True for every sentence: the day
    barge in was switched on, every reply would have stopped after its first
    sentence.

    Falls back to plain playback when barge in is off, which is the default."""
    if not BARGE_IN:
        return False, play(synthesis)

    import audio
    interrupt = threading.Event()
    watcher = threading.Thread(target=audio.watch_for_interrupt,
                               args=(interrupt,), daemon=True)
    watcher.start()
    try:
        waited_ms = play(synthesis, interrupt=interrupt)
        interrupted = interrupt.is_set()
    finally:
        interrupt.set()          # release the watcher whether or not it fired
        watcher.join(timeout=1.0)
    return interrupted, waited_ms


async def _tts_consumer(queue: asyncio.Queue, text_parts: list, leaks_seen: set) -> None:
    """Speak a reply: the first sentence at once, the rest as one piece.

    Two workers. The feeder starts the first sentence the moment Claude writes
    it, so first audio is as early as it can be. Every later sentence is held
    until Claude finishes, then all of them are synthesized together, joined by
    join_for_speech. The player plays the pieces in order.

    Together, because on eleven_v3 each request is voiced on its own, and
    sentences voiced separately came out as a string of slightly different
    deliveries, heard as "different cadences, like a different person". One
    request for the rest is one delivery, and it lets an ellipsis give a pause
    between sentences, which a separate request per sentence cannot.

    The rest normally finishes streaming while the first sentence is still
    playing. When the first sentence is very short it may not, and the wait is
    logged."""
    loop = asyncio.get_running_loop()
    ready = asyncio.Queue()
    slots = asyncio.Semaphore(_SYNTHESIS_AHEAD)

    async def start(sentences):
        await slots.acquire()
        # start_synthesis reads the pronunciation table, so it runs off the
        # event loop like everything else that touches the disk.
        synthesis = await loop.run_in_executor(
            None, start_synthesis, join_for_speech(sentences))
        await ready.put((sentences, synthesis))

    async def feed():
        try:
            rest = []
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break
                sentence = strip_leading_bracket_cue(sentence, leaks_seen)
                if played_first_request[0]:
                    rest.append(sentence)
                    continue
                played_first_request[0] = True
                await start([sentence])
            if rest:
                await start(rest)
        except Exception as exc:
            print(f"Could not start speech synthesis: {exc}", flush=True)
        finally:
            # Always, including on failure or cancellation. Without it a failure
            # in here would leave the player waiting forever, and the turn would
            # hang with nothing in the log.
            ready.put_nowait(None)

    played_first_request = [False]
    feeder = asyncio.create_task(feed())
    played = 0
    spoken = []   # (text sent, audio, silence before it), for spoken_archive
    try:
        while True:
            item = await ready.get()
            if item is None:
                break
            sentences, synthesis = item
            text_parts.extend(sentences)
            try:
                interrupted, waited_ms = await loop.run_in_executor(
                    None, _play_with_barge_in, synthesis)
            finally:
                slots.release()
            played += 1
            audio = getattr(synthesis, "audio", None)
            if audio:
                spoken.append((synthesis.text, bytes(audio), waited_ms))
            # For the rest of the reply, the silence heard before it. The first
            # sentence's wait is already logged as tts_ttfb_ms.
            if played > 1 and waited_ms is not None:
                print(f"  rest of reply ({len(sentences)} sentences) waited "
                      f"{waited_ms:.0f}ms for audio", flush=True)
            if interrupted:
                # He said the wake word, or spoke, over her. Drop the rest rather than
                # finishing the thought he interrupted, which is the entire
                # point of being able to interrupt.
                break
    finally:
        feeder.cancel()
        if spoken:
            # A thread, not the executor: this also runs when the turn is
            # cancelled, and nothing about saving a copy should wait on that.
            threading.Thread(target=spoken_archive.save_reply,
                             args=(spoken,), daemon=True).start()


async def _collect_text(queue: asyncio.Queue, text_parts: list, leaks_seen: set) -> None:
    """Stand in for _tts_consumer on a text turn: gather the sentences, play nothing.

    Same arguments and the same cue stripping, so everything that reads
    text_parts afterwards, the fire and forget branch included, behaves the same
    on both channels. Before this, a typed message was spoken through the room
    speaker by miles-server, and /chat did not return until playback ended, so
    the app showed Nova thinking while she was already talking."""
    while True:
        sentence = await queue.get()
        if sentence is None:
            return
        text_parts.append(strip_leading_bracket_cue(sentence, leaks_seen))


def _plays_in_the_room(channel, device):
    """Whether this turn should come out of the speaker in his room.

    Two conditions, and they answer different questions. channel is the same
    test build_enhanced_prompt makes, so the prompt and the speaker can never
    disagree about whether a reply was written to be spoken. device is who
    asked, and only the Pi is in the room: a turn from the phone has no business
    playing out loud there, however it was formatted.

    Splitting them is what lets the app ask for spoken formatting and then
    synthesize the reply itself through /speak. Until Sep 23 2026 channel
    decided both, so an app turn asking to be spoken was spoken here instead,
    and the app's only silent option was text formatted with markdown."""
    return channel != "text" and device == "pi"


def _consumer_for(channel, device):
    return _tts_consumer if _plays_in_the_room(channel, device) else _collect_text


async def _say(text, channel, device):
    """Speak a line code wrote rather than the model: a staged question, or the
    fallback. When it is not played it still reaches the reply."""
    if not _plays_in_the_room(channel, device):
        return
    await asyncio.get_running_loop().run_in_executor(None, speak, text)


# What the app shows while a tool works, in place of a flat "thinking". Keyed
# by tool name, and a test requires an entry for every registered tool, so a new
# tool cannot silently show nothing. None means no stage: dismiss and ignore are
# state changes, not work.
_STAGE_WORDS = {
    "accept_discovered_job":   "Finding that job",
    "cancel_reminder":         "Cancelling a reminder",
    "check_calendar_freebusy": "Checking when you're free",
    "confirm_pending_action":  "Making the change",
    "create_calendar_event":   "Adding it to your calendar",
    "delete_calendar_event":   "Deleting the event",
    "dismiss":                 None,
    "find_schedule_conflicts": "Looking for conflicts",
    "get_job_application":     "Opening that application",
    "get_job_search_summary":  "Checking your job search",
    "get_lift_history":        "Looking at that lift",
    "get_oura_activity":       "Reading your activity",
    "get_oura_heartrate":      "Reading your heart rate",
    "get_oura_readiness":      "Reading your readiness",
    "get_oura_sleep":          "Reading your sleep",
    "get_recent_workouts":     "Checking your workouts",
    "get_routines":            "Reading your routines",
    "get_session_plan":        "Planning your session",
    "get_system_state":        "Checking on the Pi",
    "get_upcoming_events":     "Checking your calendar",
    "get_weather":             "Checking the weather",
    "ignore":                  None,
    "list_discovered_jobs":    "Checking what Prowl found",
    "list_job_applications":   "Looking through your applications",
    "list_pending_memories":   "Looking at what's waiting for review",
    "lower_access":            "Changing access",
    "plan_sessions":           "Planning around your week",
    "remember":                "Remembering that",
    "rename_calendar_events":  "Finding the events",
    "review_pending_memory":   "Updating your memories",
    "set_reminder":            "Setting a reminder",
    "set_timer":               "Setting a timer",
    "undo_last_change":        "Undoing that",
    "update_calendar_event":   "Changing the event",
}


# Tools Anthropic runs (web_search.py). Apart from _STAGE_WORDS because that
# map must match the local registry exactly.
_SERVER_STAGE_WORDS = {
    "web_search": "Searching the web",
}


# The pre rendered line said when one of these tools starts on a voice turn.
# Not every tool: remember and the control tools finish at once, and a timer
# answers itself. A test requires every entry to be a registered tool and every
# key to be in the phrase bank.
_BRIDGES = {
    "get_upcoming_events":     "bridge_calendar",
    "check_calendar_freebusy": "bridge_calendar",
    "find_schedule_conflicts": "bridge_calendar",
    "plan_sessions":           "bridge_calendar",
    "create_calendar_event":   "bridge_calendar",
    "update_calendar_event":   "bridge_calendar",
    "delete_calendar_event":   "bridge_calendar",
    "rename_calendar_events":  "bridge_calendar",
    "get_oura_sleep":          "bridge_health",
    "get_oura_readiness":      "bridge_health",
    "get_oura_activity":       "bridge_health",
    "get_oura_heartrate":      "bridge_health",
    "get_weather":             "bridge_weather",
    "get_job_search_summary":  "bridge_jobs",
    "list_job_applications":   "bridge_jobs",
    "get_job_application":     "bridge_jobs",
    "list_discovered_jobs":    "bridge_jobs",
    "accept_discovered_job":   "bridge_jobs",
}


async def _bridge_if_slow(tool_future, key):
    """Play the bridge only if the tool is still running after BRIDGE_DELAY_MS.

    It used to start when the call began streaming, before the tool had run at
    all, so a calendar addition that takes 5ms still said "Let me look at your
    calendar." for a second and a half, and the read back waited behind it. A
    fast tool now answers in silence and only a slow one is covered. Returns the
    playback future, or None when nothing played. It plays under speak_lock, so
    the answer waits for it to finish."""
    if key is None:
        return None
    done, _ = await asyncio.wait({tool_future}, timeout=BRIDGE_DELAY_MS / 1000.0)
    if done:
        return None
    return asyncio.get_running_loop().run_in_executor(
        None, lambda: phrasebank.play(key, bridge=True))


def _bridge_for(event):
    """The bridge key for a tool call starting, or None."""
    if event.type != "content_block_start" or event.content_block.type != "tool_use":
        return None
    return _BRIDGES.get(event.content_block.name)


def _stage_for(event):
    """The words for a tool call starting, or None for any other stream event.

    Read from content_block_start, which arrives before the tool's arguments
    are written, so the app can say what is happening as early as possible."""
    if event.type != "content_block_start":
        return None
    block = event.content_block
    if block.type == "tool_use":
        return _STAGE_WORDS.get(block.name)
    if block.type == "server_tool_use":
        return _SERVER_STAGE_WORDS.get(block.name)
    return None


def _block_gap(event, so_far):
    """A space to feed when a new text block starts, or "".

    A search splits the reply into text blocks: a lead in, the search, then the
    answer. The deltas carry no space across that boundary, so "Let me check."
    ran into "The Gators" and the sentence splitter, which needs whitespace
    after the full stop, held both as one sentence."""
    if event.type != "content_block_start" or event.content_block.type != "text":
        return ""
    if not so_far or so_far[-1].isspace():
        return ""
    return " "


def _emit(on_text, kind, text):
    """Hand one piece of the reply to a caller streaming it, if there is one.

    kind is "delta" for text as Nova writes it, "stage" for what she is doing
    while a tool works, or "reset" to say that what was streamed so far is being
    discarded: the bridge sentence said alongside a tool call is not part of the
    answer that replaces it.

    A failure in the callback is logged and swallowed. The speaker, the tools
    and the database do not care that an HTTP client went away, and a turn that
    already ran must still finish."""
    if on_text is None or (kind != "reset" and not text):
        return
    try:
        on_text(kind, text)
    except Exception as exc:
        print(f"Streaming callback failed: {exc}", flush=True)


def _trim_history(messages: list) -> list:
    """Shorten past assistant turns before sending them as context.

    Nova's own transcript is a stronger length signal than any instruction in
    the system prompt, so full history teaches her to keep answering at the
    length she last answered at. Trimming keeps the substance of what was
    discussed and drops the demonstration of how long to take saying it.

    User turns are left intact: they are the actual context, and they are
    short anyway."""
    trimmed = []
    for message in messages:
        if message["role"] != "assistant":
            trimmed.append(message)
            continue
        words = message["content"].split()
        if len(words) <= HISTORY_ASSISTANT_WORDS:
            trimmed.append(message)
            continue
        trimmed.append({
            **message,
            "content": " ".join(words[:HISTORY_ASSISTANT_WORDS]) + "...",
        })
    return trimmed


def _resident_ids():
    """Ids of the memories already in his system prompt: tier one seeds and the
    newest learned ones. Recall looks for everything else."""
    return ({row[0] for row in get_seed_memories()}
            | {row[0] for row in get_episodic_memories()})


def _with_recalled(messages: list, query: str, log: bool = True) -> list:
    """Attach any memories the transcript matches, and the prompt lacks, to the
    final user turn.

    Same placement argument as the clock below. This changes every turn, so it
    has to sit after the cache breakpoint; in the system prompt it would
    invalidate the whole prefix on every single turn and quietly cost more than
    it saves.

    Only what the prompt does not already hold, because attaching a fact twice
    is how a model starts treating it as more important than it is. Until Sep
    28 2026 that was written as "tier two only", on the belief that tier one is
    all in the prompt. It is not: a learned memory is tier one, and only the
    newest twenty of those are resident. The twenty first would have been in
    neither place, with no way for Nova to reach it at all.

    A miss returns the messages untouched. Retrieval finding nothing is the
    normal case for most turns, and it leaves Nova exactly where she was before
    any of this existed, answering from the resident tier. The manifest block
    is what tells her to say so rather than invent something."""
    if not messages or messages[-1]["role"] != "user":
        return messages

    hits = search_memories(query, limit=RECALL_LIMIT, exclude=_resident_ids(), log=log)
    if not hits:
        return messages

    lines = "\n".join(f"- {content}" for _, content, _ in hits)
    return messages[:-1] + [{
        **messages[-1],
        "content": f"{messages[-1]['content']}\n\n[Also relevant, from what you know about him:\n{lines}]",
    }]


NAMED_TURN_NOTE = "[He said your name to start this.]"


def _with_named_note(messages: list, named: bool) -> list:
    """On a turn he opened with the wake word, say so in the final user turn.

    The wake phrase is stripped before the transcript reaches Claude, so a
    question asked by name looked exactly like room chatter caught in the follow
    up window. On Sep 26 2026, with friends talking, "Is Trump a good president?"
    and a question about a Porsche around the Nurburgring, both asked after
    "hey nova", were ignored as not addressed to her. Same placement as the
    clock, after the cache breakpoint; the database keeps the raw message."""
    if not named or not messages or messages[-1]["role"] != "user":
        return messages
    return messages[:-1] + [{
        **messages[-1],
        "content": f"{messages[-1]['content']}\n\n{NAMED_TURN_NOTE}",
    }]


def _gap_words(minutes):
    if minutes < 90:
        return f"{round(minutes)} minutes"
    hours = minutes / 60
    if hours < 36:
        return f"{round(hours)} hours"
    return f"{round(hours / 24)} days"


def _with_gap_note(messages: list, last_reply, now=None) -> list:
    """Say how long ago the conversation before this message ended, when it was
    long enough that what was said then about his day can be stale. Only the
    final user turn is touched, after the cache breakpoint like the clock."""
    if last_reply is None or not messages or messages[-1]["role"] != "user":
        return messages
    minutes = ((now or datetime.now()) - last_reply).total_seconds() / 60
    if minutes < HISTORY_GAP_NOTE_MINUTES:
        return messages
    note = (f"[The conversation before this message ended {_gap_words(minutes)} ago. "
            "Anything said then about his day, his workouts, his schedule or his "
            "health may no longer be true, so check again before answering from it.]")
    return messages[:-1] + [{**messages[-1], "content": f"{messages[-1]['content']}\n\n{note}"}]


def _with_current_time(messages: list) -> list:
    """Append the current date and time to the final user turn.

    Deliberately not in the system prompt. That is the cached prefix, and a
    value that changes every turn is the textbook silent cache invalidator:
    caching would report success while never producing a hit, quietly costing
    the ~700ms it currently saves. Appending to the last message puts it after
    the breakpoint, where it costs nothing.

    Without this Nova had no clock at all, and rather than saying so she
    copied the date out of the reminder example in her own prompt. Every
    reminder she set was due 2026-04-11, months in the past, so none of them
    could ever fire.

    The database keeps the raw utterance; this only shapes the API call."""
    if not messages or messages[-1]["role"] != "user":
        return messages

    now = datetime.now()
    stamp = f"{now:%A, %B} {now.day}, {now:%Y at %I:%M %p}".replace(" 0", " ")
    return messages[:-1] + [{
        **messages[-1],
        "content": f"{messages[-1]['content']}\n\n[Current date and time: {stamp}]",
    }]


def _with_text_note(messages: list, channel: str) -> list:
    """On a text turn, say in the final user turn that this reply is read.

    The history is mostly spoken replies with every number written out, and
    Nova's own transcript is a stronger signal than any system prompt rule, so
    NUMBER_FORMAT_TEXT alone lost to it. Same placement as the clock, after the
    cache breakpoint. Voice turns are untouched, and the database keeps the raw
    message."""
    if channel != "text" or not messages or messages[-1]["role"] != "user":
        return messages
    return messages[:-1] + [{
        **messages[-1],
        "content": f"{messages[-1]['content']}\n\n{TEXT_TURN_NOTE}",
    }]


# How many times the model may go back for another tool before it has to
# answer. Two rounds covers the real case (call a tool, read it back) plus one
# genuine chained call; the final round is made without tools so there is
# nothing left to reach for.
MAX_TOOL_ROUNDS = 3

# Phrases that tell him something was kept. Checked against the tools actually
# called; see _claims_a_save_without_calling.
_CLAIMS_A_SAVE = re.compile(
    r"\b(i've got that down|got that down|i've noted|noted that|i'll remember|"
    r"i've saved|saved that|i've stored|stored that|made a note|"
    r"i'll keep that in mind)\b",
    re.IGNORECASE)


_ASKS_A_CHANGE = re.compile(r"\b(add|move|rename|delete|change|reschedule)\b[^?]*\?", re.IGNORECASE)


def _asks_an_unstaged_change(text, staged):
    """Whether Nova asked to change the calendar herself, with nothing staged.

    Measurement, not enforcement. On Sep 14 2026 she asked "Rename Charley lesson
    on Wednesday at 4 PM to Charley lesson?" with no proposal behind it; he said
    yes, and nothing could run."""
    return staged is None and bool(_ASKS_A_CHANGE.search(text or ""))


def _claims_a_save_without_calling(text, called_tools):
    """Whether Nova told him she kept something without calling remember.

    Measurement, not enforcement. On Sep 13 2026 she said "I've got that down"
    and stored nothing, which tells him a fact is kept that is not. The prompt
    forbids it now; this makes every time it still happens visible."""
    return "remember" not in called_tools and bool(_CLAIMS_A_SAVE.search(text or ""))


def _tool_result_blocks(results):
    """Tool results shaped for the API, one block per call.

    is_error is only set when it is true. The API treats its presence as the
    flag, so sending is_error False on a success would mark every result as a
    failure."""
    return [
        {
            "type": "tool_result",
            "tool_use_id": r["block"].id,
            "content": str(r["output"]),
            **({"is_error": True} if r["is_error"] else {}),
        }
        for r in results
    ]


def _needs_second_call(results):
    """Whether Nova has to speak about what the tools returned.

    A tool declares returns_to_model at registration, so setting a timer does
    not cost a second round trip to rephrase "Timer set." A failure always
    does, whatever the tool declared. The fire and forget branch speaks "Done."
    when Nova said nothing alongside the call, and on Sep 13 2026 it said "Done."
    over three cancel calls that had each found nothing to cancel."""
    return any(r["is_error"] or registry.get(r["block"].name).returns_to_model
               for r in results)


def _with_alerts(messages: list, pending: list) -> list:
    """Attach fired timers and reminders to the final user turn.

    Same placement as the clock, and for the same reason: this sits after the
    cache breakpoint, so a value that changes every turn costs nothing. Putting
    it in the system prompt would invalidate the prefix on every alert.

    The alert arrives as context rather than as a finished sentence so Nova
    delivers it in her own words alongside whatever she was already answering,
    which is what a person interrupted by a kitchen timer would do."""
    if not pending or not messages or messages[-1]["role"] != "user":
        return messages

    lines = "\n".join(f"[Alert: {a.summary}]" for a in pending)
    return messages[:-1] + [{
        **messages[-1],
        "content": f"{messages[-1]['content']}\n\n{lines}",
    }]


class TurnRequest(NamedTuple):
    """Everything a turn sends to Claude, and a fingerprint of what it was made from."""
    model: str
    system: object
    messages: list
    fingerprint: str


def _fingerprint(user_text, channel, tier, model, prompt, history, pending_alerts,
                 named=False):
    """What a request was built from, as one comparable string.

    A speculative turn builds its request before the endpoint, and the real turn
    may only use it if nothing it was made from has changed since. Inputs
    rather than the finished request, so checking costs a prompt build and a
    history read (about 3ms) instead of recall again (about 87ms). A new memory
    changes the prompt through the manifest, so it changes this too.

    The clock is left out on purpose. It is stamped to the minute and a
    speculation runs less than a second early."""
    material = json.dumps({
        "user_text": user_text, "channel": channel, "tier": tier, "model": model,
        "prompt": prompt, "history": history,
        "alerts": [a.summary for a in pending_alerts], "named": named,
    }, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def _history_for(tier, conversation_start, user_text):
    """What a turn may see of what was said before it.

    His turns get his recent history, as always. Anyone else gets only the
    conversation they are part of, from conversation_start on: his history is
    his, and the guest prompt tells Nova those facts are not in front of her,
    which is only true if they are not. Without a start, only this message."""
    if tier == "hokage":
        return get_recent_messages(20)
    if conversation_start is None:
        return [{"role": "user", "content": user_text}]
    return get_messages_after(conversation_start, 20)


def _prompt_for(channel, tier):
    """The system prompt for this channel and tier, from the memories as they are now."""
    personal = tier == "hokage"
    seed_rows     = get_seed_memories() if personal else (
                    get_shareable_memories() if tier == "jonin" else [])
    episodic_rows = get_episodic_memories() if personal else []
    return build_enhanced_prompt(seed_rows, episodic_rows, channel,
                                 memory_manifest() if personal else [], tier=tier)


def turn_fingerprint(user_text, channel, tier, model, history, pending_alerts,
                     named=False):
    """build_request's fingerprint without the rest of it: no recall, about 3ms.
    What the real turn checks a speculation against."""
    return _fingerprint(user_text, channel, tier, model, _prompt_for(channel, tier),
                        history, pending_alerts, named)


def _log_recall_later(user_text):
    """A claimed speculation recalled with log_recall=False, so this turn has no
    retrieval_log row yet. Written on a thread, off the path to first audio."""
    threading.Thread(target=lambda: search_memories(
        user_text, limit=RECALL_LIMIT, exclude=_resident_ids(), log=True),
        daemon=True).start()


def build_request(user_text, channel, tier, model, history, pending_alerts,
                  log_recall=True, named=False, last_reply=None) -> TurnRequest:
    """The request for one turn, with no side effects.

    Nothing here saves a message, consumes an alert, advances the confirmation
    clock or plays anything; the caller does those, so a speculative turn can
    build exactly the request the real one will, before the endpoint. history
    is get_recent_messages(20) ending in this user message, and pending_alerts
    is what the caller took, or for a speculation, peeked. log_recall=False
    keeps a speculation that may be discarded out of retrieval_log.
    last_reply is when Nova last replied, read by the caller like history, so
    a long gap can be noted; None notes nothing.

    Everything personal is gated on tier here rather than asked for in the
    prompt. Below hokage the corpus is never assembled, so no phrasing,
    insistence, or injection can reach it. Today the only path to hokage is
    verified voice, so this is effectively always hokage; multi speaker
    verification will pass a real tier through without touching this."""
    personal = tier == "hokage"
    enhanced_prompt = _prompt_for(channel, tier)
    fingerprint = _fingerprint(user_text, channel, tier, model, enhanced_prompt,
                               history, pending_alerts, named)

    # Folded before the clock, so all three land inside the same final user turn.
    recent = _with_alerts(_trim_history(history), pending_alerts)
    recent = _with_recalled(recent, user_text, log=log_recall) if personal else recent
    recent = _with_named_note(recent, named)
    recent = _with_gap_note(recent, last_reply)
    recent = _with_current_time(recent)
    recent = _with_text_note(recent, channel)

    # Cache the system prompt. It is the stable prefix: the seed corpus and
    # persona change rarely, while `recent` changes every turn and therefore
    # has to sit after the breakpoint.
    system = ([{"type": "text", "text": enhanced_prompt,
                "cache_control": {"type": "ephemeral"}}]
              if PROMPT_CACHING else enhanced_prompt)
    return TurnRequest(model, system, recent, fingerprint)


def first_call_kwargs(request: TurnRequest) -> dict:
    """The arguments of a turn's first Claude call, in one place.

    A speculative turn opens this same call before the endpoint, and the real
    turn may replay it only if nothing differs, so both build it here."""
    return dict(model=request.model, max_tokens=1000, system=request.system,
                messages=request.messages, tools=registry.api_schemas())


def _run_tools(tool_uses, model, tier):
    """Execute every tool the model called, in order, logging each one.

    Runs on an executor thread from the caller, because the tool functions are
    synchronous: weather blocks on HTTP, and the timer and reminder handlers
    spawn threads. A failure returns its message as the result with is_error
    set rather than raising, so one broken tool cannot take down the turn and
    the model still gets told what happened.

    tier is the one this turn was assembled under, passed in rather than read
    from the database here. ask_nova_async already gates the prompt on it, and
    reading effective_tier() a second time gave the gate its own source of
    truth: once voice verification passes a guest's tier through, the prompt
    would treat them as a guest while the gate still saw hokage.

    A refusal is a result, not an exception. Raising would end the turn in
    silence; returning it lets Nova say out loud that she is not allowed."""
    results = []
    for block in tool_uses:
        started = time.monotonic()
        try:
            spec = registry.get(block.name)
            if not permits(spec, tier):
                output = (f"Refused: {block.name} needs more clearance than "
                          f"{tier}. Say you are not allowed to do that.")
                is_error = True
            else:
                output = registry.call(block.name, dict(block.input))
                is_error = False
        except Exception as exc:
            output = f"{block.name} failed: {exc}"
            is_error = True

        elapsed = (time.monotonic() - started) * 1000.0
        log_tool_call(block.name, dict(block.input), output,
                      is_error=is_error, duration_ms=elapsed, model=model)
        results.append({"block": block, "output": output, "is_error": is_error})
    return results


async def ask_nova_async(user_text: str, device: str = "pi",
                         channel: str = "voice",
                         tier: str = "hokage", on_text=None,
                         speculation=None, named=False,
                         conversation_start=None) -> TurnResult:
    """Run one turn.

    speculation, when given, is a speculative_turn.SpeculativeTurn that may
    already have made this turn's first Claude call during the endpoint wait.
    It is claimed only on an exact match and otherwise ignored; only the voice
    loop passes one.

    named is True when he opened this turn with the wake word; see
    _with_named_note. Only the voice loop knows, so only it passes True.

    conversation_start is the last message id before this conversation began.
    Below hokage it bounds the history; see _history_for.

    on_text, when given, is called with each piece of the reply as it is
    written, so a caller can show it arriving. It is how /chat/stream feeds the
    app; the voice loop passes nothing and is unaffected.

    device is provenance, which client sent this, and is stored as
    source_device. channel is how the answer will be rendered, voice or text,
    and selects the formatting fragment of the prompt. Together they decide
    whether the room speaker plays anything: see _plays_in_the_room. They were
    one parameter until the two meanings drifted apart: the app can want spoken
    output and the Pi could one day want text, so conflating them would make
    either impossible.
    """
    model = _select_model()
    timing.note_model(model)
    # Every Claude turn advances the confirmation clock, so an event proposed
    # on this turn can only be approved on the next one.
    pending_action.begin_turn()

    user_message_id = save_message("user", user_text, device=device)
    # Taken here, not inside build_request: consuming the alerts is a side
    # effect, and a speculative turn has to be able to build the same request
    # without one.
    history = _history_for(tier, conversation_start, user_text)
    pending = alerts.take_for_fold()

    # A speculative turn may already have made this call during the endpoint
    # wait. It is used only if what it was built from is exactly what this turn
    # would build from now; otherwise it is dropped and the call made here.
    replay = None
    if speculation is not None:
        replay = speculation.claim(
            turn_fingerprint(user_text, channel, tier, model, history, pending, named))
    if replay is not None:
        request = speculation.request
        head_start_ms = (time.monotonic() - speculation.opened_at) * 1000.0
        timing.mark('claude_head_start_ms', head_start_ms)
        print(f"(speculative call claimed, sent {head_start_ms:.0f}ms ago)", flush=True)
        if tier == "hokage":
            _log_recall_later(user_text)
    else:
        request = build_request(user_text, channel, tier, model, history, pending,
                                named=named, last_reply=last_reply_at())
    recent       = request.messages
    system_param = request.system

    sentence_queue = asyncio.Queue()
    router         = StreamRouter(sentence_queue)
    spoken_parts   = []
    loop           = asyncio.get_running_loop()
    leaks_seen     = set()  # shared across both TTS consumers and the returned-text strips this turn

    tts_task = asyncio.create_task(
        _consumer_for(channel, device)(sentence_queue, spoken_parts, leaks_seen))

    accumulated   = ""
    bridge        = None
    bridge_key    = None
    claude_start  = time.monotonic()
    first_token   = None
    sentence_seen = False
    stream_source = replay if replay is not None else _claude().messages.stream(
        **first_call_kwargs(request))
    async with stream_source as stream:
        # Iterating raw events rather than stream.text_stream. text_stream
        # yields only text, which is safe, but it also hides block boundaries,
        # so there is no way to tell that a tool call has started. Filtering
        # explicitly on text_delta puts the guarantee that partial JSON never
        # reaches the speaker in this file rather than leaving it inherited
        # from SDK behavior.
        #
        # input_json_delta events are ignored outright. The SDK accumulates
        # them and get_final_message returns the parsed input, so a half
        # written argument has no path to TTS by construction.
        async for event in stream:
            _emit(on_text, "stage", _stage_for(event))
            # Chosen here, played only if the tool turns out to be slow; see
            # _bridge_if_slow. Only when Nova wrote no lead in of her own, and
            # only once.
            key = (_bridge_for(event)
                   if _plays_in_the_room(channel, device) and bridge_key is None else None)
            if key is not None and not accumulated.strip():
                bridge_key = key
            gap = _block_gap(event, accumulated)
            if gap:
                accumulated += gap
                _emit(on_text, "delta", gap)
                await router.feed(gap)
            if event.type != "content_block_delta":
                continue
            if event.delta.type != "text_delta":
                continue

            text = event.delta.text
            if first_token is None:
                first_token = time.monotonic()
                timing.mark('claude_ttft_ms',
                            (first_token - claude_start) * 1000.0)
            accumulated += text
            _emit(on_text, "delta", text)
            await router.feed(text)

            # The wait between the first token and the first speakable
            # sentence. Nothing reaches TTS until a sentence boundary lands,
            # so this sits squarely on the path to first audio and was the
            # bulk of the previously unaccounted time.
            if not sentence_seen and router.sentences_emitted:
                sentence_seen = True
                timing.mark('first_sentence_ms',
                            (time.monotonic() - first_token) * 1000.0)

        final_message = await stream.get_final_message()

    # A turn that called a tool without saying anything first produces no text
    # delta, so time to first token would otherwise never be recorded.
    if first_token is None:
        timing.mark('claude_ttft_ms', (time.monotonic() - claude_start) * 1000.0)

    timing.mark('claude_total_ms', (time.monotonic() - claude_start) * 1000.0)

    usage = final_message.usage
    timing.note_cache(getattr(usage, 'cache_read_input_tokens', None) or 0,
                      getattr(usage, 'cache_creation_input_tokens', None) or 0)
    # A search never reaches the executor, so tool_call_log cannot show it;
    # the journal is where a slow turn gets explained.
    searches = getattr(getattr(usage, 'server_tool_use', None),
                       'web_search_requests', 0) or 0
    if searches:
        print(f"  web search: {searches} request(s), "
              f"{usage.input_tokens} input tokens", flush=True)
    # The server stops its own search loop with pause_turn when it runs long.
    # Nothing resumes it yet: max_uses keeps it rare, and this line is how we
    # find out whether it is rare enough.
    if getattr(final_message, 'stop_reason', None) == "pause_turn":
        print("  web search paused the turn; the reply may be cut short", flush=True)

    # Tags are stripped and discarded rather than saved. The prompt teaches the
    # remember tool now, so a bracket tag reaching here is a leak, not an
    # instruction. Saving it as well would double write the fact the tool
    # already stored, which is the exact duplication the tool exists to prevent.
    accumulated, _, _ = extract_memories(accumulated)
    accumulated = strip_leading_bracket_cue(accumulated, leaks_seen)

    tool_uses = [b for b in final_message.content if b.type == "tool_use"]
    called_tools = {b.name for b in tool_uses}
    # What this turn's own remember calls wrote, so the memory pass after it
    # does not queue the same fact again in pieces.
    remembered = [b.input.get("content", "") for b in tool_uses if b.name == "remember"]

    if tool_uses:
        action_start = time.monotonic()
        await router.finalize()

        # Run the tools and drain the bridge sentence at the same time, so the
        # text Nova already produced alongside the call plays while the work
        # happens rather than after it.
        # The tool is timed on its own, then the bridge is drained. Both still
        # overlap, because tts_task is already a scheduled task and the tool
        # runs on an executor thread, but the measurement now covers only the
        # tool.
        #
        # Previously this awaited asyncio.gather(tool, tts_task) inside the
        # stopwatch, so tool_ms reported whichever finished last. A weather call
        # that took 580ms logged 6006ms, because what it was really measuring
        # was the bridge sentence playing through the speaker.
        tool_future = loop.run_in_executor(None, _run_tools, tool_uses, model, tier)
        tool_start  = time.monotonic()
        bridge      = await _bridge_if_slow(tool_future, bridge_key)
        results     = await tool_future
        timing.note_tool((time.monotonic() - tool_start) * 1000.0)
        await tts_task

        # dismiss is a state change, not work. The tool executes nothing; the
        # transition happens here. Read before filtering so a turn that both
        # answers something and says goodbye still runs its real tools.
        dismissed = any(r["block"].name == "dismiss" for r in results)
        ignored   = any(r["block"].name == "ignore" for r in results)

        # This replaces the hardcoded `any(r["type"] == "weather")` whitelist.
        # Whether a second call happens is now a property of the tool, declared
        # once at registration, so adding a read shaped tool needs no edit here.
        # A proposed change is asked in code, word for word. On Sep 14 2026 the
        # staged event was dated the 21st while Nova said "September 14", and his
        # yes created the one he never heard. With the question spoken here,
        # nothing between what runs and what he hears can reword it, and a
        # confirmation no longer costs a second Claude call.
        staged = pending_action.words_for_turn()
        needs_second_call = _needs_second_call(results) and staged is None

        if needs_second_call:
            # Whatever Nova said alongside the call is not part of the answer
            # the tool result produces. A reader who was shown it has to be told
            # to drop it. A listener cannot be: see lead_in below.
            lead_in = " ".join(spoken_parts).strip()
            _emit(on_text, "reset", "")
            # Real tool_result blocks, not a synthetic user turn describing the
            # data in prose. The model gets the result attached to the call it
            # made, which is what lets it answer about the result rather than
            # about a message that happens to contain it.
            followup_messages = recent + [
                {"role": "assistant", "content": final_message.content},
                {"role": "user", "content": _tool_result_blocks(results)},
            ]

            sentence_queue2 = asyncio.Queue()
            router2         = StreamRouter(sentence_queue2)
            spoken_parts2   = []
            tts_task2       = asyncio.create_task(
                _consumer_for(channel, device)(sentence_queue2, spoken_parts2, leaks_seen)
            )

            # A loop, not a single call. The follow up is free to request
            # another tool rather than answer, and it intermittently does: it
            # re-called get_weather instead of reading the result back, which
            # produced no text at all and returned an empty turn. Treating the
            # follow up as guaranteed to speak was the bug.
            #
            # MAX_TOOL_ROUNDS bounds it, and the last round drops `tools` so the
            # model has nothing left to call and must answer. That is a hard
            # floor on the failure rather than a hope that it converges.
            # A turn that only stored something has nothing left to look up.
            # Offered tools, the follow up still ended in silence in 1 of 6
            # measured turns; without them Nova answered in 6 of 6.
            only_stored  = called_tools <= {"remember"}
            final_text   = ""
            second_start = time.monotonic()
            second_first = None
            for round_index in range(MAX_TOOL_ROUNDS):
                last_round = round_index == MAX_TOOL_ROUNDS - 1
                kwargs = {} if last_round or only_stored else {"tools": registry.api_schemas()}

                round_text = ""
                async with _claude().messages.stream(
                    model=model,
                    max_tokens=1000,
                    system=system_param,
                    messages=followup_messages,
                    **kwargs,
                ) as stream2:
                    async for event in stream2:
                        _emit(on_text, "stage", _stage_for(event))
                        gap = _block_gap(event, round_text)
                        if gap:
                            round_text += gap
                            _emit(on_text, "delta", gap)
                            await router2.feed(gap)
                        if event.type != "content_block_delta":
                            continue
                        if event.delta.type != "text_delta":
                            continue
                        if second_first is None:
                            second_first = time.monotonic()
                            timing.note_second_ttft(
                                (second_first - second_start) * 1000.0)
                        round_text += event.delta.text
                        _emit(on_text, "delta", event.delta.text)
                        await router2.feed(event.delta.text)

                    followup_message = await stream2.get_final_message()

                # Joined with a space. Concatenated bare, a sentence ending one
                # round ran into the next in history: "Monday.LeetCode".
                final_text = f"{final_text} {round_text}".strip()
                more_tools = [b for b in followup_message.content
                              if b.type == "tool_use"]
                called_tools |= {b.name for b in more_tools}
                remembered += [b.input.get("content", "") for b in more_tools
                               if b.name == "remember"]
                if not more_tools:
                    break

                more_results = await loop.run_in_executor(
                    None, _run_tools, more_tools, model, tier)
                if pending_action.words_for_turn() is not None:
                    # A change proposed after a lookup. Its question is spoken
                    # in code below, not in the model's words.
                    break
                followup_messages = followup_messages + [
                    {"role": "assistant", "content": followup_message.content},
                    {"role": "user", "content": _tool_result_blocks(more_results)},
                ]

            await router2.finalize()
            await tts_task2
            final_text, _, _ = extract_memories(final_text)
            final_text = strip_leading_bracket_cue(final_text, leaks_seen)
            late_question = pending_action.words_for_turn()
            if late_question is not None:
                await _say(late_question, channel, device)
                _emit(on_text, "delta", f" {late_question}" if final_text else late_question)
                final_text = f"{final_text} {late_question}".strip()

            # History has to be what he heard. The lead in was dropped as a
            # "let me check", but alongside remember it is often the whole
            # reply: on Sep 28 2026 she asked "How's the toe feeling now?" there,
            # the follow up said nothing, and the turn saved nothing, so her
            # next turn never saw her own question. Replayed, 3 of 5 runs saved
            # something other than what was spoken. A spoken lead in is kept; on
            # the app, where it was cleared, it comes back only when the follow
            # up is empty, so the reply is never blank.
            if _plays_in_the_room(channel, device):
                final_text = f"{lead_in} {final_text}".strip()
            elif not final_text and lead_in:
                _emit(on_text, "delta", lead_in)
                final_text = lead_in
        else:
            # Timer, reminder, cancel, dismiss: the text Nova said alongside
            # the call is the answer. A second round trip here would spend a
            # second of latency rephrasing "Timer set."
            final_text = " ".join(spoken_parts).strip()
            if staged is not None:
                await _say(staged, channel, device)
                _emit(on_text, "delta", f" {staged}" if final_text else staged)
                final_text = f"{final_text} {staged}".strip()

            # A control tool is a state transition, not work, so there is
            # nothing to confirm and nothing to say.
            #
            # This is the exception the fallback below needs, and leaving it
            # out was a real defect. NOT_ADDRESSED_TO_YOU tells Nova to say
            # nothing at all when speech was not aimed at her, so behaving
            # correctly produces an empty spoken_parts, which then hit the
            # fallback and spoke "Done." into a room she had just decided not
            # to join. Observed Sep 6 2026: a false wake at 0.50 transcribed as
            # "over here", the model called ignore, voice_main logged "Not
            # addressed to Nova, staying quiet", and the word had already been
            # said. The better she behaved, the more certain she was to speak.
            only_control = all(
                registry.get(r["block"].name).permission is Permission.CONTROL
                for r in results
            )

            # The model does not always say anything alongside a fire and
            # forget call. When it does not, this turn used to be completely
            # silent: the fallback was assigned to the returned string but
            # never spoken, so the database recorded a confirmation that was
            # never heard. Timers, reminders and cancellations all landed this
            # way, which read as the tool having failed when it had worked.
            if not final_text and not only_control:
                final_text = "Done."
                await _say(final_text, channel, device)
                _emit(on_text, "delta", final_text)

        if bridge is not None:
            try:
                await bridge
            except Exception as exc:
                print(f"Bridge line failed: {exc}", flush=True)
        timing.note_action((time.monotonic() - action_start) * 1000.0)
    else:
        dismissed = False
        ignored   = False
        await router.finalize()
        await tts_task
        final_text = accumulated

    # An empty response is a turn Nova deliberately stayed silent through, and
    # writing it down puts a blank assistant message into the history that the
    # next twenty turns read back as context. save_message is skipped rather
    # than storing "", because get_recent_messages does not filter and the API
    # rejects an empty content block.
    if _asks_an_unstaged_change(final_text, pending_action.words_for_turn()):
        print(f"Asked to change the calendar with nothing staged: {final_text[:120]!r}",
              flush=True)

    if _claims_a_save_without_calling(final_text, called_tools):
        print(f"Claimed to remember without calling remember: {final_text[:120]!r}",
              flush=True)

    if ignored and not final_text:
        delete_message(user_message_id)
    if final_text:
        save_message("assistant", final_text, device=device)
    # After the reply, on its own thread. Only for him: a guest's words are not
    # facts about Lethanial, and speech not meant for Nova is not his to keep.
    if tier == "hokage" and not ignored:
        memory_pass.notice_later(user_text, datetime.now(), remembered)
    return TurnResult(text=final_text, dismissed=dismissed, ignored=ignored)


def ask_nova(user_text: str, device: str = "pi",
             channel: str = "voice", tier: str = None, on_text=None,
             speculation=None, named=False,
             conversation_start=None) -> TurnResult:
    # None means "look it up", so a demotion set in one turn is in force on the
    # next without the caller having to thread it through.
    loop = _thread_loop()
    try:
        return loop.run_until_complete(
            ask_nova_async(user_text, device=device, channel=channel,
                           tier=tier or effective_tier(), on_text=on_text,
                           speculation=speculation, named=named,
                           conversation_start=conversation_start))
    finally:
        _cancel_leftovers(loop)
