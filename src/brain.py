import asyncio
import time
from datetime import datetime
from typing import NamedTuple

import anthropic
import timing
from tts import speak
from prompts import build_enhanced_prompt
from database import save_message, get_seed_memories, get_episodic_memories, get_recent_messages, save_memory
from parsing import extract_memories, strip_leading_bracket_cue
from stream_router import StreamRouter
from tools import registry
from database import log_tool_call

# Imported for its side effect: registering the tools. Without it the registry
# is empty, the capability block is blank, and Nova silently has no
# capabilities. Kept explicit, with this comment, because an unused looking
# import is exactly what a later cleanup deletes.
import actions        # noqa: F401
import system_state   # noqa: F401

from config import (MODEL_AB_TEST, MODEL_A, MODEL_B, PROMPT_CACHING,
                    HISTORY_ASSISTANT_WORDS)

claude = anthropic.AsyncAnthropic()


class TurnResult(NamedTuple):
    """What one turn produced.

    `dismissed` is set when Nova recognized an intent to end the conversation
    and emitted [ACTION: dismiss]. Returning it here rather than stashing it in
    module state keeps the signal explicit at the call site, which matters
    because two different callers drive this: the voice loop, which acts on it,
    and the HTTP server, which does not."""
    text: str
    dismissed: bool = False

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


async def _tts_consumer(queue: asyncio.Queue, text_parts: list, leaks_seen: set) -> None:
    """Pull sentences off the queue and speak them sequentially."""
    loop = asyncio.get_running_loop()
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        sentence = strip_leading_bracket_cue(sentence, leaks_seen)
        text_parts.append(sentence)
        # speak() blocks (holds speak_lock + waits for aplay), so run in a thread
        await loop.run_in_executor(None, speak, sentence)


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


# How many times the model may go back for another tool before it has to
# answer. Two rounds covers the real case (call a tool, read it back) plus one
# genuine chained call; the final round is made without tools so there is
# nothing left to reach for.
MAX_TOOL_ROUNDS = 3


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


def _run_tools(tool_uses, model):
    """Execute every tool the model called, in order, logging each one.

    Runs on an executor thread from the caller, because the tool functions are
    synchronous: weather blocks on HTTP, and the timer and reminder handlers
    spawn threads. A failure returns its message as the result with is_error
    set rather than raising, so one broken tool cannot take down the turn and
    the model still gets told what happened."""
    results = []
    for block in tool_uses:
        started = time.monotonic()
        try:
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


async def ask_nova_async(user_text: str, device: str = "pi") -> TurnResult:
    model = _select_model()
    timing.note_model(model)

    save_message("user", user_text, device=device)
    seed_rows       = get_seed_memories()
    episodic_rows   = get_episodic_memories()
    enhanced_prompt = build_enhanced_prompt(seed_rows, episodic_rows, device)
    recent          = _with_current_time(_trim_history(get_recent_messages(20)))

    sentence_queue = asyncio.Queue()
    router         = StreamRouter(sentence_queue)
    spoken_parts   = []
    loop           = asyncio.get_running_loop()
    leaks_seen     = set()  # shared across both TTS consumers and the returned-text strips this turn

    tts_task = asyncio.create_task(_tts_consumer(sentence_queue, spoken_parts, leaks_seen))

    # Cache the system prompt. It is the stable prefix: the seed corpus and
    # persona change rarely, while `recent` changes every turn and therefore
    # has to sit after the breakpoint.
    system_param = (
        [{"type": "text", "text": enhanced_prompt,
          "cache_control": {"type": "ephemeral"}}]
        if PROMPT_CACHING else enhanced_prompt
    )

    accumulated   = ""
    claude_start  = time.monotonic()
    first_token   = None
    sentence_seen = False
    async with claude.messages.stream(
        model=model,
        max_tokens=1000,
        system=system_param,
        messages=recent,
        tools=registry.api_schemas(),
    ) as stream:
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

    accumulated, explicit_mems, implicit_mems = extract_memories(accumulated)
    accumulated = strip_leading_bracket_cue(accumulated, leaks_seen)
    for mem in explicit_mems:
        save_memory(mem, source="explicit", status="active")
    for mem in implicit_mems:
        save_memory(mem, source="implicit", status="pending")

    tool_uses = [b for b in final_message.content if b.type == "tool_use"]

    if tool_uses:
        action_start = time.monotonic()
        await router.finalize()

        # Run the tools and drain the bridge sentence at the same time, so the
        # text Nova already produced alongside the call plays while the work
        # happens rather than after it.
        tool_start = time.monotonic()
        results, _ = await asyncio.gather(
            loop.run_in_executor(None, _run_tools, tool_uses, model),
            tts_task,
        )
        timing.note_tool((time.monotonic() - tool_start) * 1000.0)

        # dismiss is a state change, not work. The tool executes nothing; the
        # transition happens here. Read before filtering so a turn that both
        # answers something and says goodbye still runs its real tools.
        dismissed = any(r["block"].name == "dismiss" for r in results)

        # This replaces the hardcoded `any(r["type"] == "weather")` whitelist.
        # Whether a second call happens is now a property of the tool, declared
        # once at registration, so adding a read shaped tool needs no edit here.
        needs_second_call = any(
            registry.get(r["block"].name).returns_to_model for r in results
        )

        if needs_second_call:
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
                _tts_consumer(sentence_queue2, spoken_parts2, leaks_seen)
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
            final_text   = ""
            second_start = time.monotonic()
            second_first = None
            for round_index in range(MAX_TOOL_ROUNDS):
                last_round = round_index == MAX_TOOL_ROUNDS - 1
                kwargs = {} if last_round else {"tools": registry.api_schemas()}

                round_text = ""
                async with claude.messages.stream(
                    model=model,
                    max_tokens=1000,
                    system=system_param,
                    messages=followup_messages,
                    **kwargs,
                ) as stream2:
                    async for event in stream2:
                        if event.type != "content_block_delta":
                            continue
                        if event.delta.type != "text_delta":
                            continue
                        if second_first is None:
                            second_first = time.monotonic()
                            timing.note_second_ttft(
                                (second_first - second_start) * 1000.0)
                        round_text += event.delta.text
                        await router2.feed(event.delta.text)

                    followup_message = await stream2.get_final_message()

                final_text += round_text
                more_tools = [b for b in followup_message.content
                              if b.type == "tool_use"]
                if not more_tools:
                    break

                more_results = await loop.run_in_executor(
                    None, _run_tools, more_tools, model)
                followup_messages = followup_messages + [
                    {"role": "assistant", "content": followup_message.content},
                    {"role": "user", "content": _tool_result_blocks(more_results)},
                ]

            await router2.finalize()
            await tts_task2
            final_text, _, _ = extract_memories(final_text)
            final_text = strip_leading_bracket_cue(final_text, leaks_seen)
        else:
            # Timer, reminder, cancel, dismiss: the text Nova said alongside
            # the call is the answer. A second round trip here would spend a
            # second of latency rephrasing "Timer set."
            final_text = " ".join(spoken_parts) or "Done."

        timing.note_action((time.monotonic() - action_start) * 1000.0)
    else:
        dismissed = False
        await router.finalize()
        await tts_task
        final_text = accumulated

    save_message("assistant", final_text, device=device)
    return TurnResult(text=final_text, dismissed=dismissed)


def ask_nova(user_text: str, device: str = "pi") -> TurnResult:
    return asyncio.run(ask_nova_async(user_text, device=device))
