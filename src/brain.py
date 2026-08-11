import asyncio
import time

import anthropic
import timing
from tts import speak
from prompts import build_enhanced_prompt
from database import save_message, get_seed_memories, get_episodic_memories, get_recent_messages, save_memory
from parsing import extract_memories, strip_leading_bracket_cue
from actions import execute_actions
from stream_router import StreamRouter

from config import MODEL_AB_TEST, MODEL_A, MODEL_B, PROMPT_CACHING

claude = anthropic.AsyncAnthropic()

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


def _parse_action_tag(tag: str) -> dict:
    """Parse '[ACTION: weather | location: Gainesville]' into action dict."""
    inner = tag.strip().lstrip('[').rstrip(']')
    if inner.upper().startswith('ACTION:'):
        inner = inner[7:].strip()
    parts       = [p.strip() for p in inner.split('|')]
    action_type = parts[0].lower()
    params      = {}
    for part in parts[1:]:
        if ':' in part:
            key, value = part.split(':', 1)
            params[key.strip()] = value.strip()
        else:
            params['value'] = part.strip()
    return {"type": action_type, "params": params}


async def ask_nova_async(user_text: str, device: str = "pi") -> str:
    model = _select_model()
    timing.note_model(model)

    save_message("user", user_text, device=device)
    seed_rows       = get_seed_memories()
    episodic_rows   = get_episodic_memories()
    enhanced_prompt = build_enhanced_prompt(seed_rows, episodic_rows, device)
    recent          = get_recent_messages(20)

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
    ) as stream:
        async for text in stream.text_stream:
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

    if router.action_tags:
        action_start = time.monotonic()
        await router.finalize()
        actions = [_parse_action_tag(tag) for tag in router.action_tags]

        # Run every action and drain bridge-sentence TTS at the same time.
        # execute_actions already loops, so a turn asking for two things at
        # once dispatches both instead of silently dropping the second.
        results, _ = await asyncio.gather(
            loop.run_in_executor(None, execute_actions, actions),
            tts_task,
        )

        needs_data = any(r["type"] == "weather" for r in results)

        if needs_data:
            data_block = "\n".join(
                f"{r['type'].upper()} DATA: {r['data']}" for r in results
            )
            bridge_text = " ".join(spoken_parts)
            followup_messages = recent + [
                {"role": "assistant", "content": bridge_text or " ".join(router.action_tags)},
                {"role": "user", "content": (
                    f"[SYSTEM: Here is the data you requested]\n{data_block}\n"
                    "Deliver this information naturally as Nova. Stay in character. Keep it concise."
                )},
            ]

            sentence_queue2 = asyncio.Queue()
            router2         = StreamRouter(sentence_queue2)
            spoken_parts2   = []
            tts_task2       = asyncio.create_task(
                _tts_consumer(sentence_queue2, spoken_parts2, leaks_seen)
            )

            final_text = ""
            async with claude.messages.stream(
                model=model,
                max_tokens=1000,
                system=system_param,
                messages=followup_messages,
            ) as stream2:
                async for text in stream2.text_stream:
                    final_text += text
                    await router2.feed(text)

            await router2.finalize()
            await tts_task2
            final_text, _, _ = extract_memories(final_text)
            final_text = strip_leading_bracket_cue(final_text, leaks_seen)
        else:
            # Timer / reminder / cancel: TTS already spoke the confirmation
            final_text = " ".join(spoken_parts) or "Done."

        timing.note_action((time.monotonic() - action_start) * 1000.0)
    else:
        await router.finalize()
        await tts_task
        final_text = accumulated

    save_message("assistant", final_text, device=device)
    return final_text


def ask_nova(user_text: str, device: str = "pi") -> str:
    return asyncio.run(ask_nova_async(user_text, device=device))
