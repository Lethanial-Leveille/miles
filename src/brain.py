import asyncio
import anthropic
from audio import speak
from prompts import build_enhanced_prompt, SYSTEM_PROMPT
from database import save_message, get_memories, get_recent_messages, save_memory
from parsing import extract_memories
from actions import execute_actions
from stream_router import StreamRouter

claude = anthropic.AsyncAnthropic()

MODEL = "claude-sonnet-4-20250514"


async def _tts_consumer(queue: asyncio.Queue, text_parts: list) -> None:
    """Pull sentences off the queue and speak them sequentially."""
    loop = asyncio.get_running_loop()
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
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
    save_message("user", user_text, device=device)
    memory_rows     = get_memories()
    enhanced_prompt = build_enhanced_prompt(memory_rows)
    recent          = get_recent_messages(20)

    sentence_queue = asyncio.Queue()
    router         = StreamRouter(sentence_queue)
    spoken_parts   = []
    loop           = asyncio.get_running_loop()

    tts_task = asyncio.create_task(_tts_consumer(sentence_queue, spoken_parts))

    accumulated = ""
    async with claude.messages.stream(
        model=MODEL,
        max_tokens=300,
        system=enhanced_prompt,
        messages=recent,
    ) as stream:
        async for text in stream.text_stream:
            accumulated += text
            await router.feed(text)

    accumulated, explicit_mems, implicit_mems = extract_memories(accumulated)
    for mem in explicit_mems:
        save_memory(mem, source="explicit")
    for mem in implicit_mems:
        save_memory(mem, source="implicit")

    if router.action_tag:
        await router.finalize()
        action = _parse_action_tag(router.action_tag)

        # Run the action and drain bridge-sentence TTS at the same time
        results, _ = await asyncio.gather(
            loop.run_in_executor(None, execute_actions, [action]),
            tts_task,
        )

        needs_data = any(r["type"] == "weather" for r in results)

        if needs_data:
            data_block = "\n".join(
                f"{r['type'].upper()} DATA: {r['data']}" for r in results
            )
            bridge_text = " ".join(spoken_parts)
            followup_messages = recent + [
                {"role": "assistant", "content": bridge_text or router.action_tag},
                {"role": "user", "content": (
                    f"[SYSTEM: Here is the data you requested]\n{data_block}\n"
                    "Deliver this information naturally as Nova. Stay in character. Keep it concise."
                )},
            ]

            sentence_queue2 = asyncio.Queue()
            router2         = StreamRouter(sentence_queue2)
            spoken_parts2   = []
            tts_task2       = asyncio.create_task(
                _tts_consumer(sentence_queue2, spoken_parts2)
            )

            final_text = ""
            async with claude.messages.stream(
                model=MODEL,
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=followup_messages,
            ) as stream2:
                async for text in stream2.text_stream:
                    final_text += text
                    await router2.feed(text)

            await router2.finalize()
            await tts_task2
            final_text, _, _ = extract_memories(final_text)
        else:
            # Timer / reminder / cancel: TTS already spoke the confirmation
            final_text = " ".join(spoken_parts) or "Done."
    else:
        await router.finalize()
        await tts_task
        final_text = accumulated

    save_message("assistant", final_text, device=device)
    return final_text


def ask_nova(user_text: str, device: str = "pi") -> str:
    return asyncio.run(ask_nova_async(user_text, device=device))
