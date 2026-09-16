"""Notice lasting facts in what he says, after Nova has answered.

Measured Sep 16 2026 by replaying his real messages since Sep 13: when a message
also asked for something, Nova did the task and stored nothing, every time.
"Charley is spelled with an EY, can you fix that" renamed the events and kept
nothing. Remembering was a side job in a long prompt, and a side job loses to
the main one. A prompt line saying so moved it from 4 of 24 to 7 of 24.

This pass gives noticing its own call with one tool and one job: 16 of 24, with
1 of 16 plain questions wrongly stored. It runs on a thread after the turn, so
it adds nothing to what he waits for.

Everything it stores is pending, whatever the model says, so a wrong guess
costs him one discard in the app. It never supersedes: in this schema a
supersede activates the replacement at once, which would let a guess overwrite
a correct memory with no review at all.
"""

import threading

import anthropic

import memory_tool
import prompts
from config import MEMORY_PASS, MEMORY_PASS_MODEL
from database import (get_seed_memories, get_episodic_memories,
                      get_pending_memories, get_recent_messages, log_tool_call)
from tools import registry

client = anthropic.Anthropic()

# Past this, a message is being mined rather than remembered. The replay's
# worst case wrote four memories from one message, one of them from an earlier
# turn.
MAX_PER_MESSAGE = 3

_PROMPT = """You keep Lethanial's long term memory for Nova, his voice assistant. You are not talking to him, and nothing you write is shown to anyone.

You are given his latest message, and the conversation before it for reference. Store only what his latest message itself says. The earlier conversation is there so you know who "she" or "that" refers to; anything that appears only there was already considered and must not be stored again.

Store a fact when the latest message tells you something lasting about his life that the list below does not already say: a person and who they are to him, how a name is spelled, someone's pronouns, a job or a change to one, a class, something that repeats on his schedule, a date that matters, a project, or a rule he stated about his own time. A fact mentioned in passing while asking for something else counts.

For each fact, call remember once, written as a standalone statement that still makes sense months later, using only what he said. Do not add details, guesses in parentheses, or dates he did not give; if a date is relative, like "this Saturday", work it out from the date given with his message. If it stops being true on a known date, set temporary and until. Store at most three facts from one message, the ones that would most change what Nova says later.

Store nothing for questions, requests, small talk, one off logistics like moving a single event, speech that was not meant for Nova, or anything the list already says. Most messages contain nothing to store, and calling nothing is the usual right answer.

What Nova already knows, with ids, including facts still waiting for his review:"""


def _known():
    pending = get_pending_memories()
    waiting = ("\n\nWaiting for his review:\n"
               + "\n".join(f"(#{row[0]}) {row[1]}" for row in pending)) if pending else ""
    return (prompts._seed_block(get_seed_memories())
            + prompts._episodic_block(get_episodic_memories()) + waiting)


def _before(user_text, rows, keep=6):
    """The conversation before his latest message. The pass runs after the reply
    is saved, so the last rows are this turn; everything up to and including his
    message is dropped."""
    for index in range(len(rows) - 1, -1, -1):
        if rows[index]["role"] == "user" and rows[index]["content"] == user_text:
            return rows[:index][-keep:]
    return rows[-keep:]


def _request(user_text, when):
    prior = _before(user_text, get_recent_messages(10))
    lines = "\n".join(f"{'Lethanial' if m['role'] == 'user' else 'Nova'}: {m['content']}"
                      for m in prior) or "(nothing earlier)"
    stamp = f"{when:%A, %B} {when.day}, {when:%Y}"
    return (f"Earlier conversation, for reference only:\n{lines}\n\n"
            f"His latest message, sent {stamp}:\n{user_text}")


def notice(user_text, when):
    """One pass over one message. Returns what was stored, for the tests and the log."""
    schema = [s for s in registry.api_schemas() if s["name"] == "remember"]
    reply = client.messages.create(
        model=MEMORY_PASS_MODEL, max_tokens=600, tools=schema,
        system=[{"type": "text", "text": _PROMPT + _known(),
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _request(user_text, when)}])

    waiting = {row[1].strip().casefold() for row in get_pending_memories()}
    stored = []
    for block in reply.content:
        if block.type != "tool_use" or block.name != "remember":
            continue
        if len(stored) >= MAX_PER_MESSAGE:
            break
        content = str(block.input.get("content") or "").strip()
        # Seen from Sonnet 5 in the replay: a call with no content at all.
        if not content or content.casefold() in waiting:
            continue
        arguments = {"content": content, "certainty": "inferred",
                     "temporary": bool(block.input.get("temporary")),
                     "until": block.input.get("until") or None}
        result = memory_tool.remember(**arguments)
        log_tool_call("remember", arguments, result, model=f"memory_pass:{MEMORY_PASS_MODEL}")
        waiting.add(content.casefold())
        stored.append(content)
    return stored


def notice_later(user_text, when):
    """Run notice on a thread, so the turn never waits for it. A failure is logged
    and dropped: a missed memory is recoverable, a broken turn is not."""
    if not MEMORY_PASS:
        return None

    def run():
        try:
            stored = notice(user_text, when)
            if stored:
                print(f"Memory pass queued {len(stored)} for review", flush=True)
        except Exception as exc:
            print(f"Memory pass failed ({type(exc).__name__}): {exc}", flush=True)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
