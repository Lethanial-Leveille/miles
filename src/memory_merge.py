"""A first draft of one sentence that holds two memories, for him to edit.

Only a draft. On Sep 28 2026 Haiku, asked to decide whether two memories were
the same fact, called two in three different facts duplicates or updates
(docs/BRAIN.md). Writing a sentence he then reads is a different and easier
job: nothing is stored until he saves it in the app, and he sees every word.
"""

import anthropic

from config import MEMORY_PASS_MODEL

client = anthropic.Anthropic()

_PROMPT = """Two stored facts about Lethanial say overlapping things. Write one standalone statement that holds everything both say.

Keep every detail from both: names, dates, numbers, reasons. Add nothing neither says. Where they disagree, the second is newer and wins. Write it the way the facts are written, in the third person, as plainly as they are.

Reply with the statement alone."""


def draft(older, newer):
    """(text, drafted). drafted is False when the model could not be reached;
    the text is then the longer of the two, which is the better start for an
    edit by hand than an empty box."""
    fallback = max((older, newer), key=len)
    try:
        reply = client.messages.create(
            model=MEMORY_PASS_MODEL, max_tokens=300, system=_PROMPT,
            messages=[{"role": "user", "content": f"First:\n{older}\n\nSecond:\n{newer}"}])
        text = "".join(b.text for b in reply.content if b.type == "text").strip()
    except Exception as exc:
        print(f"Merge draft failed ({type(exc).__name__}): {exc}", flush=True)
        return fallback, False
    return (text, True) if text else (fallback, False)
