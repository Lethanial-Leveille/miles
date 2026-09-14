"""The remember tool.

Replaces the [MEMORY:] and [MEMORY-EXPLICIT:] bracket tags. Those could only
ever do one thing, add a row, which is why an implicit memory about a Singapore
trip ended up in the review queue alongside two seed rows that already covered
it. The duplicate guard was exact string match, so two different sentences about
the same fact both got stored.

The tool exists to make the other two moves expressible:

  store       the fact is new
  supersede   the fact is already known and has changed or sharpened
  skip        the fact is already known, so do nothing

Only the model can tell those apart, and it can, because every memory is in its
prompt with an id. That is why this needed ids more than it needed retrieval.

Blocked until Aug 11 2026 on supersede and expiry existing. Automating writes
into a store that cannot correct itself makes errors permanent and accumulate,
and the failure gets worse the better the tool works.
"""

from database import (approve_memory, delete_memory, get_pending_memories,
                      memory_content, save_memory, supersede_memory)
from tools import Permission, tool


@tool(
    name="remember",
    description=(
        "Store something worth knowing about Lethanial for future "
        "conversations, or correct something already stored. "
        "Call this in the same turn whenever he tells you to remember "
        "something: store what he said, in his words, with certainty asked, "
        "even if part of it is unclear to you. Ask your question after, never "
        "instead. "
        "Also call this, with certainty inferred, when he mentions something "
        "about his life that would change what you say to him later: a new "
        "person, tool, routine, commitment, plan or preference, or a change to "
        "something you already hold. Those wait for his review, so a "
        "reasonable guess costs little. "
        "Read what you already know first, in the list above. If the fact is "
        "already there in some form, pass its id as supersedes to replace it, "
        "or do not call this at all if nothing changed. "
        "Do not store questions. Asking when his exam is is a retrieval, not "
        "new information. "
        "This returns nothing to say. Never announce that you stored, updated "
        "or skipped something unless he asks, and never say you noted, saved "
        "or will remember something unless you called this tool in this turn."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact, written as a standalone statement "
                               "that still makes sense months later with no "
                               "surrounding conversation.",
            },
            "supersedes": {
                "type": "integer",
                "description": "Id of the memory this replaces, from the "
                               "numbers shown as (#61) in your memory list. "
                               "Use it whenever this is a changed or sharper "
                               "version of something already stored, rather "
                               "than storing a second copy.",
            },
            "certainty": {
                "type": "string",
                "enum": ["asked", "inferred"],
                "description": "'asked' only when Lethanial directly told you "
                               "to remember it, which stores it immediately. "
                               "'inferred' when you noticed it yourself, which "
                               "holds it for his review. Defaults to inferred, "
                               "because guessing confidently puts your "
                               "assumptions into his permanent record.",
            },
            "temporary": {
                "type": "boolean",
                "description": "True when the fact has a shelf life, such as "
                               "training for a meet in March. Pair it with "
                               "until, since a temporary fact with no date "
                               "never expires.",
            },
            "until": {
                "type": "string",
                "description": "ISO 8601 date after which a temporary fact "
                               "stops being true, computed from the clock "
                               "supplied with his message.",
            },
        },
        "required": ["content"],
    },
    permission=Permission.WRITE,
    # Nothing to speak about. The answer is whatever Nova was already saying,
    # and a second round trip to announce a save would make every remembered
    # fact cost a full extra turn of latency.
    returns_to_model=False,
    min_tier="jonin",
)
def remember(content, supersedes=None, certainty="inferred",
             temporary=False, until=None):
    # 'asked' is trusted immediately because Lethanial said it out loud.
    # 'inferred' goes to the review queue, which is the same split the old
    # explicit and implicit tags encoded and the reason that queue exists.
    source = "explicit" if certainty == "asked" else "implicit"
    status = "active" if certainty == "asked" else "pending"

    # Replacing a memory with the same words changes nothing except its history:
    # a new row, the old one retired, a chain link that records no change. Seen
    # Sep 13 2026, when "I'm taking twelve credits" rewrote the memory that
    # already said exactly that.
    if supersedes is not None and _same_fact(memory_content(supersedes), content):
        return "already stored, nothing changed"

    if supersedes is not None:
        new_id = supersede_memory(supersedes, content, source=source,
                                  volatile=temporary, references_date=until)
        if new_id is None:
            # The id was wrong. Store it rather than losing the fact, since a
            # bad reference is a worse reason to drop information than a
            # duplicate is to keep it.
            save_memory(content, source=source, status=status,
                        volatile=temporary, references_date=until)
            return f"no memory {supersedes}; stored as new instead"
        return f"replaced memory {supersedes}"

    saved = save_memory(content, source=source, status=status,
                        volatile=temporary, references_date=until)
    if not saved:
        return "already stored, nothing changed"
    return "stored for review" if status == "pending" else "stored"


# The review queue, reachable by voice. Inferred memories wait for approval,
# and until Sep 13 2026 the only way to see them was scripts/memory.py, which is
# part of why nothing Nova might have noticed ever reached him.

@tool(
    name="list_pending_memories",
    description=(
        "The things you noticed and stored on your own that are waiting for "
        "Lethanial's approval, newest first, each with an id. Call this when he "
        "asks what you have noted, what you have been remembering, or wants to "
        "go through what is waiting. Read them back plainly, one short line "
        "each, without the ids."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def list_pending_memories():
    rows = get_pending_memories(limit=20)
    if not rows:
        return "Nothing is waiting for his review."
    return "\n".join(f"(#{row[0]}) {row[1]}" for row in rows)


@tool(
    name="review_pending_memory",
    description=(
        "Keep or discard one memory that is waiting for Lethanial's approval, "
        "by its id from list_pending_memories. Call this when he says to keep "
        "or drop something you read back to him. Keeping makes it part of what "
        "you know; discarding deletes it. Only for items he actually answered "
        "about; never approve anything on your own."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer",
                          "description": "The id from list_pending_memories."},
            "keep": {"type": "boolean",
                     "description": "True to keep it, false to discard it."},
        },
        "required": ["memory_id", "keep"],
    },
    permission=Permission.WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def review_pending_memory(memory_id, keep):
    # Only a pending id is accepted. delete_memory would happily remove an
    # established memory, and a misheard "discard that" must not be able to.
    if memory_id not in {row[0] for row in get_pending_memories(limit=1000)}:
        raise LookupError(f"no memory {memory_id} is waiting for review")
    if keep:
        approve_memory(memory_id)
        return "Kept."
    delete_memory(memory_id)
    return "Discarded."


def _same_fact(old, new):
    """Same words, ignoring case, spacing and a trailing full stop."""
    if old is None:
        return False
    normal = lambda text: " ".join(text.casefold().rstrip(". ").split())
    return normal(old) == normal(new)
