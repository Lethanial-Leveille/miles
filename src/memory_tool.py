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

from database import save_memory, supersede_memory
from tools import Permission, tool


@tool(
    name="remember",
    description=(
        "Store something worth knowing about Lethanial for future "
        "conversations, or correct something already stored. "
        "Call this when he tells you to remember something, or when he shares "
        "a fact, preference, habit or schedule detail that will still matter "
        "next week. "
        "Read what you already know first, in the list above. If the fact is "
        "already there in some form, either pass its id as supersedes to "
        "replace it, or do not call this at all. A second copy of a fact you "
        "already hold is worse than not storing it: both end up in front of "
        "you, and when one is corrected they disagree. "
        "Do not store questions. Asking when his exam is is a retrieval, not "
        "new information. "
        "This returns nothing to say. Never announce that you stored, updated "
        "or skipped something unless he asks."
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
)
def remember(content, supersedes=None, certainty="inferred",
             temporary=False, until=None):
    # 'asked' is trusted immediately because Lethanial said it out loud.
    # 'inferred' goes to the review queue, which is the same split the old
    # explicit and implicit tags encoded and the reason that queue exists.
    source = "explicit" if certainty == "asked" else "implicit"
    status = "active" if certainty == "asked" else "pending"

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
