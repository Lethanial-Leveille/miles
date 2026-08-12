"""Demotion by voice.

Two uses in one tool. Lowering his own access is temporary and exists so the
guest boundary can be tested without a second person in the room. Lowering
someone else's is a real decision about their clearance.

Demotion only, in both cases. Removing capability is safe over a channel that
can be replayed from a recording; granting it is not, so escalation lives at a
keyboard and always will.

The caller must currently be hokage. Once he has demoted himself for testing he
has no authority either, which is correct: the boundary is not worth much if
the person behind it can still reach through.
"""

from database import (TIERS, effective_tier, find_person, lower_person_tier,
                      set_tier_override)
from tools import Permission, tool


@tool(
    name="lower_access",
    description=(
        "Lower someone's clearance. Ranks are genin, chunin, jonin, hokage. "
        "Call this when Lethanial asks to demote someone, or to drop his own "
        "access to see how you behave with someone who is not him. "
        "With no person named it lowers his own, temporarily, until he clears "
        "it at his keyboard. With a person named it lowers theirs for real. "
        "This can only lower. You cannot raise anyone's access and must not "
        "offer to, because a recording of his voice must not be able to hand "
        "itself the keys. "
        "Say which rank is now in force. Do not read the rank list back."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "enum": ["genin", "chunin", "jonin"],
                "description": "The rank to drop to. Must be below their current one.",
            },
            "person": {
                "type": "string",
                "description": "Who to demote, as Lethanial said it. Omit for himself.",
            },
        },
        "required": ["tier"],
    },
    permission=Permission.WRITE,
    returns_to_model=True,
)
def lower_access(tier, person=None):
    if effective_tier() != "hokage":
        return ("Refused. Only Lethanial at full access can change clearance, "
                "and that is not who is speaking.")

    if not person:
        result = set_tier_override(tier)
        if result is None:
            return f"Refused: {tier} is not below his current access."
        return (f"His access is now {result}, until he clears it with "
                f"scripts/people.py restore.")

    match = find_person(person)
    if match is None:
        return (f"No single person matches {person!r}. Ask him which one he "
                f"means rather than guessing.")
    result = lower_person_tier(match["id"], tier)
    if result is None:
        return (f"Refused: {match['full_name']} is already at or below {tier}. "
                f"This can only lower.")
    return f"{match['preferred_name'] or match['full_name']} is now {result}."
