"""Confirmation for writes that leave the Pi, over a channel with no buttons.

create_calendar_event does not create anything. It stages the event here, reads
the resolved time back, and ends the turn with a question. The write happens
only when confirm_pending_action is called on the turn after.

Why this is code and not a prompt instruction: "always ask first" in a prompt
is a suggestion to a probabilistic system. Here the model cannot skip the
question, because confirming on the same turn as the proposal is refused. A
human turn has to happen in between.

Why confirm takes no event details: it runs exactly what was read back. If it
accepted a title and a time, the model could confirm something other than what
Lethanial agreed to.

Deciding whether "yeah, do it" is a yes is judgment, and that stays with the
model. What the code guarantees is narrower and absolute: nothing runs unless
he was asked, the answer came on the very next turn, and it came soon.

State is per process. A proposal made by voice is confirmed by voice, and one
made in the app is confirmed in the app, which is the right boundary anyway.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from tools import Permission, tool

# Long enough to answer a question, short enough that a "yes" to something
# else minutes later cannot reach back and approve a stale proposal.
CONFIRM_WINDOW_S = 120


@dataclass
class _Pending:
    description: str
    run: Callable[[], Any]
    turn: int
    created: float


_lock = threading.Lock()
_turn = 0
_pending = None


def begin_turn():
    """Called once at the top of every Claude turn. A proposal records the turn
    it was made on, and confirmation is accepted only on the one after it."""
    global _turn
    with _lock:
        _turn += 1


def propose(description, run, now=None):
    """Stage an action. A new proposal replaces an old one, so "actually make it
    eleven" reads the corrected event back instead of confirming the first."""
    global _pending
    with _lock:
        _pending = _Pending(description, run, _turn,
                            time.monotonic() if now is None else now)


def resolve(approved, now=None):
    global _pending
    now = time.monotonic() if now is None else now

    with _lock:
        pending = _pending
        if pending is None:
            return "Nothing is waiting for confirmation, so nothing was done."
        if pending.turn == _turn:
            # Kept, not cleared: he can still answer the question on his turn.
            return ("Refused: Lethanial has not answered yet. Ask him and end "
                    "your turn. Nothing was done.")
        # Cleared either way from here, so one answer can never be used twice.
        _pending = None
        stale = pending.turn != _turn - 1 or now - pending.created > CONFIRM_WINDOW_S

    if stale:
        return ("That request expired before he confirmed it, so nothing was "
                "done. If he still wants it, propose it again.")
    if not approved:
        return f"Cancelled. Nothing was done: {pending.description}."
    # Outside the lock: this is a network call, and holding the lock through it
    # would stall the next turn's begin_turn behind Google.
    return pending.run()


@tool(
    name="confirm_pending_action",
    description=(
        "Carry out or cancel the action you just read back to Lethanial and "
        "asked him to confirm, such as a calendar event. Call this on his reply "
        "to that question: approved true if he agreed, false if he declined. "
        "If he asks for a change instead, do not call this; propose the changed "
        "version so he hears it again. It takes no details because it runs "
        "exactly what he heard. Say what happened in one short sentence."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "True only if he clearly agreed.",
            },
        },
        "required": ["approved"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
)
def confirm_pending_action(approved):
    return resolve(approved)
