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

Live turns bind a presented proposal to its conversation and check explicit
user assent before consuming it.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from tools import Permission, permits, tool

# Long enough to answer a question, short enough that a "yes" to something
# else minutes later cannot reach back and approve a stale proposal.
CONFIRM_WINDOW_S = 120


@dataclass
class _Pending:
    changes: list          # [(description, run)], confirmed or cancelled together
    turn: int
    created: float
    tier: str | None = None
    specs: list = field(default_factory=list)
    presented: bool = False
    source_cancelled: threading.Event | None = None
    delivered: threading.Event | None = None


_lock = threading.Lock()
_turn = 0
_pending = None
_announcements = []   # (turn, text) said by code this turn


@dataclass
class _Conversation:
    index: int = 0
    proposal: _Pending | None = None
    last_active: float = 0
    active: _Turn | None = None


@dataclass
class _Turn:
    conversation: _Conversation
    index: int
    tier: str
    utterance: str
    cancelled: threading.Event
    delivery: threading.Event | None
    announcements: list = field(default_factory=list)


_conversations = {}
_turn_context = ContextVar("pending_conversation_turn", default=None)
_tool_context = ContextVar("pending_proposal_tool", default=None)
_YES = {"yes", "yes please", "yes go ahead", "yes do it", "go ahead", "do it", "yeah", "yeah do it", "i confirm"}
_NO = {"no", "no thanks", "no cancel", "cancel", "cancel it", "do not do it", "don't do it"}


@contextmanager
def committed_turn(scope, tier, utterance, cancelled, delivery=None):
    now = time.monotonic()
    with _lock:
        for key, state in list(_conversations.items()):
            if state.active is None and now - state.last_active > CONFIRM_WINDOW_S:
                del _conversations[key]
        state = _conversations.setdefault(scope, _Conversation())
        if state.active is not None:
            state.active.cancelled.set()
        state.index += 1
        state.last_active = now
        context = _Turn(state, state.index, tier, utterance, cancelled, delivery)
        state.active = context
    token = _turn_context.set(context)
    try:
        yield
    finally:
        with _lock:
            if state.active is context:
                state.active = None
        _turn_context.reset(token)


@contextmanager
def proposing_tool(spec):
    token = _tool_context.set(spec)
    try:
        yield
    finally:
        _tool_context.reset(token)


def _reply_words(text):
    return re.sub(r"[^\w']+", " ", text.casefold()).strip()


def _matches(context, proposal, approved, now):
    return (not context.cancelled.is_set()
            and context.index == context.conversation.index
            and proposal is not None and proposal.presented
            and (proposal.source_cancelled is None or not proposal.source_cancelled.is_set())
            and (proposal.delivered is None or proposal.delivered.is_set())
            and proposal.turn == context.index - 1
            and 0 <= now - proposal.created <= CONFIRM_WINDOW_S
            and _reply_words(context.utterance) in (_YES if approved else _NO)
            and context.tier == proposal.tier
            and all(permits(spec, context.tier) for spec in proposal.specs))


def confirmation_allowed(approved):
    context = _turn_context.get()
    if context is None:
        return False
    with _lock:
        return _matches(context, context.conversation.proposal, approved, time.monotonic())


def mark_presented(question):
    context = _turn_context.get()
    if context is None:
        return
    with _lock:
        proposal = context.conversation.proposal
        if (not context.cancelled.is_set() and context.index == context.conversation.index
                and proposal is not None and proposal.turn == context.index
                and question == _question(proposal.changes)):
            proposal.presented = True


def begin_turn():
    """Called once at the top of every Claude turn. A proposal records the turn
    it was made on, and confirmation is accepted only on the one after it."""
    global _turn
    with _lock:
        _turn += 1
        _announcements.clear()


def _question(changes):
    """One question covering every staged change, in the order they were made."""
    parts = [description for description, _ in changes]
    if len(parts) == 1:
        return parts[0] + "?"
    rest = [p[0].lower() + p[1:] for p in parts[1:]]
    return ", ".join([parts[0]] + rest[:-1]) + ", and " + rest[-1] + "?"


def propose(description, run, now=None):
    """Stage a change and return the question that now covers everything staged.

    Several proposals on the same turn join one batch, confirmed or cancelled
    together. They used to replace each other: on Sep 13 2026 Nova staged seven
    tutoring lessons in one turn, asked about the first, and his yes created the
    seventh, a lesson he had never agreed to.

    A proposal on a later turn still replaces the old one, so "actually make it
    eleven" is asked again instead of confirming the first version."""
    global _pending
    now = time.monotonic() if now is None else now
    context = _turn_context.get()
    if context is not None:
        spec = _tool_context.get()
        with _lock:
            state = context.conversation
            if (context.index != state.index or context.cancelled.is_set()
                    or spec is None or not permits(spec, context.tier)):
                return "Nothing was proposed."
            proposal = state.proposal
            if proposal is None or proposal.turn != context.index:
                proposal = state.proposal = _Pending(
                    [], context.index, now, context.tier,
                    source_cancelled=context.cancelled, delivered=context.delivery)
            proposal.changes.append((description, run))
            proposal.specs.append(spec)
            proposal.presented = False
            return _question(proposal.changes)
    with _lock:
        if _pending is not None and _pending.turn == _turn:
            _pending.changes.append((description, run))
        else:
            _pending = _Pending([(description, run)], _turn, now)
        return _question(_pending.changes)


def announce(text):
    """Have code say text word for word at the end of this turn.

    For changes made at once, like an event added without a question since Sep
    14 2026. Said by code for the same reason a staged question is: what he
    hears is exactly what happened, with no model turn in between to reword it."""
    context = _turn_context.get()
    if context is not None:
        context.announcements.append(text)
        return
    with _lock:
        _announcements.append((_turn, text))


def current_turn():
    with _lock:
        return _turn


def words_for_turn():
    """Everything code must say this turn: announcements, then any staged
    question. None when there is nothing, which leaves the reply to the model."""
    context = _turn_context.get()
    if context is not None:
        staged = staged_question()
        words = context.announcements + ([staged] if staged else [])
        return " ".join(words) if words else None
    with _lock:
        said = [text for turn, text in _announcements if turn == _turn]
        if _pending is not None and _pending.turn == _turn:
            said.append(_question(_pending.changes))
    return " ".join(said) if said else None


def staged_question():
    """The question covering what was proposed on this turn, or None.

    brain.py speaks this word for word instead of letting the model phrase it.
    On Sep 14 2026 a proposal was staged for the 21st while Nova said "Monday
    September 14", and his yes created the event he never heard. Only a
    proposal from this turn counts: on the next turn the question has already
    been asked, and that turn is his answer."""
    context = _turn_context.get()
    if context is not None:
        with _lock:
            proposal = context.conversation.proposal
            if proposal is not None and proposal.turn == context.index == context.conversation.index:
                return _question(proposal.changes)
            return None
    with _lock:
        if _pending is None or _pending.turn != _turn:
            return None
        return _question(_pending.changes)


def resolve(approved, now=None):
    global _pending
    now = time.monotonic() if now is None else now

    context = _turn_context.get()
    if context is not None:
        with _lock:
            proposal = context.conversation.proposal
            if not _matches(context, proposal, approved, now):
                return "Nothing was changed. There is no matching confirmation to apply."
            context.conversation.proposal = None
        if not approved:
            return "Cancelled. Nothing was changed."
        outcomes = []
        for description, run in proposal.changes:
            if context.cancelled.is_set():
                outcomes.append("Remaining changes were cancelled before execution.")
                break
            try:
                outcomes.append(str(run()))
            except Exception:
                outcomes.append(f"Could not confirm the result of: {description}.")
        return " ".join(outcomes)
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
        return f"Cancelled. Nothing was done: {_question(pending.changes).rstrip('?')}."
    # Outside the lock: these are network calls, and holding the lock through
    # them would stall the next turn's begin_turn behind Google.
    if len(pending.changes) == 1:
        return pending.changes[0][1]()
    # In a batch one failure must not hide the rest, in either direction: what
    # was done and what was not are both reported, so Nova never says "Done"
    # over a half finished batch.
    results = []
    for description, run in pending.changes:
        try:
            results.append(str(run()))
        except Exception as exc:
            results.append(f"Failed, not done: {description} ({exc}).")
    return " ".join(results)


@tool(
    name="confirm_pending_action",
    description=(
        "Carry out or cancel the change he was just asked to confirm, such as "
        "moving, renaming or deleting a calendar event. Call this on his reply "
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
