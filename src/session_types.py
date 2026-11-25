"""The session type registry: what each kind of session is.

A session is a chat with one job, kept apart from Nova's main conversation: the
nightly review today, a plain chat thread and learn sessions later. Each type
brings its own purpose, system prompt, tools and model, and optionally a
schedule and something to save when it closes. The engine in sessions.py runs
any of them the same way, so a new kind of session is a registration here and
never a change to the engine or the schema.

Each type owns its own tools.ToolRegistry. Nova's main registry never sees
session tools, so a voice turn cannot reach close_session or schedule_block,
and her capability block stays exactly what it was.

Nothing here talks to the API or the database. It is a registry, like tools.py.
"""

import contextvars
import datetime
import re
from dataclasses import dataclass, field
from typing import Callable

from config import MODEL_A, SESSION_MODELS
from tools import ToolNotice, ToolRegistry


@dataclass(frozen=True)
class Schedule:
    """When a scheduled type opens, nudges and settles, as local clock times
    on its 4am day. settles_at before opens_at means the next morning: the
    nightly review opens at 21:30 and settles at 03:00 the day after."""
    opens_at: datetime.time
    remind_at: datetime.time
    settles_at: datetime.time


@dataclass(frozen=True)
class SessionType:
    name: str
    # One line, served as `purpose` on every session object.
    purpose: str
    # Called with the session row on every turn. A callable rather than a
    # string so it can carry live data (last night's summary); it should depend
    # only on the session, not the clock, or the cached prefix changes per turn.
    system_prompt: Callable[[dict], str]
    tools: ToolRegistry
    # Called with the time it is created; returns the first assistant message,
    # built in code from data, or None to open with no message. It may return
    # (message, plan) instead, where plan is a dict saved with the session
    # (sessions.plan) so the type can act on exactly what he was shown.
    opening: Callable[[datetime.datetime], str | tuple | None] | None = None
    # True for a type whose tools post code authored messages (post_message).
    # The engine then holds the model's text until the round's tools have run,
    # because whether it may be shown depends on whether code posted.
    posts_messages: bool = False
    # Called with the session row when it closes; returns the summary to save.
    # None means closing saves nothing, which is what a plain chat will want.
    on_close: Callable[[dict], str | None] | None = None
    # None for a type that can be started any number of times a day, which
    # also leaves for_day NULL. Set only for once a day types.
    schedule: Schedule | None = None
    # The push the timer sends when it opens a scheduled session, and the
    # reminder. push_body takes the time so it can say how many tasks there are.
    push_title: str | None = None
    push_body: Callable[[datetime.datetime], str] | None = None
    reminder_body: str | None = None
    # What the app shows while each tool runs, sent as the stage event.
    stages: dict = field(default_factory=dict)

    @property
    def model(self) -> str:
        """Read from config each time rather than fixed at registration, so a
        test or a config change is seen without re registering the type."""
        return SESSION_MODELS.get(self.name, MODEL_A)

    def stage_for(self, tool_name: str) -> str:
        return self.stages.get(tool_name, "Working on it")


_NAME_RULE = re.compile(r"^[a-z][a-z0-9_]*$")
_TYPES: dict[str, SessionType] = {}


def register_type(session_type: SessionType) -> SessionType:
    """Add a type. Fails at import time for the mistakes that would otherwise
    show up as a session behaving strangely at ten at night."""
    name = session_type.name
    if not _NAME_RULE.match(name or ""):
        raise ValueError(f"session type name {name!r} must be lowercase and underscores")
    if name in _TYPES:
        raise ValueError(f"session type {name!r} is already registered")
    if not session_type.purpose.strip():
        raise ValueError(f"session type {name!r} needs a purpose")
    if session_type.schedule is not None and not (
            session_type.push_title and session_type.push_body and session_type.reminder_body):
        # The timer would open it and have nothing to tell him.
        raise ValueError(f"scheduled session type {name!r} needs its push words")
    _TYPES[name] = session_type
    return session_type


def get_type(name: str) -> SessionType | None:
    return _TYPES.get(name)


def all_types() -> list[SessionType]:
    return [_TYPES[name] for name in sorted(_TYPES)]


# ── the turn in progress ──
# The engine sets this around each turn's tool calls, so a tool can find the
# session it is running in without the engine passing it through every tool's
# arguments, and without session tools importing sessions.py (which imports
# them). A contextvar rather than a global, because the server runs turns for
# different sessions on different threads at the same time.

@dataclass
class TurnState:
    session_id: int
    close_requested: bool = False
    # Messages tools posted this turn, written by code. The engine stores and
    # streams them after the model's own text.
    posted: list = field(default_factory=list)


_turn: contextvars.ContextVar[TurnState | None] = contextvars.ContextVar(
    "session_turn", default=None)


def begin_turn(session_id: int):
    """Returns the token end_turn needs, and the state the engine reads after."""
    state = TurnState(session_id)
    return _turn.set(state), state


def end_turn(token) -> None:
    _turn.reset(token)


def current_session_id() -> int | None:
    state = _turn.get()
    return state.session_id if state else None


def request_close() -> None:
    """Ask the engine to close this session once the turn's tools are done.

    A request, not the close itself: closing writes the summary with a model
    call, and doing that from inside a tool would run it under the turn's own
    lock, halfway through the turn."""
    state = _turn.get()
    if state is None:
        raise ToolNotice("There is no session running to close.")
    state.close_requested = True


def post_message(text: str) -> bool:
    """Post a message written by code, not the model, into the turn in
    progress. A time shown to him must come from code reading the calendar, so
    the plan and the week are built by code and posted here; the engine then
    holds the model's own words to one line with no clock times (see
    sessions.guard_text). False when no turn is running (a tool called on its
    own, as a test does)."""
    state = _turn.get()
    if state is None or not text.strip():
        return False
    state.posted.append(text.strip())
    return True
