"""The session engine: create, read, run a turn, close, and the timer's tick.

A session is a chat with one job (see session_types.py). This engine is apart
from brain.py on purpose. Nova's voice turn is tuned for a room: short replies,
history trimmed to thirty words a reply, a speaker, barge in. A session is the
opposite: written replies, a whole transcript kept intact with every tool call
and result, its own tools and model. Sharing ask_nova would mean a branch on
every one of those, so the engine is small and separate, and nothing here
imports brain.py or the voice path.

A turn is a streamed tool loop. Each round streams one model call, sends its
text to the caller as it is written, runs any tools it asked for through the
same permission gate brain.py uses, and goes round again while a tool's result
needs reading back. Every message is stored whole, as the API's content blocks,
so the next turn sends the model exactly what happened, not a summary of it.

The events a caller receives through on_text(kind, text) are the ones
/chat/stream sends, so the app reuses its stream reader: delta, stage, reset,
closed. The server adds done or error at the end.
"""

import json
import threading
import time
from datetime import datetime

import anthropic

import database
import push
import session_schedule
from database import effective_tier, log_tool_call, task_day
from session_types import begin_turn, end_turn, get_type, all_types
from tools import ToolNotice, permits

# Rounds a single turn may take. The review reads today, updates several
# statuses, reads the week, creates tasks and schedules blocks, and the model
# may spread that over a few rounds; six covers it with room. The last round is
# sent with tool_choice none, so a turn always ends in words.
MAX_ROUNDS = 6
MAX_TOKENS = 1500

# The API wants a conversation to start with the user. A session opens with
# Nova's message, built in code, so that message is sent behind this line. It
# is never stored: the transcript shows what was said, and he did not say it.
OPENING_CUE = "[The session is open. Begin with your opening message.]"

# The words the close summary is asked for in. Counts are computed in code by
# the type's on_close; this only asks for what was said.
SUMMARY_MAX_TOKENS = 300


class SessionError(Exception):
    """A request the engine refuses. status is the HTTP code the server uses."""
    status = 400


class UnknownType(SessionError):
    status = 400


class NotFound(SessionError):
    status = 404


class NotOpen(SessionError):
    status = 409


_client_instance = None


def _client():
    """One synchronous client, made on first use. Synchronous because a turn
    runs on its own worker thread with nothing else to overlap, so an event
    loop would be machinery with no job. Tests replace this function."""
    global _client_instance
    if _client_instance is None:
        _client_instance = anthropic.Anthropic()
    return _client_instance


# One lock per session, so two turns (two taps of send, or a turn and a close)
# cannot interleave their messages. The guard only protects the dict itself.
_locks = {}
_locks_guard = threading.Lock()


def _lock(session_id):
    with _locks_guard:
        return _locks.setdefault(session_id, threading.Lock())


def _emit(on_text, kind, text):
    """Hand one event to a streaming caller. A failing callback is logged and
    swallowed: the client may have gone away, and the turn still has to finish
    storing what it did."""
    if on_text is None or (kind != "reset" and not text):
        return
    try:
        on_text(kind, text)
    except Exception as exc:
        print(f"Session stream callback failed: {exc}", flush=True)


# ── what the app sees ──

def public(session):
    """The contract's session object. purpose comes from the registry."""
    session_type = get_type(session["type"])
    return {
        "id": session["id"],
        "type": session["type"],
        "title": session["title"],
        # Never null, per the contract: a row whose type was later renamed
        # or removed still shows, labelled by its type name.
        "purpose": session_type.purpose if session_type else session["type"],
        "status": session["status"],
        "for_day": session["for_day"],
        "created_at": session["created_at"],
        "opened_at": session["opened_at"],
        "last_message_at": session["last_message_at"],
        "closed_at": session["closed_at"],
        "summary": session["summary"],
    }


def text_of(content):
    """The text blocks of a stored message, joined with a blank line."""
    parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n\n".join(part for part in parts if part.strip())


def public_messages(rows):
    """Messages as the app renders them. A tool call or a tool result alone has
    no text and is left out entirely."""
    out = []
    for row in rows:
        text = text_of(row["content"])
        if text:
            out.append({"id": row["id"], "role": row["role"], "text": text,
                        "created_at": row["created_at"]})
    return out


# ── create, read, list ──

def get(session_id):
    session = database.get_session(session_id)
    if session is None:
        raise NotFound("No session with that id")
    return session


def create(type_name, now=None):
    """Start a session of a type. Returns (session row, created).

    A once a day type gets its 4am day as for_day, and a second start that day
    returns the first one with created False; UNIQUE (type, for_day) is what
    decides, so two timer runs racing still make one. Any other type gets
    for_day NULL and a new session every time."""
    session_type = get_type(type_name)
    if session_type is None:
        raise UnknownType(f"No session type named {type_name!r}")
    now = now or datetime.now()
    for_day = task_day(now) if session_type.schedule is not None else None

    opening = session_type.opening(now) if session_type.opening else None
    blocks = [{"type": "text", "text": opening}] if opening else None
    session_id = database.create_session(type_name, for_day, session_type.model, blocks, now)
    if session_id is None:
        return database.session_for_day(type_name, for_day), False
    return database.get_session(session_id), True


def body(session):
    """{"session": ..., "messages": [...]}, the shape GET and POST share."""
    return {"session": public(session),
            "messages": public_messages(database.session_messages(session["id"]))}


def view(session_id, now=None):
    """The contract's GET /sessions/{id} body. Marks it opened the first time,
    before reading, so the session returned already shows it."""
    get(session_id)
    database.mark_session_opened(session_id, now)
    return body(get(session_id))


def listing(status="open"):
    return [public(s) for s in database.list_sessions(status)]


# ── a turn ──

def _block_dict(block):
    """A content block from the API as plain JSON, for storage and for sending
    back. Text and tool calls are written out by hand so nothing the API added
    for its own bookkeeping (a null citations field) is sent back to it."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "input": dict(block.input)}
    return block.model_dump(exclude_none=True)


def _api_messages(rows):
    """Stored rows as the API's messages list.

    An opening assistant message is led by OPENING_CUE. Consecutive rows of one
    role are merged into one message: a turn that ended on tool results is
    followed by his next message, and a turn whose stream failed leaves his
    message with no reply, and either way the next request is still well formed."""
    messages = []
    if rows and rows[0]["role"] == "assistant":
        messages.append({"role": "user", "content": [{"type": "text", "text": OPENING_CUE}]})
    for row in rows:
        if messages and messages[-1]["role"] == row["role"]:
            messages[-1] = {"role": row["role"],
                            "content": messages[-1]["content"] + row["content"]}
        else:
            messages.append({"role": row["role"], "content": list(row["content"])})
    return messages


def _result_text(output):
    if isinstance(output, str):
        return output
    return json.dumps(output, default=str)


def _run_tools(session_type, tool_uses, model, tier):
    """Run every tool the model called, in order, each through the gate.

    The same shape as brain._run_tools, against the type's own registry: a
    refusal or a failure is a result the model reads and speaks, never an
    exception that ends the turn. A ToolNotice is an expected outcome (the
    must do cap, a slot that overlaps) and is logged as a notice, not an error."""
    registry = session_type.tools
    results = []
    for block in tool_uses:
        started = time.monotonic()
        notice = False
        try:
            spec = registry.get(block.name)
            if not permits(spec, tier):
                output = (f"Refused: {block.name} needs more clearance than "
                          f"{tier}. Say you are not allowed to do that.")
                is_error = True
            else:
                output = registry.call(block.name, dict(block.input))
                is_error = False
        except ToolNotice as exc:
            output = str(exc)
            is_error = True
            notice = True
        except Exception as exc:
            output = f"{block.name} failed: {exc}"
            is_error = True

        elapsed = (time.monotonic() - started) * 1000.0
        log_tool_call(block.name, dict(block.input), output,
                      is_error=is_error and not notice, is_notice=notice,
                      duration_ms=elapsed, model=model)
        results.append({"block": block, "output": _result_text(output), "is_error": is_error})
    return results


def _result_blocks(results):
    # is_error only when true: the API reads its presence as the flag.
    return [{"type": "tool_result", "tool_use_id": r["block"].id, "content": r["output"],
             **({"is_error": True} if r["is_error"] else {})} for r in results]


def _needs_another_round(session_type, results):
    """Whether the model has to read what the tools returned. A failure always
    does; otherwise the tool decides, as in brain.py. close_session does not,
    so a goodbye written beside it is the last word."""
    for r in results:
        if r["is_error"]:
            return True
        if r["block"].name in session_type.tools and \
                session_type.tools.get(r["block"].name).returns_to_model:
            return True
    return False


def _usage(final):
    usage = getattr(final, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", None),
    }


def _stream_round(session_type, kwargs, on_text):
    """One streamed model call. Returns (the text written, the final message)."""
    text = ""
    with _client().messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    _emit(on_text, "stage", session_type.stage_for(block.name))
                elif block.type == "text" and text and not text[-1].isspace():
                    # Two text blocks carry no space between them.
                    text += "\n\n"
                    _emit(on_text, "delta", "\n\n")
                continue
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                text += event.delta.text
                _emit(on_text, "delta", event.delta.text)
        final = stream.get_final_message()
    return text, final


def run_turn(session_id, message, on_text=None, now=None):
    """Run one turn of a session and return the reply's text.

    Raises NotFound or NotOpen before anything is stored. The tier is read once,
    here, and every tool call in the turn is gated on it."""
    with _lock(session_id):
        session = get(session_id)
        if session["status"] != "open":
            raise NotOpen("That session is not open")
        session_type = get_type(session["type"])
        if session_type is None:
            raise SessionError(f"Session type {session['type']!r} is not registered")

        tier = effective_tier()
        model = session["model"] or session_type.model
        database.add_session_messages(
            session_id, [{"role": "user", "content": [{"type": "text", "text": message}]}], now)
        messages = _api_messages(database.session_messages(session_id))
        # Cached: the prompt depends on the session, not the clock, so every
        # turn after the first reads it (and the tools before it) from cache.
        system = [{"type": "text", "text": session_type.system_prompt(session),
                   "cache_control": {"type": "ephemeral"}}]
        tools = session_type.tools.api_schemas()

        token, state = begin_turn(session_id)
        reply = ""
        try:
            for round_index in range(MAX_ROUNDS):
                kwargs = dict(model=model, max_tokens=MAX_TOKENS, system=system,
                              messages=messages, tools=tools)
                if round_index == MAX_ROUNDS - 1:
                    # tools stay in the request, because the history holds tool
                    # calls, but none may be made: this round has to answer.
                    kwargs["tool_choice"] = {"type": "none"}
                started = time.monotonic()
                round_text, final = _stream_round(session_type, kwargs, on_text)
                blocks = [b for b in (_block_dict(b) for b in final.content)
                          if not (b["type"] == "text" and not b["text"].strip())]
                assistant = {"role": "assistant", "content": blocks, "model": model,
                             "ms": int((time.monotonic() - started) * 1000), **_usage(final)}
                reply = round_text
                tool_uses = [b for b in final.content if b.type == "tool_use"]

                if not tool_uses:
                    if blocks:
                        database.add_session_messages(session_id, [assistant], now)
                    break

                results = _run_tools(session_type, tool_uses, model, tier)
                result_row = {"role": "user", "content": _result_blocks(results)}
                database.add_session_messages(session_id, [assistant, result_row], now)
                messages = messages + [{"role": "assistant", "content": blocks},
                                       {"role": "user", "content": result_row["content"]}]
                if not _needs_another_round(session_type, results):
                    break
                if round_text:
                    # What was written beside the tool call is replaced by the
                    # answer the next round writes from its result.
                    _emit(on_text, "reset", "")
        finally:
            end_turn(token)

        if state.close_requested:
            # The timer runs in its own process, so this lock does not cover
            # it. If it settled the session during this turn, the reply is
            # already stored; say closed with what it saved, not an error.
            try:
                closed = _close_locked(session_id, now)
            except NotOpen:
                closed = get(session_id)
            _emit(on_text, "closed", closed["summary"] or "Closed.")
        return reply.strip()


# ── closing ──

def transcript(session_id):
    """The conversation as plain lines, text only, for the summary call."""
    lines = []
    for row in database.session_messages(session_id):
        text = text_of(row["content"])
        if text:
            speaker = "Nova" if row["role"] == "assistant" else "Lethanial"
            lines.append(f"{speaker}: {text}")
    return "\n\n".join(lines)


def summarize(model, instructions, text):
    """One plain model call, no tools, no stream. For a type's on_close."""
    response = _client().messages.create(
        model=model, max_tokens=SUMMARY_MAX_TOKENS, system=instructions,
        messages=[{"role": "user", "content": text}])
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _close_locked(session_id, now=None):
    session = get(session_id)
    if session["status"] != "open":
        raise NotOpen("That session is not open")
    session_type = get_type(session["type"])
    summary = None
    if session_type is not None and session_type.on_close is not None:
        try:
            summary = session_type.on_close(session)
        except Exception as exc:
            # A close at 3am with the network down still has to close; a
            # session left open forever is worse than one with no summary.
            print(f"Session {session_id} summary failed: {exc}", flush=True)
    if not database.finish_session(session_id, "closed", summary, now):
        raise NotOpen("That session is not open")
    return get(session_id)


def close(session_id, now=None):
    """Close an open session now and write its summary. Returns the row."""
    with _lock(session_id):
        return _close_locked(session_id, now)


def miss(session_id, now=None):
    with _lock(session_id):
        return database.finish_session(session_id, "missed", None, now)


# ── the timer ──

def _push_data(session):
    return {"session_id": session["id"], "session_type": session["type"],
            "url": f"nova://session/{session['id']}"}


def tick(now=None, notify=None):
    """One run of miles-sessions.timer: create tonight's sessions, push, remind
    and settle, each as session_schedule decides. Returns what it did, as
    (session id, action) pairs, for the journal.

    A push is marked sent only when Apple accepted it for at least one phone,
    so one that could not go is tried again on the next run until midnight."""
    now = now or datetime.now()
    notify = notify or push.notify
    done = []
    for session_type in all_types():
        schedule = session_type.schedule
        if schedule is None:
            continue
        exists = database.session_for_day(session_type.name, task_day(now)) is not None
        if session_schedule.should_create(now, schedule, exists):
            session, created = create(session_type.name, now)
            if created:
                done.append((session["id"], "create"))

        for session in database.list_sessions("open", types=[session_type.name]):
            if session["for_day"] is None:
                continue
            wrote = database.session_has_user_message(session["id"])
            action = session_schedule.decide(now, schedule, session, wrote)
            if action in (session_schedule.PUSH, session_schedule.REMIND):
                body = (session_type.push_body(now) if action == session_schedule.PUSH
                        else session_type.reminder_body)
                sent = notify(session_type.push_title, body, kind="session",
                              data=_push_data(session), collapse_id=f"session-{session['id']}")
                if sent:
                    database.mark_session_pushed(
                        session["id"], "push" if action == session_schedule.PUSH else "reminder", now)
                    done.append((session["id"], action))
            elif action == session_schedule.CLOSE:
                try:
                    close(session["id"], now)
                    done.append((session["id"], action))
                except NotOpen:
                    pass
            elif action == session_schedule.MISS:
                if miss(session["id"], now):
                    done.append((session["id"], action))
    return done
