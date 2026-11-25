"""The session engine against a fake Anthropic stream.

Nothing here reaches Claude, Google or Apple. Each test registers a throwaway
session type, or uses nightly_review, on a throwaway database."""

import json
import sqlite3
from datetime import datetime

import pytest

import nightly_review
import session_types
import sessions
from conftest import text_block, tool_block
from session_types import SessionType, request_close
from tools import Permission, ToolNotice, ToolRegistry

NOW = datetime(2026, 11, 24, 21, 45)


def _type(monkeypatch, name="test_chat", on_close=None, opening=None, schedule=None, **extra):
    registry = ToolRegistry()

    @registry.register(name="lookup", description="Look something up.",
                       input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
                       permission=Permission.READ, returns_to_model=True, min_tier="hokage")
    def lookup(q=""):
        return {"answer": 42}

    @registry.register(name="full", description="Always refuses politely.",
                       input_schema={"type": "object", "properties": {}},
                       permission=Permission.WRITE, returns_to_model=True)
    def full():
        raise ToolNotice("Tomorrow already has 3 must dos.")

    @registry.register(name="close_session", description="Close it.",
                       input_schema={"type": "object", "properties": {}},
                       permission=Permission.CONTROL, returns_to_model=False)
    def close_it():
        request_close()
        return "closing"

    session_type = SessionType(name=name, purpose="A test chat", system_prompt=lambda s: "You test.",
                               tools=registry, opening=opening, on_close=on_close,
                               schedule=schedule, stages={"lookup": "Looking it up"}, **extra)
    monkeypatch.setitem(session_types._TYPES, name, session_type)
    return session_type


@pytest.fixture
def hokage(monkeypatch):
    monkeypatch.setattr(sessions, "effective_tier", lambda: "hokage")


def _recorder():
    events = []
    return events, lambda kind, text: events.append((kind, text))


def _rows(db, session_id):
    return db.session_messages(session_id)


def _tool_log(db):
    conn = sqlite3.connect(db.DB_PATH)
    try:
        return conn.execute("SELECT tool_name, is_error, is_notice, model FROM tool_call_log "
                            "ORDER BY id").fetchall()
    finally:
        conn.close()


# ── the tool loop ──

def test_a_tool_round_then_the_answer(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch)
    session, created = sessions.create("test_chat", NOW)
    assert created
    fake = fake_claude(
        [text_block("Let me check."), tool_block("lookup", {"q": "x"}, "toolu_1")],
        [text_block("It is 42.")])
    events, on_text = _recorder()

    reply = sessions.run_turn(session["id"], "what is it", on_text=on_text, now=NOW)

    assert reply == "It is 42."
    assert events == [
        ("delta", "Let me"), ("delta", " check."),
        ("stage", "Looking it up"),
        ("reset", ""),
        ("delta", "It i"), ("delta", "s 42."),
    ]
    # The second request carries the call and its result, attached to it.
    second = fake.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}]}
    assert second[-1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"answer": 42}'}]}
    assert fake.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in fake.calls[0]["tools"]] == ["close_session", "full", "lookup"]


def test_every_message_is_stored_whole(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    fake_claude([text_block("Let me check."), tool_block("lookup", {"q": "x"}, "toolu_1")],
                [text_block("It is 42.")])
    sessions.run_turn(session["id"], "what is it", now=NOW)

    rows = _rows(db, session["id"])
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", [{"type": "text", "text": "what is it"}]),
        ("assistant", [{"type": "text", "text": "Let me check."},
                       {"type": "tool_use", "id": "toolu_1", "name": "lookup",
                        "input": {"q": "x"}}]),
        ("user", [{"type": "tool_result", "tool_use_id": "toolu_1",
                   "content": '{"answer": 42}'}]),
        ("assistant", [{"type": "text", "text": "It is 42."}]),
    ]
    assistant = rows[1]
    assert assistant["model"] == "claude-haiku-4-5"
    assert (assistant["input_tokens"], assistant["output_tokens"],
            assistant["cache_read_tokens"], assistant["cache_write_tokens"]) == (120, 30, 100, 0)
    assert assistant["ms"] is not None
    assert rows[0]["model"] is None and rows[0]["input_tokens"] is None
    assert _tool_log(db) == [("lookup", 0, 0, "claude-haiku-4-5")]
    assert db.get_session(session["id"])["last_message_at"] == NOW.isoformat()


def test_the_app_sees_text_only(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch, opening=lambda now: "Hello.")
    session, _ = sessions.create("test_chat", NOW)
    fake_claude([tool_block("lookup", {}, "toolu_1")], [text_block("It is 42.")])
    sessions.run_turn(session["id"], "what is it", now=NOW)
    shown = sessions.view(session["id"], NOW)["messages"]
    assert [(m["role"], m["text"]) for m in shown] == [
        ("assistant", "Hello."), ("user", "what is it"), ("assistant", "It is 42.")]
    assert set(shown[0]) == {"id", "role", "text", "created_at"}


def test_an_opening_is_sent_behind_a_short_user_line(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch, opening=lambda now: "Here is today.")
    session, _ = sessions.create("test_chat", NOW)
    assert [(r["role"], r["content"]) for r in _rows(db, session["id"])] == [
        ("assistant", [{"type": "text", "text": "Here is today."}])]
    fake = fake_claude([text_block("Nice.")])
    sessions.run_turn(session["id"], "did it all", now=NOW)
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": sessions.OPENING_CUE}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Here is today."}]},
        {"role": "user", "content": [{"type": "text", "text": "did it all"}]},
    ]


def test_a_turn_after_tool_results_merges_into_one_user_message(db, monkeypatch, fake_claude, hokage):
    """A turn that ended on close_session leaves tool results last; a turn that
    failed leaves his message last. Either way the next request alternates."""
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    db.add_session_messages(session["id"], [{"role": "user", "content": [
        {"type": "text", "text": "first"}]}], NOW)
    fake = fake_claude([text_block("Here.")])
    sessions.run_turn(session["id"], "second", now=NOW)
    assert fake.calls[0]["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}]


def test_the_last_round_may_not_call_a_tool(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    rounds = [[tool_block("lookup", {}, f"toolu_{i}")] for i in range(sessions.MAX_ROUNDS - 1)]
    fake = fake_claude(*rounds, [text_block("Done looking.")])
    assert sessions.run_turn(session["id"], "go", now=NOW) == "Done looking."
    assert len(fake.calls) == sessions.MAX_ROUNDS
    assert "tool_choice" not in fake.calls[0]
    assert fake.calls[-1]["tool_choice"] == {"type": "none"}


# ── the gate ──

def test_a_tool_above_his_tier_is_refused_and_said(db, monkeypatch, fake_claude):
    monkeypatch.setattr(sessions, "effective_tier", lambda: "chunin")
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    fake = fake_claude([tool_block("lookup", {}, "toolu_1")], [text_block("I can't.")])
    assert sessions.run_turn(session["id"], "look", now=NOW) == "I can't."
    [result] = fake.calls[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    assert result["content"].startswith("Refused: lookup needs more clearance than chunin")
    assert _tool_log(db) == [("lookup", 1, 0, "claude-haiku-4-5")]


def test_a_notice_is_logged_as_a_notice(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    fake = fake_claude([tool_block("full", {}, "toolu_1")], [text_block("No room.")])
    sessions.run_turn(session["id"], "add one", now=NOW)
    [result] = fake.calls[1]["messages"][-1]["content"]
    assert result == {"type": "tool_result", "tool_use_id": "toolu_1",
                      "content": "Tomorrow already has 3 must dos.", "is_error": True}
    assert _tool_log(db) == [("full", 0, 1, "claude-haiku-4-5")]


def test_session_tools_never_reach_novas_registry():
    from tools import registry
    for name in nightly_review.registry.names():
        assert name not in registry


# ── closing ──

def test_close_during_a_turn_sends_closed_before_the_turn_ends(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch, on_close=lambda s: "Planned tomorrow.")
    session, _ = sessions.create("test_chat", NOW)
    fake = fake_claude([text_block("Good night."), tool_block("close_session", {}, "toolu_1")])
    events, on_text = _recorder()

    reply = sessions.run_turn(session["id"], "done", on_text=on_text, now=NOW)

    assert reply == "Good night."
    assert len(fake.calls) == 1        # close_session needs no second round
    assert events[-1] == ("closed", "Planned tomorrow.")
    assert ("reset", "") not in events
    closed = db.get_session(session["id"])
    assert (closed["status"], closed["summary"], closed["closed_at"]) == (
        "closed", "Planned tomorrow.", NOW.isoformat())


def test_a_closed_session_takes_no_turns(db, monkeypatch, fake_claude, hokage):
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    sessions.close(session["id"], NOW)
    with pytest.raises(sessions.NotOpen):
        sessions.run_turn(session["id"], "hello", now=NOW)
    with pytest.raises(sessions.NotOpen):
        sessions.close(session["id"], NOW)
    with pytest.raises(sessions.NotFound):
        sessions.run_turn(999, "hello", now=NOW)


def test_a_failing_summary_still_closes(db, monkeypatch, hokage):
    def broken(session):
        raise RuntimeError("network down")
    _type(monkeypatch, on_close=broken)
    session, _ = sessions.create("test_chat", NOW)
    closed = sessions.close(session["id"], NOW)
    assert (closed["status"], closed["summary"]) == ("closed", None)


# ── for_day and UNIQUE ──

def test_two_sessions_of_one_type_with_no_day_both_exist(db, monkeypatch):
    """The contract's rule: for_day NULL never collides, so a later chat type
    can have any number of sessions a day."""
    _type(monkeypatch)
    first, made_first = sessions.create("test_chat", NOW)
    second, made_second = sessions.create("test_chat", NOW)
    assert made_first and made_second
    assert first["id"] != second["id"]
    assert first["for_day"] is None and second["for_day"] is None
    assert len(db.list_sessions("all", types=["test_chat"])) == 2


def test_unique_blocks_a_second_nightly_review_for_one_day(db):
    first, made = sessions.create("nightly_review", NOW)
    again, made_again = sessions.create("nightly_review", datetime(2026, 11, 25, 1, 0))
    assert made and not made_again
    assert again["id"] == first["id"]
    assert first["for_day"] == "2026-11-24"
    # The table itself refuses it, so two timer runs racing still make one.
    assert db.create_session("nightly_review", "2026-11-24", "claude-haiku-4-5", None, NOW) is None
    assert len(db.list_sessions("all")) == 1


def test_an_unknown_type_is_refused(db):
    with pytest.raises(sessions.UnknownType):
        sessions.create("nonsense", NOW)


def test_opening_marks_it_opened_once(db, monkeypatch):
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    first = sessions.view(session["id"], datetime(2026, 11, 24, 21, 50))
    later = sessions.view(session["id"], datetime(2026, 11, 24, 22, 10))
    assert first["session"]["opened_at"] == "2026-11-24T21:50:00"
    assert later["session"]["opened_at"] == "2026-11-24T21:50:00"


def test_listing_is_newest_activity_first(db, monkeypatch):
    _type(monkeypatch)
    old, _ = sessions.create("test_chat", datetime(2026, 11, 24, 9, 0))
    new, _ = sessions.create("test_chat", datetime(2026, 11, 24, 10, 0))
    db.add_session_messages(old["id"], [{"role": "user", "content": [
        {"type": "text", "text": "hi"}]}], datetime(2026, 11, 24, 11, 0))
    assert [s["id"] for s in sessions.listing("open")] == [old["id"], new["id"]]
    assert sessions.listing("closed") == []


# ── the timer ──

def _pushes():
    sent = []

    def notify(title, body, kind="info", data=None, ttl_s=None, collapse_id=None, category=None):
        sent.append({"title": title, "body": body, "kind": kind, "data": data,
                     "collapse_id": collapse_id})
        return 1
    return sent, notify


def test_the_timer_through_a_night_he_never_opened(db):
    db.add_planned_tasks([{"text": "Lab report", "scheduled_for": "2026-11-24", "must_do": True}],
                         datetime(2026, 11, 24, 8, 0))
    sent, notify = _pushes()

    assert sessions.tick(datetime(2026, 11, 24, 21, 25), notify) == []
    did = sessions.tick(datetime(2026, 11, 24, 21, 30), notify)
    [session] = db.list_sessions("all")
    assert did == [(session["id"], "create"), (session["id"], "push")]
    assert sent == [{"title": "Nightly review",
                     "body": "1 task today. What got done, and what matters tomorrow?",
                     "kind": "session",
                     "data": {"session_id": session["id"], "session_type": "nightly_review",
                              "url": f"nova://session/{session['id']}"},
                     "collapse_id": f"session-{session['id']}"}]

    assert sessions.tick(datetime(2026, 11, 24, 21, 35), notify) == []
    assert sessions.tick(datetime(2026, 11, 24, 22, 30), notify) == [(session["id"], "remind")]
    assert sent[-1]["body"] == "Still open. Two minutes to plan tomorrow."
    assert sent[-1]["collapse_id"] == f"session-{session['id']}"
    assert sessions.tick(datetime(2026, 11, 25, 0, 30), notify) == []
    assert len(sent) == 2

    assert sessions.tick(datetime(2026, 11, 25, 3, 0), notify) == [(session["id"], "miss")]
    settled = db.get_session(session["id"])
    assert (settled["status"], settled["summary"]) == ("missed", None)
    assert len(db.list_sessions("all")) == 1


def test_the_timer_closes_one_he_wrote_in(db, fake_claude):
    fake = fake_claude(summary="He finished the lab and planned tomorrow.")
    sent, notify = _pushes()
    sessions.tick(datetime(2026, 11, 24, 21, 30), notify)
    [session] = db.list_sessions("all")
    db.add_session_messages(session["id"], [{"role": "user", "content": [
        {"type": "text", "text": "did the lab"}]}], datetime(2026, 11, 24, 21, 40))

    # A Pi that was off until 9am settles it on its first run.
    assert sessions.tick(datetime(2026, 11, 25, 9, 0), notify) == [(session["id"], "close")]
    settled = db.get_session(session["id"])
    assert settled["status"] == "closed"
    assert settled["summary"] == (
        "Today: 0 done, 0 skipped, 0 moved, 0 left open, of 0 tasks. "
        "Tomorrow: 0 tasks, 0 must dos, 0 blocks on the calendar.\n\n"
        "He finished the lab and planned tomorrow.")
    assert "did the lab" in fake.summary_calls[0]["messages"][0]["content"]


def test_a_push_nobody_accepted_is_tried_again(db):
    tries = []

    def unaccepted(*args, **kwargs):
        tries.append(1)
        return 0
    sessions.tick(datetime(2026, 11, 24, 21, 30), unaccepted)
    sessions.tick(datetime(2026, 11, 24, 21, 35), unaccepted)
    assert len(tries) == 2
    assert db.list_sessions("all")[0]["push_sent_at"] is None


def test_a_session_started_by_hand_gets_no_push(db):
    sent, notify = _pushes()
    sessions.create("nightly_review", datetime(2026, 11, 24, 20, 0))
    sessions.tick(datetime(2026, 11, 24, 21, 30), notify)
    sessions.tick(datetime(2026, 11, 24, 22, 30), notify)
    assert sent == []
    assert len(db.list_sessions("all")) == 1


# ── a reload reads the way the stream looked ──

def test_text_beside_a_replaced_tool_call_is_hidden_on_reload(db, monkeypatch, fake_claude, hokage):
    # Live, "Let me check." is streamed and then reset before "It is 42.", so a
    # reload must not show it as its own message either.
    _type(monkeypatch)
    session, _ = sessions.create("test_chat", NOW)
    fake_claude([text_block("Let me check."), tool_block("lookup", {"q": "x"}, "toolu_1")],
                [text_block("It is 42.")])
    events, on_text = _recorder()
    sessions.run_turn(session["id"], "what is it", on_text=on_text, now=NOW)
    assert ("reset", "") in events
    shown = sessions.view(session["id"], NOW)["messages"]
    assert [(m["role"], m["text"]) for m in shown] == [
        ("user", "what is it"), ("assistant", "It is 42.")]


def test_text_beside_a_closing_call_stays_on_reload(db, monkeypatch, fake_claude, hokage):
    # close_session does not return to the model, so no round follows and
    # nothing was reset live: the goodbye stays.
    _type(monkeypatch, on_close=lambda s: "Summary.")
    session, _ = sessions.create("test_chat", NOW)
    fake_claude([text_block("Good night."), tool_block("close_session", {}, "toolu_1")])
    events, on_text = _recorder()
    sessions.run_turn(session["id"], "done", on_text=on_text, now=NOW)
    assert ("reset", "") not in events
    shown = sessions.view(session["id"], NOW)["messages"]
    assert shown[-1]["text"] == "Good night."
    assert db.get_session(session["id"])["closed_by"] == "user"


def test_the_timer_closes_as_timer(db, monkeypatch, hokage):
    _type(monkeypatch, on_close=lambda s: "Summary.")
    session, _ = sessions.create("test_chat", NOW)
    sessions.close(session["id"], NOW, closed_by="timer")
    assert db.get_session(session["id"])["closed_by"] == "timer"
