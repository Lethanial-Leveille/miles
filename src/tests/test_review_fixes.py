"""The fix round after the independent review of review v2. One test or more
per finding, each written to fail on the code before the fix."""

import datetime
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime as dt

import pytest

import calendar_roles as roles
import calendar_tools
import fake_google
import nightly_review as nr
import review_plan as rp
import session_types
import sessions
from conftest import text_block, tool_block
from fake_google import ACM, HOLIDAYS, MILES, PRIMARY
from test_calendar_roles import ev
from test_review_v2 import (NOW, TODAY, TOMORROW, _backlog, _in_turn, _session, _task, at,
                            fixed_night, world)  # noqa: F401  (world is a fixture)
from tools import ToolNotice


def _must_dos(db):
    return [t for t in db.tasks_between(TOMORROW, TOMORROW) if t["must_do"]]


# ── 1. confirming twice books once ──

def test_two_confirms_in_one_round_make_one_task_and_one_event_each(db, world, fake_claude):
    fixed_night(db, world)
    session = _session()
    fake_claude([text_block("Good night."), tool_block("confirm_plan", {}, "t1"),
                 tool_block("confirm_plan", {}, "t2")],
                [text_block("Booked.")])
    sessions.run_turn(session["id"], "good", now=NOW)
    on_day = db.tasks_between(TOMORROW, TOMORROW)
    assert sorted(t["text"] for t in on_day) == sorted([
        "Study for linear algebra test", "Fix Friday: Test Prep with Alex overlaps Dentist",
        "Email professor", "Order NVMe HAT and SSD"])
    assert len([c for c, _ in world.inserts if c == MILES]) == 3


def test_a_second_confirm_is_refused_and_changes_nothing(db, world):
    fixed_night(db, world)
    session = _session()
    token, state = _in_turn(session["id"])
    try:
        nr.confirm_plan(now=NOW)
        before = (len(db.tasks_between(TOMORROW, TOMORROW)), len(world.inserts))
        with pytest.raises(ToolNotice, match="already confirmed"):
            nr.confirm_plan(now=NOW)
    finally:
        session_types.end_turn(token)
    assert (len(db.tasks_between(TOMORROW, TOMORROW)), len(world.inserts)) == before
    plan = db.session_plan(session["id"])
    assert plan["confirmed"] is True
    assert all(m["task_id"] for m in plan["must_dos"] + plan["optional"])   # ids written back


def test_a_google_error_while_booking_leaves_the_ids_saved_and_the_rest_booked(
        db, world, monkeypatch):
    fixed_night(db, world)
    session = _session()
    study = db.session_plan(session["id"])["must_dos"][0]["task_id"]
    db.set_task_calendar_event(study, "someblock")
    calls = {"n": 0}

    def flaky(event_id):
        # The first lookup (placing) works; the one in the booking loop fails.
        calls["n"] += 1
        if calls["n"] == 1:
            return {"id": "someblock", "title": "Study", "start": at(25, 10),
                    "end": at(25, 12), "all_day": False}
        raise RuntimeError("Google is down")
    monkeypatch.setattr(calendar_tools, "get_miles_event", flaky)
    token, state = _in_turn(session["id"])
    try:
        said = nr.confirm_plan(now=NOW)            # does not raise
    finally:
        session_types.end_turn(token)
    assert "Could not book" in said and "Google is down" in said
    plan = db.session_plan(session["id"])
    assert plan["confirmed"] and all(m["task_id"] for m in plan["must_dos"])
    assert len(db.tasks_between(TOMORROW, TOMORROW)) == 4      # saved once, not duplicated
    assert state.posted and state.posted[0].startswith("Tomorrow's plan")


# ── 2. a posted message survives a failing later round ──

def test_a_posted_message_is_stored_when_the_next_round_fails(db, world, fake_claude):
    session = _session()
    # Round 1 posts the week and has a failing call, so a second round is needed;
    # the fake has no second round, so that round raises.
    fake_claude([text_block("One moment."), tool_block("show_week", {}, "t1"),
                 tool_block("update_task_status", {"task_id": 999, "status": "done"}, "t2")])
    events = []
    with pytest.raises(Exception):
        sessions.run_turn(session["id"], "week", on_text=lambda k, t: events.append((k, t)), now=NOW)
    rows = db.session_messages(session["id"])
    assert [r["author"] for r in rows if r["role"] == "assistant"][-1] == "code"
    assert "The week ahead" in rows[-1]["content"][0]["text"]
    assert any(k == "delta" and "The week ahead" in t for k, t in events)


def test_a_requested_close_is_carried_out_when_the_next_round_fails(db, world, fake_claude):
    session = _session()
    fake_claude([text_block("Night."), tool_block("close_session", {}, "t1"),
                 tool_block("update_task_status", {"task_id": 999, "status": "done"}, "t2")],
                summary="He closed it.")
    events = []
    with pytest.raises(Exception):
        sessions.run_turn(session["id"], "bye", on_text=lambda k, t: events.append((k, t)), now=NOW)
    assert db.get_session(session["id"])["status"] == "closed"
    assert events[-1][0] == "closed"


# ── 3. a block on another day is not tomorrow's time ──

def test_a_block_on_another_day_reads_not_booked(db, world):
    task_id = _task(db, "Study", TOMORROW, must_do=True)
    event_id = world.add(MILES, "wrong_day", "Study", at(27, 14), at(27, 16))
    db.set_task_calendar_event(task_id, event_id)
    assert rp.render_plan(TOMORROW).splitlines()[1] == "1. Study, not booked"


def test_confirming_moves_the_block_that_is_on_another_day(db, world):
    fixed_night(db, world)
    session = _session()
    study = db.session_plan(session["id"])["must_dos"][0]["task_id"]
    old = world.add(MILES, "wrong_day", "Study", at(27, 14), at(27, 16))
    db.set_task_calendar_event(study, old)
    token, _ = _in_turn(session["id"])
    try:
        nr.confirm_plan(now=NOW)
    finally:
        session_types.end_turn(token)
    # The same event, now on the 25th: nothing left behind, nothing booked twice.
    assert db.get_task(study)["calendar_event_id"] == old
    assert world.events[MILES][old]["start"]["dateTime"].startswith("2026-11-25")
    assert world.events[MILES][old]["summary"] == "Linear algebra study"
    assert len([c for c, _ in world.inserts if c == MILES]) == 2      # the other two items


def test_a_preview_never_clears_a_block_id_and_remove_block_still_finds_it(db, world):
    fixed_night(db, world)
    session = _session()
    study = db.session_plan(session["id"])["must_dos"][0]["task_id"]
    old = world.add(MILES, "wrong_day", "Study", at(27, 14), at(27, 16))
    db.set_task_calendar_event(study, old)
    token, _ = _in_turn(session["id"])
    try:
        nr.set_plan(must_dos=["Study for linear algebra test"], now=NOW)
    finally:
        session_types.end_turn(token)
    assert db.get_task(study)["calendar_event_id"] == old            # a preview wrote nothing
    assert "Study for linear algebra test, not booked" not in rp.render_plan(TOMORROW)
    assert nr.remove_block(study, now=NOW).startswith("Removed the block")
    assert old not in world.events[MILES]


def test_schedule_block_moves_a_block_from_another_day(db, world):
    task_id = _task(db, "Study", TOMORROW)
    old = world.add(MILES, "wrong_day", "Study", at(27, 14), at(27, 16))
    db.set_task_calendar_event(task_id, old)
    nr.schedule_block(task_id, 60, start_time="15:00", now=NOW)
    assert world.inserts == []
    assert world.events[MILES][old]["start"]["dateTime"].startswith("2026-11-25T15:00")
    assert db.get_task(task_id)["calendar_event_id"] == old


# ── 4. a dropped must do becomes optional ──

def test_a_dropped_must_do_becomes_optional_and_loses_its_block(db, world):
    old = _task(db, "Old must do", TOMORROW, must_do=True)
    session = _session()
    block = world.add(MILES, "oldblock", "Old must do", at(25, 15), at(25, 16))
    db.set_task_calendar_event(old, block)
    token, _ = _in_turn(session["id"])
    try:
        nr.set_plan(must_dos=["Something new"], now=NOW)
        nr.confirm_plan(now=NOW)
    finally:
        session_types.end_turn(token)
    task = db.get_task(old)
    assert (task["must_do"], task["scheduled_for"], task["calendar_event_id"]) == (0, TOMORROW, None)
    assert block not in world.events[MILES]
    assert [t["text"] for t in _must_dos(db)] == ["Something new"]


def test_dropped_must_dos_do_not_count_against_the_cap(db, world):
    for name in ("A", "B", "C"):
        _task(db, name, TOMORROW, must_do=True)
    session = _session()
    token, _ = _in_turn(session["id"])
    try:
        nr.set_plan(must_dos=["X", "Y", "Z"], now=NOW)
        nr.confirm_plan(now=NOW)                     # would have been refused as 6 must dos
    finally:
        session_types.end_turn(token)
    assert sorted(t["text"] for t in _must_dos(db)) == ["X", "Y", "Z"]


# ── 5. club events are not exams ──

def test_a_club_event_with_an_exam_word_is_not_suggested():
    events = roles.annotate([ev("Final GBM of the Semester", at(26, 18), at(26, 19), calendar=ACM),
                             ev("Calc Exam", at(26, 9), at(26, 10))])
    assert [p["title"] for p in rp.suggest(date(2026, 11, 24), [], events)] == ["Study for Calc Exam"]


def test_an_attended_club_exam_word_event_still_counts():
    gbm = ev("Final GBM of the Semester", at(26, 18), at(26, 19), calendar=ACM, series="gbm")
    events = roles.annotate([gbm], attending={(ACM, "gbm")})
    assert [p["title"] for p in rp.suggest(date(2026, 11, 24), [], events)] == [
        "Study for Final GBM of the Semester"]


# ── 6. observances are not break days ──

def test_an_observance_does_not_cancel_a_class():
    observance = ev("St. Patrick's Day", at(17, 0, month=3), at(18, 0, month=3), calendar=HOLIDAYS,
                    all_day=True, description="Observance\nTo hide observances, go to Settings")
    dsa = ev("DSA", at(17, 13, 55, month=3), at(17, 14, 45, month=3), repeating=True)
    [_, cls] = roles.annotate([observance, dsa])
    assert cls["busy"] and cls["mark"] == ""


def test_a_public_holiday_in_the_list_cancels_a_class():
    holiday = ev("Thanksgiving Day", at(26, 0), at(27, 0), calendar=HOLIDAYS, all_day=True,
                 description="Public holiday")
    dsa = ev("DSA", at(26, 13, 55), at(26, 15, 50), repeating=True)
    # Same date as the holiday but outside UF_BREAKS would still be a break: check
    # on a day UF_BREAKS does not cover.
    veterans = ev("Veterans Day", at(11, 0), at(12, 0), calendar=HOLIDAYS, all_day=True,
                  description="Public holiday")
    dsa2 = ev("DSA", at(11, 13, 55), at(11, 14, 45), repeating=True)
    assert roles.annotate([holiday, dsa])[1]["mark"] == "(likely cancelled, Thanksgiving break)"
    assert roles.annotate([veterans, dsa2])[1]["mark"] == "(likely cancelled, Veterans Day)"


def test_a_public_holiday_not_in_the_uf_list_is_a_class_day():
    presidents = ev("Presidents' Day", at(16, 0, month=2), at(17, 0, month=2), calendar=HOLIDAYS,
                    all_day=True, description="Public holiday")
    dsa = ev("DSA", at(16, 13, 55, month=2), at(16, 14, 45, month=2), repeating=True)
    assert roles.annotate([presidents, dsa])[1]["busy"] is True


# ── 7. the guard catches more ways to say a time ──

@pytest.mark.parametrize("line", [
    "Your study block is at ten.", "Booked study for two to four.", "Study 2-4 today.",
    "Study 2–4 today.", "See you at 2 tomorrow.", "Study at 1430.", "Ten pm works.",
    "Back by six o'clock."])
def test_more_time_forms_are_dropped(line):
    assert sessions.has_clock_time(line)
    assert sessions.guard_text(line) == ""


@pytest.mark.parametrize("line", ["Good luck on the 3 tasks.", "Added 2 items.", "Nice work tonight."])
def test_ordinary_text_passes_the_guard(line):
    assert sessions.guard_text(line) == line


# ── 8. one task twice in a list ──

def test_the_same_task_twice_is_refused(db, world):
    task_id = _task(db, "Gym", TOMORROW)
    with pytest.raises(ToolNotice, match="listed twice"):
        nr.schedule_block(tasks=[{"task_id": task_id}, {"task_id": task_id}], now=NOW)
    assert world.inserts == []


# ── 9. a reload reads as the stream did ──

def test_a_reload_matches_the_stream_when_the_guard_empties_the_last_round(db, world, fake_claude):
    session = _session()
    fake_claude([text_block("Let me look."), tool_block("show_week", {}, "t1"),
                 tool_block("update_task_status", {"task_id": 999, "status": "done"}, "t2")],
                [text_block("Starts at 2 PM sharp.")])
    streamed = []
    sessions.run_turn(session["id"], "week",
                      on_text=lambda k, t: streamed.append(t) if k == "delta" else None, now=NOW)
    shown = [m["text"] for m in sessions.public_messages(db.session_messages(session["id"]))]
    assert shown[2:] == ["".join(streamed).strip()] or shown[2:] == streamed
    assert not any("Let me look" in t for t in shown)


# ── 10. paging ──

class _Pages:
    """A calendar whose list answers in pages of two."""

    def __init__(self, events):
        self.events, self.tokens = events, []

    def calendarList(self):
        from types import SimpleNamespace
        item = {"id": PRIMARY, "summary": "me", "selected": True, "accessRole": "owner",
                "primary": True}
        return SimpleNamespace(list=lambda pageToken=None: fake_google._Exec({"items": [item]}))

    def events_(self):
        from types import SimpleNamespace
        service = self

        def list_(calendarId, pageToken=None, **_):
            service.tokens.append(pageToken)
            start = int(pageToken or 0)
            page = service.events[start:start + 2]
            more = start + 2 < len(service.events)
            return fake_google._Exec({"items": page, **({"nextPageToken": str(start + 2)} if more else {})})
        return SimpleNamespace(list=list_)


def test_events_are_read_across_every_page(db, monkeypatch):
    events = [{"id": f"e{n}", "summary": f"Event {n}",
               "start": {"dateTime": at(25, 10 + n).astimezone().isoformat()},
               "end": {"dateTime": at(25, 10 + n, 30).astimezone().isoformat()}} for n in range(5)]
    pages = _Pages(events)
    service = fake_google._Service(pages)
    monkeypatch.setattr(calendar_tools, "_service", lambda: service)
    calendar_tools._calendar_cache.clear()
    busy, _ = calendar_tools.busy_blocks(at(25, 0), at(26, 0))
    assert len(busy) == 5                       # the first page alone would give two
    assert pages.tokens == [None, "2", "4"]


# ── 11. the Evallo guard refetches ──

def test_a_stale_event_copy_cannot_get_past_the_guard(world):
    world.add(MILES, "tutor", "Tutoring", at(25, 16), at(25, 17),
              extendedProperties={"private": {"evalloSessionId": "9"}})
    stale = {"id": "tutor", "summary": "Tutoring"}            # no extendedProperties
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools._write_patch(MILES, "tutor", {"summary": "x"}, stale)
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools._write_delete(MILES, "tutor", stale)
    assert world.written() == []


# ── 12. a bare file name for the key ──

def test_a_bare_key_file_name_works(tmp_path):
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts", "capture_key.py")
    env = {**os.environ, "MILES_DB_PATH": str(tmp_path / "k.db")}
    out = subprocess.run([sys.executable, script, "new", "bare", "--path", "bare.key"],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "bare.key").read_text().startswith("mcap_")


# ── A. short block titles, details in the description ──

@pytest.mark.parametrize("title, short", [
    ("Study for linear algebra test", "Linear algebra study"),
    ("Study for the DSA exam", "DSA study"),
    ("Fix Friday: Test Prep with Alex overlaps Dentist", "Fix Friday"),
    ("Exam 3 prep: Lessons 19 to 21", "Exam 3 prep"),
    ("Email professor about the lab report", "Email professor lab report"),
    ("Order NVMe HAT and SSD", "Order NVMe HAT SSD"),
    ("Gym", "Gym"),
])
def test_block_titles_are_short_and_deterministic(title, short):
    assert rp.block_title(title) == short
    assert len(rp.block_title(title).split()) <= 4


def test_an_explicit_short_title_wins_but_is_capped():
    assert rp.block_title("Study for linear algebra test", "Exam prep") == "Exam prep"
    assert rp.block_title("Study", "Exam 3 prep: Lessons 19 to 21") == "Exam 3 prep: Lessons 19 to 21"
    long = rp.block_title("x", "Supercalifragilistic Expialidocious Extraordinary Words Again")
    assert len(long) <= 40 and long == "Supercalifragilistic Expialidocious"
    assert rp.block_title("x", "A" * 60) == "A" * 40


@pytest.mark.parametrize("title, short", [
    (": notes", "Notes"), ("Study for", "Study for"), ("", ""),
    (":", ""), ("Fix: nothing here", "Fix: nothing here")])
def test_a_title_that_comes_out_empty_or_one_word_falls_back_to_the_task_words(title, short):
    assert rp.block_title(title) == short


def test_confirming_puts_the_details_in_the_description(db, world):
    fixed_night(db, world)
    session = _session()
    study = db.session_plan(session["id"])["must_dos"][0]["task_id"]
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE tasks SET notes = 'work practice problems, not rereading' WHERE id = ?",
                 (study,))
    conn.commit()
    conn.close()
    token, _ = _in_turn(session["id"])
    try:
        nr.confirm_plan(now=NOW)
    finally:
        session_types.end_turn(token)
    block = world.events[MILES][db.get_task(study)["calendar_event_id"]]
    assert block["summary"] == "Linear algebra study"
    assert block["description"].splitlines() == [
        "Study for linear algebra test", "Why: Due Mon Nov 30",
        "Notes: work practice problems, not rereading"]


def test_a_study_plan_block_names_its_topic_and_source_folder(db, world):
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, source_material TEXT)")
    conn.execute("INSERT INTO topics VALUES (3, 'Lessons 19 to 21', '/notes/dsa/exam3')")
    conn.commit()
    conn.close()
    task_id = _task(db, "Exam 3 prep: Lessons 19 to 21", TOMORROW, must_do=True)
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE tasks SET source = 'study_plan', topic_id = 3, short_title = 'Exam 3 prep' "
                 "WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    nr.schedule_block(task_id, 120, start_time="14:00", now=NOW)
    [(_, body)] = world.inserts
    assert body["summary"] == "Exam 3 prep"
    assert body["description"].splitlines() == [
        "Exam 3 prep: Lessons 19 to 21", "Topic: Lessons 19 to 21",
        "Source material: /notes/dsa/exam3"]


def test_moving_a_block_refreshes_its_title_and_description(db, world):
    task_id = _task(db, "Study for linear algebra test", TOMORROW)
    nr.schedule_block(task_id, 120, start_time="14:00", now=NOW)
    nr.move_block(task_id, "17:00", now=NOW)
    block = world.events[MILES][db.get_task(task_id)["calendar_event_id"]]
    assert block["summary"] == "Linear algebra study"
    assert block["description"].startswith("Study for linear algebra test")
    assert block["start"]["dateTime"].startswith("2026-11-25T17:00")


# ── B. class titles ──

@pytest.mark.parametrize("title", ["DSA", "COP3530", "COP 3530", "CDA3101", "CDA 3101",
                                   "MAS3114", "MAS 3114", "ENC2256", "ENC 2256", "cda 3101"])
def test_class_titles_in_both_spellings(title):
    assert roles.is_class(title)


@pytest.mark.parametrize("title", ["MAS3114 Exam 3", "DSA Final Exam", "CDA3101 Office Hours",
                                   "Test Prep"])
def test_a_title_with_more_words_is_not_a_class(title):
    assert not roles.is_class(title)


def test_a_course_code_class_is_cancelled_on_a_break_day():
    [cls] = roles.annotate([ev("MAS 3114", at(25, 9), at(25, 10), repeating=True)])
    assert cls["mark"].startswith("(likely cancelled") and not cls["busy"]
    [exam] = roles.annotate([ev("MAS3114 Exam 3", at(25, 9), at(25, 10))])
    assert exam["busy"]


# ── C. moving only moves ──

def test_a_hand_renamed_block_keeps_its_title_and_description_when_moved(db, world):
    task_id = _task(db, "Study for linear algebra test", TOMORROW)
    nr.schedule_block(task_id, 120, start_time="14:00", now=NOW)
    event = world.events[MILES][db.get_task(task_id)["calendar_event_id"]]
    event["summary"], event["description"] = "My own name", "My own notes"      # he edited it
    nr.move_block(task_id, "17:00", now=NOW)
    assert (event["summary"], event["description"]) == ("My own name", "My own notes")
    assert event["start"]["dateTime"].startswith("2026-11-25T17:00")


# ── the guard, more spoken times ──

@pytest.mark.parametrize("line", ["2 o'clock works.", "Half past two.", "Ten thirty works.",
                                  "Study 10.30 tomorrow.", "Quarter to three."])
def test_still_more_time_forms_are_dropped(line):
    assert sessions.guard_text(line) == ""
