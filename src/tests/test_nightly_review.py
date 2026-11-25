"""The nightly review's rules and tools: the must do cap, status updates and
moves, slot finding, and schedule_block against a fake Google that records
where it was asked to write."""

import re
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

import calendar_tools
import nightly_review as nr
from tools import ToolNotice

NOW = datetime(2026, 11, 24, 22, 0)       # a Tuesday night; the 4am day is the 24th
TODAY, TOMORROW = "2026-11-24", "2026-11-25"


def _add(db, text, day, must_do=False, notes=None):
    [task_id] = db.add_planned_tasks(
        [{"text": text, "scheduled_for": day, "must_do": must_do, "notes": notes}],
        datetime(2026, 11, 24, 8, 0))
    return task_id


def _completions(db, task_id):
    conn = sqlite3.connect(db.DB_PATH)
    try:
        return conn.execute("SELECT date FROM task_completions WHERE task_id = ?",
                            (task_id,)).fetchall()
    finally:
        conn.close()


# ── the must do cap ──

def test_overflow_is_empty_when_the_batch_fits():
    assert nr.must_do_overflow({TOMORROW: 1}, [(TOMORROW, True), (TOMORROW, True),
                                                (TOMORROW, False)]) == {}


def test_overflow_names_the_day_and_the_slots_left():
    assert nr.must_do_overflow({TOMORROW: 2}, [(TOMORROW, True), (TOMORROW, True)]) == {
        TOMORROW: (1, 2)}
    assert nr.must_do_overflow({}, [(TOMORROW, True)] * 4) == {TOMORROW: (3, 4)}
    assert nr.must_do_overflow({TOMORROW: 3}, [(TOMORROW, False)]) == {}


def test_create_tasks_refuses_a_fourth_must_do_and_saves_nothing(db):
    _add(db, "Lab report", TOMORROW, must_do=True)
    _add(db, "Problem set", TOMORROW, must_do=True)
    with pytest.raises(ToolNotice) as refused:
        nr.create_tasks([{"title": "Gym", "date": TOMORROW, "must_do": True},
                         {"title": "Read chapter 4", "date": TOMORROW, "must_do": True},
                         {"title": "Laundry", "date": TOMORROW}], now=NOW)
    assert str(refused.value) == (
        "Nothing was saved. At most 3 must dos a day: Wednesday November 25 has room for "
        "1 more must do and this asked for 2. Ask which to make optional.")
    assert len(db.tasks_between(TOMORROW, TOMORROW)) == 2


def test_create_tasks_saves_must_dos_and_optional_ones(db):
    made = nr.create_tasks([{"title": "Gym", "date": TOMORROW, "must_do": True},
                            {"title": "Laundry", "date": TOMORROW, "notes": "towels too"}],
                           now=NOW)
    assert [(c["title"], c["date"], c["must_do"]) for c in made["created"]] == [
        ("Gym", TOMORROW, True), ("Laundry", TOMORROW, False)]
    laundry = db.get_task(made["created"][1]["id"])
    assert (laundry["status"], laundry["notes"], laundry["source"]) == ("pending", "towels too", "manual")


def test_a_moved_must_do_does_not_hold_its_old_slot(db):
    first = _add(db, "One", TOMORROW, must_do=True)
    _add(db, "Two", TOMORROW, must_do=True)
    _add(db, "Three", TOMORROW, must_do=True)
    nr.update_task_status(first, "moved", move_to="2026-11-26", now=NOW)
    nr.create_tasks([{"title": "Four", "date": TOMORROW, "must_do": True}], now=NOW)


def test_create_tasks_refuses_a_past_date(db):
    with pytest.raises(ToolNotice):
        nr.create_tasks([{"title": "Gym", "date": "2026-11-23"}], now=NOW)


# ── status updates ──

def test_done_writes_the_completion_voice_reads(db):
    task_id = _add(db, "Lab report", TODAY)
    assert nr.update_task_status(task_id, "done", now=NOW) == "Marked Lab report done."
    assert db.get_task(task_id)["status"] == "done"
    assert _completions(db, task_id) == [(TODAY,)]
    assert task_id not in [t[0] for t in db.tasks_today(NOW)]


def test_skipped_and_back_to_pending(db):
    task_id = _add(db, "Laundry", TODAY)
    assert nr.update_task_status(task_id, "skipped", now=NOW) == "Marked Laundry skipped."
    assert db.get_task(task_id)["status"] == "skipped"
    nr.update_task_status(task_id, "done", now=NOW)
    assert nr.update_task_status(task_id, "pending", now=NOW) == "Marked Laundry not done yet."
    assert db.get_task(task_id)["status"] == "pending"
    assert _completions(db, task_id) == []


def test_moving_copies_the_task_to_the_new_day(db):
    task_id = _add(db, "Read chapter 4", TODAY, must_do=True, notes="pages 80 to 95")
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE tasks SET source = 'canvas', assignment_id = 7, topic_id = 3, "
                 "calendar_event_id = 'evt_old' WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()

    said = nr.update_task_status(task_id, "moved", move_to=TOMORROW, now=NOW)

    old = db.get_task(task_id)
    [copy] = [t for t in db.tasks_between(TOMORROW, TOMORROW)]
    assert said == f"Moved Read chapter 4 to Wednesday November 25 (the copy there is task {copy['id']})."
    assert (old["status"], old["scheduled_for"]) == ("moved", TODAY)
    assert (copy["text"], copy["notes"], copy["must_do"], copy["source"], copy["assignment_id"],
            copy["topic_id"], copy["status"], copy["scheduled_for"]) == (
        "Read chapter 4", "pages 80 to 95", True, "canvas", 7, 3, "pending", TOMORROW)
    # The block was for the old day; the copy gets its own.
    assert copy["calendar_event_id"] is None


def test_moving_needs_a_date(db):
    task_id = _add(db, "Laundry", TODAY)
    with pytest.raises(ToolNotice, match="needs move_to"):
        nr.update_task_status(task_id, "moved", now=NOW)
    assert db.get_task(task_id)["status"] == "pending"


def test_a_must_do_cannot_move_onto_a_full_day(db):
    for title in ("One", "Two", "Three"):
        _add(db, title, TOMORROW, must_do=True)
    task_id = _add(db, "Four", TODAY, must_do=True)
    with pytest.raises(ToolNotice, match="already has 3 must dos"):
        nr.update_task_status(task_id, "moved", move_to=TOMORROW, now=NOW)
    assert db.get_task(task_id)["status"] == "pending"


def test_a_daily_task_can_only_be_done(db):
    task_id = db.add_task("Take vitamins", "daily")
    with pytest.raises(ToolNotice, match="daily task"):
        nr.update_task_status(task_id, "skipped", now=NOW)
    assert nr.update_task_status(task_id, "done", now=NOW) == "Marked Take vitamins done for today."
    [daily] = [t for t in db.tasks_for_day(TODAY, TODAY) if t["id"] == task_id]
    assert daily["status"] == "done"


def test_an_unknown_task_is_a_notice(db):
    with pytest.raises(ToolNotice, match="no task with id 99"):
        nr.update_task_status(99, "done", now=NOW)


# ── what the review says ──

def test_the_opening_lists_today_with_statuses(db):
    db.add_task("Take vitamins", "daily")
    _add(db, "Lab report", TODAY, must_do=True)
    done = _add(db, "Laundry", TODAY)
    nr.update_task_status(done, "done", now=NOW)
    _add(db, "Gym", TOMORROW)
    assert nr.opening(NOW) == (
        "Here is Tuesday November 24:\n\n"
        "1. Take vitamins (daily): not done yet\n"
        "2. Lab report (must do): not done yet\n"
        "3. Laundry: done\n\n"
        "What got done?")
    assert nr.push_body(NOW) == "3 tasks today. What got done, and what matters tomorrow?"


def test_an_empty_day_opens_with_a_question(db):
    assert nr.opening(NOW) == ("Nothing was on your list for Tuesday November 24. What did "
                               "you get done today, and what matters tomorrow?")
    assert nr.push_body(NOW) == "No tasks today. What got done, and what matters tomorrow?"


def test_get_tasks_for_date_has_the_ids(db):
    task_id = _add(db, "Lab report", TODAY, must_do=True)
    result = nr.get_tasks_for_date(TODAY, now=NOW)
    assert result["tasks"] == [{"id": task_id, "title": "Lab report", "status": "pending",
                                "must_do": True, "daily": False, "notes": None,
                                "scheduled_for": TODAY, "has_block": False}]


def test_a_calendar_failure_is_reported_not_raised(db, monkeypatch):
    _add(db, "Gym", TOMORROW, must_do=True)

    def down(start, end):
        raise RuntimeError("token expired")
    monkeypatch.setattr(calendar_tools, "events_for_app", down)
    result = nr.get_upcoming_items(now=NOW)
    assert result["calendar_error"] == "could not read the calendar: token expired"
    assert [t["title"] for t in result["tasks"]] == ["Gym"]
    assert (result["from"], result["to"]) == (TOMORROW, "2026-12-01")


def test_the_summary_counts_come_from_the_table(db):
    a = _add(db, "Lab report", TODAY)
    b = _add(db, "Laundry", TODAY)
    c = _add(db, "Read", TODAY)
    nr.update_task_status(a, "done", now=NOW)
    nr.update_task_status(b, "skipped", now=NOW)
    nr.update_task_status(c, "moved", move_to=TOMORROW, now=NOW)
    _add(db, "Gym", TOMORROW, must_do=True)
    assert nr.summary_facts(TODAY) == (
        "Today: 1 done, 1 skipped, 1 moved, 0 left open, of 3 tasks. "
        "Tomorrow: 2 tasks, 1 must do, 0 blocks on the calendar.")


def test_last_nights_summary_reaches_the_prompt(db):
    import sessions
    yesterday, _ = sessions.create("nightly_review", datetime(2026, 11, 23, 21, 30))
    db.finish_session(yesterday["id"], "closed", "Planned the lab for Tuesday.",
                      datetime(2026, 11, 23, 22, 0))
    tonight, _ = sessions.create("nightly_review", NOW)
    prompt = nr.system_prompt(tonight)
    assert "Last night's review (Monday November 23) ended with:\nPlanned the lab for Tuesday." in prompt
    assert "Wednesday November 25: 2026-11-25" in prompt
    # No hyphens in the prose: only the date formats and list bullets carry them.
    prose = re.sub(r"\d{4}-\d{2}-\d{2}|^- ", "", prompt.replace("YYYY-MM-DD", ""), flags=re.M)
    assert "-" not in prose


# ── finding a slot ──

def _at(hour, minute=0, day=25):
    return datetime(2026, 11, day, hour, minute)


def test_the_first_gap_that_fits():
    busy = [(_at(9), _at(10)), (_at(10, 30), _at(12))]
    # 10:00 to 10:30 is too short for an hour; 12:00 is the first that fits.
    assert nr.find_slot(busy, _at(9), _at(21), 60) == (_at(12), _at(13))
    assert nr.find_slot(busy, _at(9), _at(21), 30) == (_at(10), _at(10, 30))


def test_the_window_is_respected():
    busy = [(_at(9), _at(20, 30))]
    assert nr.find_slot(busy, _at(9), _at(21), 60) is None
    assert nr.find_slot(busy, _at(9), _at(22), 60) == (_at(20, 30), _at(21, 30))
    # Busy time outside the window does not matter, and nothing starts before it.
    assert nr.find_slot([(_at(6), _at(8))], _at(9), _at(21), 60) == (_at(9), _at(10))


def test_not_before_is_rounded_to_a_quarter_hour():
    assert nr.find_slot([], _at(9, day=24), _at(23, day=24), 30, not_before=_at(22, 7, day=24)) == (
        _at(22, 15, day=24), _at(22, 45, day=24))


def test_overlap():
    busy = [(_at(14), _at(15))]
    assert nr.overlaps(busy, _at(14, 30), _at(15, 30))
    assert not nr.overlaps(busy, _at(15), _at(16))
    assert not nr.overlaps(busy, _at(13), _at(14))


# ── schedule_block against a fake Google ──

class _Exec:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class FakeGoogle:
    """Three calendars: his primary, MILES, and a club calendar he follows.
    Records every insert, so the test can say where writes went."""

    MILES_ID = "fake-calendar-06@example.com"

    def __init__(self, busy):
        self.busy = busy
        self.inserts = []

    def calendarList(self):
        items = [
            {"id": "lee@example.com", "summary": "lee@example.com", "primary": True,
             "selected": True, "accessRole": "owner"},
            {"id": self.MILES_ID, "summary": "MILES", "selected": True, "accessRole": "owner"},
            {"id": "fake-calendar-03@example.com", "summary": "UF IEEE", "selected": True,
             "accessRole": "reader"},
        ]
        return SimpleNamespace(list=lambda pageToken=None: _Exec({"items": items}))

    def freebusy(self):
        def query(body):
            return _Exec({"calendars": {
                "lee@example.com": {"busy": [
                    {"start": a.astimezone().isoformat(), "end": b.astimezone().isoformat()}
                    for a, b in self.busy]},
                self.MILES_ID: {"busy": []},
                "fake-calendar-03@example.com": {"busy": []},
            }})
        return SimpleNamespace(query=query)

    def events(self):
        def insert(calendarId, body):
            self.inserts.append((calendarId, body))
            return _Exec({"id": f"evt{len(self.inserts)}"})
        return SimpleNamespace(insert=insert)

    def calendars(self):
        raise AssertionError("MILES exists; nothing should create a calendar")


@pytest.fixture
def google(monkeypatch):
    def install(busy=()):
        fake = FakeGoogle(list(busy))
        calendar_tools._calendar_cache.clear()
        monkeypatch.setattr(calendar_tools, "_service", lambda: fake)
        return fake
    yield install
    calendar_tools._calendar_cache.clear()


def test_a_block_takes_the_first_free_gap_on_miles(db, google):
    fake = google(busy=[(_at(9), _at(11)), (_at(13), _at(14))])
    task_id = _add(db, "Lab report", TOMORROW, must_do=True)

    said = nr.schedule_block(task_id, 90, now=NOW)

    assert said == ("Blocked Lab report on Wednesday November 25 from 11 AM to 12:30 PM "
                    "on the MILES calendar.")
    [(calendar_id, body)] = fake.inserts
    assert calendar_id == FakeGoogle.MILES_ID
    assert body["summary"] == "Lab report"
    assert db.get_task(task_id)["calendar_event_id"] == "evt1"


def test_a_chosen_time_that_overlaps_is_refused_with_the_gaps(db, google):
    fake = google(busy=[(_at(9), _at(11)), (_at(13), _at(14))])
    task_id = _add(db, "Lab report", TOMORROW)
    with pytest.raises(ToolNotice) as refused:
        nr.schedule_block(task_id, 60, start_time="13:30", now=NOW)
    assert str(refused.value) == (
        "1:30 PM to 2:30 PM on Wednesday November 25 overlaps something on his calendar. "
        "Nothing was added. Free gaps of 60 minutes or more between 9 AM and 9 PM: "
        "11 AM to 1 PM, 2 PM to 9 PM.")
    assert fake.inserts == []
    assert db.get_task(task_id)["calendar_event_id"] is None


def test_a_chosen_free_time_is_booked(db, google):
    fake = google(busy=[(_at(9), _at(11))])
    task_id = _add(db, "Gym", TOMORROW)
    nr.schedule_block(task_id, 60, start_time="17:00", now=NOW)
    [(calendar_id, body)] = fake.inserts
    assert calendar_id == FakeGoogle.MILES_ID
    assert body["start"]["dateTime"].startswith("2026-11-25T17:00:00")


def test_a_task_with_a_block_is_refused(db, google):
    fake = google()
    task_id = _add(db, "Gym", TOMORROW)
    nr.schedule_block(task_id, 60, now=NOW)
    with pytest.raises(ToolNotice, match="already has a block"):
        nr.schedule_block(task_id, 60, now=NOW)
    assert len(fake.inserts) == 1


def test_no_gap_in_the_window_is_a_notice(db, google):
    fake = google(busy=[(_at(8), _at(22))])
    task_id = _add(db, "Gym", TOMORROW)
    with pytest.raises(ToolNotice, match="No free 60 minute gap between 9 AM and 9 PM"):
        nr.schedule_block(task_id, 60, now=NOW)
    nr.schedule_block(task_id, 60, earliest="22:00", latest="23:30", now=NOW)
    assert [c for c, _ in fake.inserts] == [FakeGoogle.MILES_ID]


def test_the_review_tools_and_their_tiers():
    from tools import Permission
    expected = {
        "get_tasks_for_date": (Permission.READ, "hokage"),
        "update_task_status": (Permission.WRITE, None),
        "get_upcoming_items": (Permission.READ, "hokage"),
        "create_tasks": (Permission.WRITE, None),
        "schedule_block": (Permission.EXTERNAL_WRITE, "hokage"),
        "close_session": (Permission.CONTROL, None),
    }
    assert set(nr.registry.names()) == set(expected)
    for name, (permission, tier) in expected.items():
        spec = nr.registry.get(name)
        assert (spec.permission, spec.min_tier) == (permission, tier)
        assert name in nr.STAGES
    assert nr.registry.get("close_session").returns_to_model is False
