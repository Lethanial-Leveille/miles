"""The Tasks tab routes against docs/TASKS_CONTRACT.md: shapes, the 4am day,
ordering, the 409 and 422 cases, and the backlog route's guarded block removal."""

import os
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier, Event
from types import SimpleNamespace
from datetime import datetime

import database

database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402
import tasks_view                                # noqa: E402
import fake_google
import nightly_review as nr
import session_types
from test_review_v2 import (world, fixed_night, _session, _in_turn,   # noqa: F401
                            TOMORROW, NOW as REVIEW_NOW)

TASK_KEYS = {"id", "title", "short_title", "notes", "daily", "status", "must_do",
             "scheduled_for", "due_date", "overdue", "has_block", "source"}
NOW = datetime(2026, 11, 25, 12, 0)              # Wednesday noon, the 4am day is the 25th


@pytest.fixture
def clock(monkeypatch):
    state = {"now": NOW}
    monkeypatch.setattr(tasks_view, "now", lambda: state["now"])
    return state


@pytest.fixture
def client(db, clock):
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def _add(db, text, **kw):
    return db.add_task(text, now=NOW, **kw)


def _ids(tasks):
    return [t["id"] for t in tasks]


def test_overview_shape_and_labels(client, db):
    daily = _add(db, "Vitamins", recurrence="daily")
    one = _add(db, "Call the bank", scheduled_for="2026-11-25")
    later = _add(db, "Lab report", scheduled_for="2026-11-26", due_date="2026-11-27")
    backlog = _add(db, "Project 3", due_date="2026-11-30")
    body = client.get("/tasks/overview").json()
    assert set(body) == {"today", "days", "backlog"}
    assert body["today"] == "2026-11-25"
    assert [(d["date"], d["label"]) for d in body["days"]] == [
        ("2026-11-25", "Today"), ("2026-11-26", "Thursday Nov 26"),
        ("2026-11-27", "Friday Nov 27")]
    assert _ids(body["days"][0]["tasks"]) == [daily, one]
    assert _ids(body["days"][1]["tasks"]) == [later]
    assert _ids(body["backlog"]) == [backlog]
    for task in body["days"][0]["tasks"] + body["backlog"]:
        assert set(task) == TASK_KEYS
    assert body["backlog"][0] == {
        "id": backlog, "title": "Project 3", "short_title": None, "notes": None,
        "daily": False, "status": "pending", "must_do": False, "scheduled_for": None,
        "due_date": "2026-11-30", "overdue": False, "has_block": False, "source": "manual"}


def test_days_parameter(client, db):
    assert len(client.get("/tasks/overview?days=1").json()["days"]) == 1
    assert len(client.get("/tasks/overview?days=7").json()["days"]) == 7
    assert client.get("/tasks/overview?days=0").status_code == 422
    assert client.get("/tasks/overview?days=8").status_code == 422


def test_later_days_hold_one_time_tasks_only_pending_or_done(client, db):
    _add(db, "Vitamins", recurrence="daily")
    pending = _add(db, "A", scheduled_for="2026-11-26")
    done = _add(db, "B", scheduled_for="2026-11-26")
    skipped = _add(db, "C", scheduled_for="2026-11-26")
    db.complete_task(done, NOW)
    db.set_task_status(skipped, "skipped", NOW)
    tasks = client.get("/tasks/overview").json()["days"][1]["tasks"]
    assert _ids(tasks) == [pending, done]
    assert [t["status"] for t in tasks] == ["pending", "done"]


def test_overdue_flag_and_earlier_tasks_on_today(client, db):
    late = _add(db, "Late", scheduled_for="2026-11-23")
    due_past = _add(db, "Due past", scheduled_for="2026-11-25", due_date="2026-11-24")
    fine = _add(db, "Fine", scheduled_for="2026-11-25", due_date="2026-11-25")
    today = {t["id"]: t for t in client.get("/tasks/overview").json()["days"][0]["tasks"]}
    assert today[late]["overdue"] is True
    assert today[due_past]["overdue"] is True
    assert today[fine]["overdue"] is False


def test_done_today_stays_visible_and_is_not_overdue(client, db):
    task = _add(db, "Late", scheduled_for="2026-11-23")
    assert client.post(f"/tasks/{task}/complete").status_code == 200
    shown = client.get("/tasks/overview").json()["days"][0]["tasks"]
    assert [(t["id"], t["status"], t["overdue"]) for t in shown] == [(task, "done", False)]


def test_moved_skipped_archived_never_appear(client, db):
    moved = _add(db, "M", scheduled_for="2026-11-25")
    db.move_task(moved, "2026-11-26", NOW)
    skipped = _add(db, "S", scheduled_for="2026-11-25")
    db.set_task_status(skipped, "skipped", NOW)
    archived = _add(db, "A", scheduled_for="2026-11-25")
    db.archive_task(archived, NOW)
    body = client.get("/tasks/overview").json()
    shown = _ids(body["days"][0]["tasks"]) + _ids(body["days"][1]["tasks"])
    assert moved not in shown[:1] and skipped not in shown and archived not in shown
    assert all(t["status"] != "moved" for t in body["days"][0]["tasks"])


def test_four_am_boundary(client, db, clock):
    daily = _add(db, "Vitamins", recurrence="daily")
    clock["now"] = datetime(2026, 11, 26, 1, 0)
    body = client.get("/tasks/overview").json()
    assert body["today"] == "2026-11-25"
    r = client.post(f"/tasks/{daily}/complete").json()
    assert r["day"] == "2026-11-25"
    clock["now"] = datetime(2026, 11, 26, 4, 0)
    body = client.get("/tasks/overview").json()
    assert body["today"] == "2026-11-26"
    assert body["days"][0]["tasks"][0]["status"] == "pending"


def test_backlog_order_due_soonest_then_null_then_added(client, db):
    a = _add(db, "no due first")
    b = _add(db, "due late", due_date="2026-12-07")
    c = _add(db, "due soon", due_date="2026-11-30")
    d = _add(db, "no due second")
    e = _add(db, "same date later", due_date="2026-11-30")
    _add(db, "scheduled", scheduled_for="2026-11-26")
    done = _add(db, "done backlog")
    db.complete_task(done, NOW)
    assert _ids(client.get("/tasks/overview").json()["backlog"]) == [c, e, b, a, d]


def test_create_today_and_backlog(client, db):
    r = client.post("/tasks", json={"title": "  Buy stamps ", "where": "today"})
    assert r.status_code == 201
    task = r.json()["task"]
    assert set(task) == TASK_KEYS
    assert (task["title"], task["scheduled_for"], task["due_date"]) == (
        "Buy stamps", "2026-11-25", None)
    r = client.post("/tasks", json={"title": "Read", "where": "backlog",
                                    "due_date": "2026-12-01"})
    task = r.json()["task"]
    assert (task["scheduled_for"], task["due_date"], task["status"]) == (
        None, "2026-12-01", "pending")
    task = client.post("/tasks", json={"title": "Plain", "where": "backlog"}).json()["task"]
    assert task["scheduled_for"] is None and task["due_date"] is None
    task = client.post("/tasks", json={"title": "Both", "where": "today",
                                       "due_date": "2026-11-28"}).json()["task"]
    assert (task["scheduled_for"], task["due_date"]) == ("2026-11-25", "2026-11-28")


def test_create_after_midnight_before_4am_is_the_previous_day(client, clock):
    clock["now"] = datetime(2026, 11, 26, 1, 0)
    task = client.post("/tasks", json={"title": "Late night", "where": "today"}).json()["task"]
    assert task["scheduled_for"] == "2026-11-25"


@pytest.mark.parametrize("body", [
    {"title": "", "where": "today"},
    {"title": "   ", "where": "today"},
    {"title": "x" * 201, "where": "today"},
    {"title": "ok", "where": "tomorrow"},
    {"where": "today"},
    {"title": "ok"},
    {"title": "ok", "where": "today", "due_date": "11/30/2026"},
    {"title": "ok", "where": "today", "due_date": "2026-02-30"},
    {"title": "ok", "where": "today", "due_date": "2026-11-30T10:00"},
])
def test_create_422(client, body):
    assert client.post("/tasks", json=body).status_code == 422


def test_title_of_200_is_fine(client):
    assert client.post("/tasks", json={"title": "x" * 200, "where": "today"}).status_code == 201


def test_complete_keeps_keys_and_adds_task(client, db):
    task = _add(db, "Do it", scheduled_for="2026-11-25")
    body = client.post(f"/tasks/{task}/complete").json()
    assert set(body) == {"completed", "day", "task"}
    assert body["completed"] is True and body["day"] == "2026-11-25"
    assert set(body["task"]) == TASK_KEYS and body["task"]["status"] == "done"
    assert client.post("/tasks/9999/complete").status_code == 404
    gone = _add(db, "Gone")
    db.archive_task(gone, NOW)
    assert client.post(f"/tasks/{gone}/complete").status_code == 404


def test_existing_today_route_still_works(client, db):
    _add(db, "Vitamins", recurrence="daily")
    assert client.get("/tasks/today").json()["tasks"][0]["text"] == "Vitamins"


def test_reopen_one_time(client, db):
    task = _add(db, "Do it", scheduled_for="2026-11-25")
    client.post(f"/tasks/{task}/complete")
    r = client.post(f"/tasks/{task}/reopen")
    assert r.status_code == 200
    assert set(r.json()) == {"task"} and r.json()["task"]["status"] == "pending"
    assert not db.task_completed_on(task, "2026-11-25")
    assert [t[0] for t in db.tasks_today(NOW)] == [task]


def test_reopen_daily(client, db):
    daily = _add(db, "Vitamins", recurrence="daily")
    client.post(f"/tasks/{daily}/complete")
    shown = client.get("/tasks/overview").json()["days"][0]["tasks"][0]
    assert shown["status"] == "done" and shown["daily"] is True
    r = client.post(f"/tasks/{daily}/reopen")
    assert r.json()["task"]["status"] == "pending"
    assert client.get("/tasks/overview").json()["days"][0]["tasks"][0]["status"] == "pending"


def test_reopen_errors(client, db):
    assert client.post("/tasks/9999/reopen").status_code == 404
    pending = _add(db, "P", scheduled_for="2026-11-25")
    assert client.post(f"/tasks/{pending}/reopen").status_code == 409
    daily = _add(db, "Vitamins", recurrence="daily")
    assert client.post(f"/tasks/{daily}/reopen").status_code == 409
    # a daily task done yesterday is not done today
    db_conn = __import__("sqlite3").connect(db.DB_PATH)
    db_conn.execute("INSERT INTO task_completions VALUES (?, '2026-11-24')", (daily,))
    db_conn.commit()
    db_conn.close()
    assert client.post(f"/tasks/{daily}/reopen").status_code == 409


def test_backlog_without_block(client, db):
    task = _add(db, "Plan", scheduled_for="2026-11-26", due_date="2026-11-30")
    db.save_day_plan("2026-11-26", [{"task_id": task, "text": "Plan", "must_do": True}], NOW)
    r = client.post(f"/tasks/{task}/backlog")
    assert r.status_code == 200
    t = r.json()["task"]
    assert (t["scheduled_for"], t["must_do"], t["has_block"], t["due_date"]) == (
        None, False, False, "2026-11-30")
    assert _ids(client.get("/tasks/overview").json()["backlog"]) == [task]


def test_backlog_errors(client, db):
    assert client.post("/tasks/9999/backlog").status_code == 404
    daily = _add(db, "Vitamins", recurrence="daily")
    assert client.post(f"/tasks/{daily}/backlog").status_code == 409
    already = _add(db, "Already")
    assert client.post(f"/tasks/{already}/backlog").status_code == 409


def test_backlog_removes_only_a_miles_block(client, db, monkeypatch):
    fake = fake_google.install(monkeypatch)
    start, end = datetime(2026, 11, 26, 10), datetime(2026, 11, 26, 11)
    fake.add(fake_google.MILES, "milesblock", "Study", start, end)
    task = _add(db, "Study", scheduled_for="2026-11-26")
    db.set_task_calendar_event(task, "milesblock", NOW)
    assert client.get("/tasks/overview").json()["days"][1]["tasks"][0]["has_block"] is True
    r = client.post(f"/tasks/{task}/backlog")
    assert r.status_code == 200 and r.json()["task"]["has_block"] is False
    assert fake.written() == [("delete", fake_google.MILES, "milesblock")]
    assert db.get_task(task)["calendar_event_id"] is None


def test_backlog_never_touches_an_evallo_event(client, db, monkeypatch):
    fake = fake_google.install(monkeypatch)
    start, end = datetime(2026, 11, 26, 10), datetime(2026, 11, 26, 11)
    fake.add(fake_google.PRIMARY, "evallo1", "Evallo session", start, end)
    task = _add(db, "Tutoring", scheduled_for="2026-11-26")
    db.set_task_calendar_event(task, "evallo1", NOW)
    r = client.post(f"/tasks/{task}/backlog")
    assert r.status_code == 200
    assert fake.written() == []
    assert "evallo1" in fake.events[fake_google.PRIMARY]
    assert db.get_task(task)["scheduled_for"] is None


def test_backlog_google_failure_leaves_the_task_alone(client, db):
    task = _add(db, "Study", scheduled_for="2026-11-26")
    db.set_task_calendar_event(task, "milesblock", NOW)
    r = client.post(f"/tasks/{task}/backlog")   # the autouse fixture refuses Google
    assert r.status_code == 502
    assert db.get_task(task)["scheduled_for"] == "2026-11-26"

# ── review follow ups: backlog during an open review, done tasks, a 410 on delete ──

def test_backlog_route_prunes_the_open_review_and_confirm_does_not_bring_it_back(client, db, world):
    fixed_night(db, world)
    session = _session()
    study_id = db.session_plan(session["id"])["must_dos"][0]["task_id"]
    db.save_day_plan(TOMORROW, [{"task_id": study_id, "text": "x", "must_do": True}], REVIEW_NOW)
    r = client.post(f"/tasks/{study_id}/backlog")
    assert r.status_code == 200
    plan = db.session_plan(session["id"])
    assert study_id not in [e.get("task_id") for e in plan["must_dos"] + plan["optional"]]
    token, _ = _in_turn(session["id"])
    try:
        nr.confirm_plan(now=REVIEW_NOW)
    finally:
        session_types.end_turn(token)
    task = db.get_task(study_id)
    assert (task["scheduled_for"], task["calendar_event_id"]) == (None, None)
    assert world.inserts == [] or all(b["summary"] != "Linear algebra study" for _, b in world.inserts)


def test_confirm_skips_a_planned_task_that_is_already_done(db, world):
    fixed_night(db, world)
    session = _session()
    study_id = db.session_plan(session["id"])["must_dos"][0]["task_id"]
    db.complete_task(study_id, REVIEW_NOW)
    token, _ = _in_turn(session["id"])
    try:
        nr.confirm_plan(now=REVIEW_NOW)
    finally:
        session_types.end_turn(token)
    task = db.get_task(study_id)
    assert (task["status"], task["calendar_event_id"]) == ("done", None)
    assert all(b["summary"] != "Linear algebra study" for _, b in world.inserts)


def test_a_410_on_delete_counts_as_already_gone(client, db, monkeypatch):
    fake = fake_google.install(monkeypatch)
    fake.add(fake_google.MILES, "milesblock", "Study", datetime(2026, 11, 26, 10),
             datetime(2026, 11, 26, 11))
    task = _add(db, "Study", scheduled_for="2026-11-26")
    db.set_task_calendar_event(task, "milesblock", NOW)
    real = fake.events_

    def racing():
        ev = real()
        def delete(calendarId, eventId):
            raise fake_google.http_error(410)
        ev.delete = delete
        return ev
    monkeypatch.setattr(fake, "events_", racing)
    r = client.post(f"/tasks/{task}/backlog")
    assert r.status_code == 200
    assert db.get_task(task)["scheduled_for"] is None


def test_a_500_on_delete_is_still_502(client, db, monkeypatch):
    fake = fake_google.install(monkeypatch)
    fake.add(fake_google.MILES, "milesblock", "Study", datetime(2026, 11, 26, 10),
             datetime(2026, 11, 26, 11))
    task = _add(db, "Study", scheduled_for="2026-11-26")
    db.set_task_calendar_event(task, "milesblock", NOW)
    real = fake.events_

    def failing():
        ev = real()
        def delete(calendarId, eventId):
            raise fake_google.http_error(500)
        ev.delete = delete
        return ev
    monkeypatch.setattr(fake, "events_", failing)
    assert client.post(f"/tasks/{task}/backlog").status_code == 502


def _task_state(db):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        return tuple(conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                     for table in ("tasks", "task_completions"))


@pytest.mark.parametrize("action", ["complete", "reopen"])
@pytest.mark.parametrize("hour", [0, 3])
def test_guard_uses_previous_task_day_at_midnight_and_until_four(client, db, clock, action, hour):
    clock["now"] = datetime(2026, 11, 26, hour, 0)
    task = _add(db, "Daily", recurrence="daily")
    if action == "reopen":
        db.complete_task(task, NOW)
    response = client.post(f"/tasks/{task}/{action}", json={"day": "2026-11-25"})
    assert response.status_code == 200
    assert response.json()["task"]["status"] == ("done" if action == "complete" else "pending")
    assert db.task_completed_on(task, "2026-11-25") == (action == "complete")
    assert not db.task_completed_on(task, "2026-11-26")


@pytest.mark.parametrize("action", ["complete", "reopen"])
def test_guard_captures_clock_once_when_request_crosses_four(client, db, monkeypatch, action):
    task = _add(db, "Daily", recurrence="daily")
    if action == "reopen":
        db.complete_task(task, NOW)
    calls = []

    def clock_crosses_four():
        instant = datetime(2026, 11, 26, 3, 59, 59) if not calls else datetime(2026, 11, 26, 4)
        calls.append(instant)
        return instant

    monkeypatch.setattr(tasks_view, "now", clock_crosses_four)
    result = client.post(f"/tasks/{task}/{action}", json={"day": "2026-11-25"})
    assert len(calls) == 1
    assert result.status_code == 200
    if action == "complete":
        assert result.json()["day"] == "2026-11-25"
    assert result.json()["task"]["status"] == ("done" if action == "complete" else "pending")
    assert db.task_completed_on(task, "2026-11-25") == (action == "complete")
    assert not db.task_completed_on(task, "2026-11-26")


@pytest.mark.parametrize("action", ["complete", "reopen"])
@pytest.mark.parametrize("recurrence", [None, "daily"])
@pytest.mark.parametrize("day", ["2026-11-25", "2026-11-27"])
def test_reject_stale_and_future_dates_at_four_without_writes(client, db, clock, action, recurrence, day):
    clock["now"] = datetime(2026, 11, 26, 4)
    task = _add(db, "Sample", recurrence=recurrence)
    db.complete_task(task, NOW)
    db.complete_task(task, clock["now"])
    before = _task_state(db)
    response = client.post(f"/tasks/{task}/{action}", json={"day": day})
    assert response.status_code == 409
    assert response.json() == {
        "detail": "The day changed at 4 a.m. Nothing was changed.",
        "code": "day_changed", "today": "2026-11-26"}
    assert _task_state(db) == before


@pytest.mark.parametrize("action", ["complete", "reopen"])
@pytest.mark.parametrize("day", ["2026-11-25", "2026-11-24"])
def test_inactive_task_has_precedence_over_day_mismatch(client, db, action, day):
    task = _add(db, "Archived", recurrence="daily")
    db.complete_task(task, NOW)
    db.archive_task(task, NOW)
    before = _task_state(db)
    for task_id in (task, 9999):
        response = client.post(f"/tasks/{task_id}/{action}", json={"day": day})
        assert response.status_code == 404
    assert _task_state(db) == before


@pytest.mark.parametrize("action", ["complete", "reopen"])
@pytest.mark.parametrize("raw", [
    b"null", b"[]", b"1", b"true", b'"2026-11-25"', b" ", b'{"day":',
    b'{"day":null}', b'{"day":42}', b'{"day":"2026-02-30"}',
    b'{"day":"20261125"}', b'{"day":"2026-1-25"}', b'{"day":"2026-11-25T00:00"}',
    b'{"date":"2026-11-25"}', b'{"day":"2026-11-25","extra":true}',
    b'{"day":"2026-11-24","day":"2026-11-25"}', b"\xff",
])
def test_invalid_action_body_precedes_lookup_and_cannot_write(client, db, action, raw):
    task = _add(db, "Sample")
    db.complete_task(task, NOW)
    before = _task_state(db)
    for task_id in (task, 9999):
        response = client.post(f"/tasks/{task_id}/{action}", content=raw,
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    assert _task_state(db) == before


@pytest.mark.parametrize("action", ["complete", "reopen"])
@pytest.mark.parametrize("legacy", [{}, {"content": b""}, {"json": {}}])
def test_legacy_forms_target_server_day_and_log(client, db, clock, caplog, action, legacy):
    clock["now"] = datetime(2026, 11, 26, 4)
    task = _add(db, "Daily", recurrence="daily")
    db.complete_task(task, NOW)
    if action == "reopen":
        db.complete_task(task, clock["now"])
    with caplog.at_level("INFO", logger="uvicorn.error"):
        response = client.post(f"/tasks/{task}/{action}", **legacy)
    assert response.status_code == 200
    assert f"Legacy Tasks {action} for task {task}" in caplog.text
    assert db.task_completed_on(task, "2026-11-25")
    assert db.task_completed_on(task, "2026-11-26") == (action == "complete")


@pytest.mark.parametrize("recurrence", [None, "daily"])
def test_repeated_actions_leave_consistent_occurrences(client, db, recurrence):
    task = _add(db, "Sample", recurrence=recurrence)
    db.complete_task(task, datetime(2026, 11, 24, 12))
    body = {"day": "2026-11-25"}
    for _ in range(2):
        assert client.post(f"/tasks/{task}/complete", json=body).status_code == 200
    assert _task_state(db)[1] == [(task, "2026-11-24"), (task, "2026-11-25")]
    response = client.post(f"/tasks/{task}/reopen", json=body)
    assert response.status_code == 200 and response.json()["task"]["status"] == "pending"
    after = _task_state(db)
    assert after[1] == ([(task, "2026-11-24")] if recurrence else [])
    response = client.post(f"/tasks/{task}/reopen", json=body)
    assert response.status_code == 409 and response.json()["code"] == "not_done"
    assert _task_state(db) == after


@pytest.mark.parametrize("action", ["complete", "reopen"])
def test_guarded_retry_after_four_keeps_original_occurrence(client, db, clock, action):
    task = _add(db, "Daily", recurrence="daily")
    if action == "reopen":
        db.complete_task(task, NOW)
    body = {"day": "2026-11-25"}
    assert client.post(f"/tasks/{task}/{action}", json=body).status_code == 200
    clock["now"] = datetime(2026, 11, 26, 4)
    db.complete_task(task, clock["now"])
    before = _task_state(db)
    response = client.post(f"/tasks/{task}/{action}", json=body)
    assert response.status_code == 409 and response.json()["code"] == "day_changed"
    assert _task_state(db) == before


@pytest.mark.parametrize("action", ["complete", "reopen"])
def test_action_loses_to_concurrent_archive(client, db, monkeypatch, action):
    task = _add(db, "Daily", recurrence="daily")
    if action == "reopen":
        db.complete_task(task, NOW)
    writer_requested = Event()

    def observe_connection(*args, **kwargs):
        conn = sqlite3.connect(*args, **kwargs)

        def trace(statement):
            if statement == "BEGIN IMMEDIATE":
                writer_requested.set()

        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(db, "sqlite3", SimpleNamespace(connect=observe_connection, Row=sqlite3.Row))
    with ThreadPoolExecutor(max_workers=1) as pool, closing(sqlite3.connect(db.DB_PATH)) as other:
        other.execute("BEGIN IMMEDIATE")
        other.execute("UPDATE tasks SET archived_at = ? WHERE id = ?", (NOW.isoformat(), task))
        try:
            future = pool.submit(client.post, f"/tasks/{task}/{action}", json={"day": "2026-11-25"})
            assert writer_requested.wait(timeout=5)
        finally:
            other.commit()
        assert future.result(timeout=5).status_code == 404
    assert db.get_task(task)["archived_at"] == NOW.isoformat()
    assert _task_state(db)[1] == ([(task, "2026-11-25")] if action == "reopen" else [])


@pytest.mark.parametrize("recurrence", [None, "daily"])
@pytest.mark.parametrize("action", ["complete", "reopen"])
def test_competing_actions_serialize_mutation_and_response(client, db, monkeypatch, action, recurrence):
    task = _add(db, "Sample", recurrence=recurrence)
    if action == "reopen":
        db.complete_task(task, NOW)
    ready = Barrier(2)

    def clock():
        ready.wait(timeout=5)
        return NOW

    def submit():
        return client.post(f"/tasks/{task}/{action}", json={"day": "2026-11-25"})

    monkeypatch.setattr(tasks_view, "now", clock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        responses = [future.result(timeout=6) for future in futures]
    if action == "complete":
        assert [response.status_code for response in responses] == [200, 200]
        assert _task_state(db)[1] == [(task, "2026-11-25")]
    else:
        assert sorted(response.status_code for response in responses) == [200, 409]
        assert next(r for r in responses if r.status_code == 409).json()["code"] == "not_done"
        assert _task_state(db)[1] == []


@pytest.mark.parametrize("action", ["complete", "reopen"])
def test_guarded_actions_keep_full_login_requirement(db, clock, monkeypatch, action):
    import auth

    monkeypatch.setenv("MILES_JWT_SECRET", "synthetic-test-signing-value")
    client = TestClient(server.app)
    task = _add(db, "Sample")
    if action == "reopen":
        db.complete_task(task, NOW)
    before = _task_state(db)
    for scope in (auth.AGENT_SCOPE, auth.DEVICE_SCOPE):
        token = auth.create_token("test-user", scope=scope)
        assert client.post(f"/tasks/{task}/{action}", content=b"null",
                           headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert client.post(f"/tasks/{task}/{action}", content=b"null").status_code == 401
    assert _task_state(db) == before
    token = auth.create_token("test-user")
    assert client.post(f"/tasks/{task}/{action}", json={"day": "2026-11-25"},
                       headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_impossible_due_date_leaves_task_tables_unchanged(client, db):
    before = _task_state(db)
    response = client.post("/tasks", json={"title": "Synthetic", "where": "today", "due_date": "2026-02-30"})
    assert response.status_code == 422 and _task_state(db) == before
