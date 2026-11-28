"""Migration 048, skipped days, and the 047 receipt expiry. Synthetic data only."""

from contextlib import closing
from datetime import datetime, timedelta
import sqlite3

import pytest

import database
import tasks_view

NOON = datetime(2026, 11, 25, 12, 0)


def _version(conn):
    return conn.execute("SELECT version FROM schema_version").fetchone()[0]


def _at(tmp_path, monkeypatch, version):
    path = str(tmp_path / "skips.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS", [e for e in database.MIGRATIONS if e[0] <= version])
        database.init_db()
    return path


def _run(path):
    with closing(sqlite3.connect(path)) as conn:
        database._run_migrations(conn)


def test_head_is_50_and_registered_after_048():
    versions = [version for version, _ in database.MIGRATIONS]
    assert versions[-4:] == [47, 48, 49, 50]


def test_048_at_47_keeps_every_seeded_row_and_adds_only_the_table(tmp_path, monkeypatch):
    path = _at(tmp_path, monkeypatch, 47)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""INSERT INTO tasks (id, text, recurrence, created_at)
                        VALUES (1, 'Alex reading', 'daily', '2026-11-20')""")
        conn.execute("INSERT INTO task_completions VALUES (1, '2026-11-24')")
        conn.execute("""INSERT INTO chat_request_receipts
                        (actor, request_id, fingerprint, reply, created_at, state)
                        VALUES ('jordan', 'r1', 'f', 'ok', '2026-11-25T10:00:00', 'done')""")
        before = {t: conn.execute(f"SELECT * FROM {t}").fetchall()
                  for t in ("tasks", "task_completions", "chat_request_receipts")}
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    _run(path)
    _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 50
        after = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert after - tables == {"task_skips"}
        for table, rows in before.items():
            assert conn.execute(f"SELECT * FROM {table}").fetchall() == rows
        assert conn.execute("SELECT count(*) FROM task_skips").fetchone() == (0,)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_048_rolls_back_on_injected_failure_and_retries(tmp_path, monkeypatch):
    path = _at(tmp_path, monkeypatch, 47)
    real = database._migration_048_task_skips

    def fail_after_ddl(conn):
        real(conn)
        raise sqlite3.OperationalError("synthetic failure before version stamp")

    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS",
                      [e for e in database.MIGRATIONS if e[0] < 48] + [(48, fail_after_ddl)])
        with pytest.raises(sqlite3.OperationalError, match="synthetic failure"):
            _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 47
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'task_skips'").fetchone() is None
    _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 50


# ── the receipt table as the earlier 047 created it ──

OLD_047 = """CREATE TABLE chat_request_receipts (
    actor TEXT NOT NULL, request_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
    reply TEXT, PRIMARY KEY (actor, request_id))"""


def test_049_repairs_a_database_that_applied_the_original_047(tmp_path, monkeypatch):
    path = _at(tmp_path, monkeypatch, 48)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DROP TABLE chat_request_receipts")
        conn.execute(OLD_047)
        conn.execute("INSERT INTO chat_request_receipts VALUES ('jordan', 'old-done', 'f', 'ok')")
        conn.execute("INSERT INTO chat_request_receipts VALUES ('jordan', 'old-open', 'f', NULL)")
        conn.execute("UPDATE schema_version SET version = 48")
    _run(path)
    _run(path)
    monkeypatch.setattr(database, "DB_PATH", path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 50
        rows = {r[0]: r[1:] for r in conn.execute(
            "SELECT request_id, reply, state, created_at FROM chat_request_receipts")}
    # Identities are kept, and unknown times read as old.
    assert set(rows) == {"old-done", "old-open"}
    assert rows["old-done"][1] == "done" and rows["old-open"][1] == "pending"
    # Identified chat now works on that database, and the old ids are still refused.
    now = datetime.now()
    assert database.reserve_chat_request("jordan", "new", "f", now) is None
    assert database.reserve_chat_request("jordan", "old-done", "f", now) == ("f", None, "done")
    assert database.reserve_chat_request("jordan", "old-open", "f", now) == ("f", None, "failed")
    database.finish_chat_request("jordan", "new", "reply")
    assert database.reserve_chat_request("jordan", "new", "f", now) == ("f", "reply", "done")


# ── receipt deadlines apply at retry time, before any cleanup ──

def test_replay_honours_the_deadlines_without_running_cleanup(db):
    _receipt(db, "fresh", NOON - timedelta(hours=1), "kept", "done")
    _receipt(db, "stale-reply", NOON - timedelta(hours=25), "old", "done")
    _receipt(db, "stale-open", NOON - timedelta(hours=25), None, "pending")
    _receipt(db, "week-old", NOON - timedelta(days=7, seconds=1), "older", "done")
    assert db.reserve_chat_request("jordan", "fresh", "f", NOON) == ("f", "kept", "done")
    assert db.reserve_chat_request("jordan", "stale-reply", "f", NOON) == ("f", None, "done")
    assert db.reserve_chat_request("jordan", "stale-open", "f", NOON) == ("f", None, "failed")
    # Past seven days the id is free again, so a new reservation is made.
    assert db.reserve_chat_request("jordan", "week-old", "f", NOON) is None


# ── skip and completion cannot both succeed ──

def test_skip_checks_everything_in_one_transaction(db):
    daily = db.add_task("Reading", "daily", now=NOON)
    once = db.add_task("Call the bank", None, now=NOON)
    future = db.add_task("Stretch", "daily", now=NOON, start_date="2026-11-30")
    assert db.skip_task_if_due(once, "2026-11-25") == "not_daily"
    assert db.skip_task_if_due(future, "2026-11-25") == "not_started"
    assert db.skip_task_if_due(999, "2026-11-25") == "inactive"
    assert db.skip_task_if_due(daily, "2026-11-25") == "ok"
    assert db.skip_task_if_due(daily, "2026-11-25") == "already"
    other = db.add_task("Vitamins", "daily", now=NOON)
    db.complete_task(other, NOON)
    assert db.skip_task_if_due(other, "2026-11-25") == "done"
    assert not db.task_skipped_on(other, "2026-11-25")
    # The other order: a skipped day refuses completion.
    with pytest.raises(db.TaskSkipped):
        db.complete_task(daily, NOON)


# ── a skipped day is not shown and not due ──

@pytest.fixture
def tasks(db, monkeypatch):
    monkeypatch.setattr(tasks_view, "now", lambda: NOON)
    return db


def test_a_skipped_day_leaves_today_and_the_overview_but_only_today(tasks):
    daily = tasks.add_task("Reading", "daily", now=NOON)
    other = tasks.add_task("Stretch", "daily", now=NOON)
    assert tasks.skip_task_day(daily, "2026-11-25") is True
    assert tasks.skip_task_day(daily, "2026-11-25") is False
    assert [row[1] for row in tasks.tasks_today(NOON)] == ["Stretch"]
    overview = tasks_view.overview(days=1)
    assert [t["title"] for t in overview["days"][0]["tasks"]] == ["Stretch"]
    # Tomorrow it is back, and the task itself is not archived.
    assert [row[1] for row in tasks.tasks_today(NOON + timedelta(days=1))] == ["Reading", "Stretch"]
    assert tasks.get_task(daily)["archived_at"] is None
    assert other


def test_completing_a_skipped_day_is_refused_and_writes_nothing(tasks):
    daily = tasks.add_task("Reading", "daily", now=NOON)
    tasks.skip_task_day(daily, "2026-11-25")
    with pytest.raises(tasks_view.TaskError) as refused:
        tasks_view.apply_completion(daily, None, complete=True)
    assert refused.value.status == 409 and refused.value.code == "skipped"
    with pytest.raises(tasks.TaskSkipped):
        tasks.complete_task(daily, NOON)
    assert not tasks.task_completed_on(daily, "2026-11-25")


# ── 047 receipt expiry ──

def _receipt(db, request_id, created, reply=None, state="pending"):
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute("""INSERT INTO chat_request_receipts
                        (actor, request_id, fingerprint, reply, created_at, state)
                        VALUES ('jordan', ?, 'f', ?, ?, ?)""",
                     (request_id, reply, created.isoformat(), state))


def _rows(db):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        return {r[0]: r[1:] for r in conn.execute(
            "SELECT request_id, reply, state FROM chat_request_receipts")}


def test_reply_text_is_erased_only_after_twenty_four_hours(db):
    _receipt(db, "exact", NOON - timedelta(hours=24), "kept", "done")
    _receipt(db, "over", NOON - timedelta(hours=24, seconds=1), "gone", "done")
    _receipt(db, "fresh", NOON - timedelta(hours=1), "kept", "done")
    assert db.expire_chat_requests(NOON) == (1, 0, 0)
    rows = _rows(db)
    assert rows["exact"] == ("kept", "done") and rows["fresh"] == ("kept", "done")
    assert rows["over"] == (None, "done")


def test_unfinished_rows_fail_after_twenty_four_hours(db):
    _receipt(db, "exact", NOON - timedelta(hours=24))
    _receipt(db, "over", NOON - timedelta(hours=24, seconds=1))
    assert db.expire_chat_requests(NOON) == (0, 1, 0)
    assert _rows(db)["exact"][1] == "pending" and _rows(db)["over"][1] == "failed"


def test_rows_are_deleted_only_after_seven_days(db):
    _receipt(db, "exact", NOON - timedelta(days=7), "x", "done")
    _receipt(db, "over", NOON - timedelta(days=7, seconds=1), "x", "done")
    _receipt(db, "stuck", NOON - timedelta(days=8))
    assert db.expire_chat_requests(NOON)[2] == 2
    assert set(_rows(db)) == {"exact"}
    # A retry with a deleted id is a new request.
    assert db.reserve_chat_request("jordan", "over", "f", now=NOON) is None


def test_reserve_reports_state_and_finish_marks_done(db):
    assert db.reserve_chat_request("jordan", "a", "f", now=NOON) is None
    assert db.reserve_chat_request("jordan", "a", "f", now=NOON) == ("f", None, "pending")
    db.finish_chat_request("jordan", "a", "Hello")
    assert db.reserve_chat_request("jordan", "a", "f", now=NOON) == ("f", "Hello", "done")


def test_the_daily_tick_runs_cleanup_once_per_task_day(db, monkeypatch):
    import sessions
    _receipt(db, "old", NOON - timedelta(days=9), "x", "done")
    calls = []
    real = db.expire_chat_requests
    monkeypatch.setattr(db, "expire_chat_requests", lambda now=None: calls.append(now) or real(now))
    sessions.expire_chat_receipts_daily(NOON)
    sessions.expire_chat_receipts_daily(NOON + timedelta(hours=3))
    assert len(calls) == 1 and "old" not in _rows(db)
    sessions.expire_chat_receipts_daily(NOON + timedelta(days=1))
    assert len(calls) == 2


def test_a_retry_of_an_expired_or_failed_request_is_told_so_and_never_rerun(db, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from uuid import uuid4
    from fastapi.testclient import TestClient
    import brain
    import server

    monkeypatch.setitem(server.app.dependency_overrides, server.get_current_user, lambda: "fake-actor")
    monkeypatch.setattr(brain, "ask_nova", lambda *a, **k: pytest.fail("must not run again"))
    client = TestClient(server.app)
    body = {"message": "good morning", "channel": "text",
            "conversation_id": str(uuid4()), "request_id": str(uuid4())}
    request_id = server.ChatRequest(**body).request_id.hex
    fingerprint = hashlib.sha256(server.ChatRequest(**body).model_dump_json().encode()).hexdigest()
    assert db.reserve_chat_request("fake-actor", request_id, fingerprint, now=NOON - timedelta(days=2)) is None
    db.finish_chat_request("fake-actor", request_id, "Morning.")
    db.expire_chat_requests(NOON)
    expired = client.post("/chat", json=body)
    assert expired.status_code == 409 and "expired" in expired.json()["detail"]

    other = {**body, "request_id": str(uuid4())}
    other_id = server.ChatRequest(**other).request_id.hex
    other_fp = hashlib.sha256(server.ChatRequest(**other).model_dump_json().encode()).hexdigest()
    db.reserve_chat_request("fake-actor", other_id, other_fp, now=NOON - timedelta(days=2))
    db.expire_chat_requests(NOON)
    failed = client.post("/chat", json=other)
    assert failed.status_code == 409 and "outcome is unknown" in failed.json()["detail"]
    # It must not invite a blind retry of something that may have been done.
    assert "Check" in failed.json()["detail"] and "Ask again" not in failed.json()["detail"]


def test_production_websocket_runs_the_turn_off_the_event_loop(db, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    import server
    from auth import create_token

    monkeypatch.setattr(server, "headless", lambda: False)
    seen = {}

    def turn(body, user, **transport):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return SimpleNamespace(text="fake reply")

    monkeypatch.setattr(server, "_app_chat_turn", turn)
    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"token": create_token("Alex")})
        assert ws.receive_json() == {"status": "authenticated"}
        ws.send_json({"message": "hello", "channel": "text"})
        assert ws.receive_json() == {"response": "fake reply"}
    assert seen == {"on_loop": False}
