"""Reminders and system status, as the app sees them. Sep 16 2026: both
existed only by voice."""

import os
import sqlite3
import tempfile
from datetime import datetime

import database

# server.py calls init_db() at import. Point it somewhere throwaway first.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402


@pytest.fixture
def client(db):
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def _reminder(content, due_at, completed=0):
    conn = sqlite3.connect(database.DB_PATH)
    cursor = conn.execute(
        "INSERT INTO reminders (content, due_at, completed, created_at) VALUES (?, ?, ?, ?)",
        (content, due_at, completed, datetime(2026, 9, 16, 8).isoformat()))
    conn.commit()
    conn.close()
    return cursor.lastrowid


def test_reminders_are_listed_soonest_first_with_notes_last(client):
    _reminder("call mom", "2026-09-17T18:00:00")
    _reminder("buy tape", None)
    _reminder("submit lab", "2026-09-16T21:00:00")
    _reminder("already said", "2026-09-15T09:00:00", completed=1)
    listed = client.get("/reminders").json()
    assert [r["content"] for r in listed] == ["submit lab", "call mom", "buy tape"]
    assert {r["kind"] for r in listed} == {"reminder"}


def test_cancelling_one_reminder_leaves_the_rest(client):
    keep = _reminder("call mom", "2026-09-17T18:00:00")
    drop = _reminder("call mom", "2026-09-18T18:00:00")
    assert client.delete(f"/reminders/{drop}").json() == {"cancelled": True}
    assert [r["id"] for r in client.get("/reminders").json()] == [keep]


def test_a_delivered_reminder_cannot_be_cancelled(client):
    done = _reminder("already said", "2026-09-15T09:00:00", completed=1)
    assert client.delete(f"/reminders/{done}").status_code == 404


def test_status_details_adds_services_to_what_nova_knows(client, monkeypatch):
    monkeypatch.setattr(server, "get_system_state", lambda: {"uptime_hours": 3.5})
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: type(
        "Done", (), {"stdout": "active\nactive\nfailed\n"})())
    assert client.get("/status/details").json() == {
        "uptime_hours": 3.5,
        "services": {"miles-voice": True, "miles-server": True, "miles-tunnel": False},
    }


def test_status_details_survive_systemctl_failing(client, monkeypatch):
    monkeypatch.setattr(server, "get_system_state", lambda: {})

    def broken(*a, **k):
        raise FileNotFoundError("systemctl")
    monkeypatch.setattr(server.subprocess, "run", broken)
    assert client.get("/status/details").json()["services"] == {
        "miles-voice": None, "miles-server": None, "miles-tunnel": None}


def test_a_timer_is_listed_and_cancelled_like_a_reminder(client, monkeypatch):
    import actions
    monkeypatch.setattr(actions, "DB_PATH", database.DB_PATH)
    actions.set_timer("10 minutes")
    timer, = client.get("/reminders").json()
    assert (timer["content"], timer["kind"]) == ("10 minute timer", "timer")
    assert client.delete(f"/reminders/{timer['id']}").json() == {"cancelled": True}
