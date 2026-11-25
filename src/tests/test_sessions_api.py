"""The five /sessions routes against docs/SESSIONS_CONTRACT.md: every JSON
shape, the status codes, and the turn stream's events in order."""

import json
from datetime import datetime
import os
import tempfile

import database

# server.py calls init_db() at import. Point the database somewhere throwaway
# first, so importing it here can never touch the real ~/miles/data/miles.db.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402
import sessions                                  # noqa: E402
from conftest import text_block, tool_block      # noqa: E402

SESSION_KEYS = {"id", "type", "title", "purpose", "status", "for_day", "created_at",
                "opened_at", "last_message_at", "closed_at", "summary"}
MESSAGE_KEYS = {"id", "role", "text", "created_at"}


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setattr(sessions, "effective_tier", lambda: "hokage")
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def _events(body):
    events, kind = [], None
    for line in body.splitlines():
        if line.startswith("event: "):
            kind = line[len("event: "):]
        elif line.startswith("data: "):
            events.append((kind, json.loads(line[len("data: "):])["text"]))
    return events


def _start(client):
    response = client.post("/sessions", json={"type": "nightly_review"})
    return response, response.json()


def test_every_route_needs_a_login():
    bare = TestClient(server.app)
    for method, path in (("get", "/sessions"), ("get", "/sessions/1"), ("post", "/sessions"),
                         ("post", "/sessions/1/turns"), ("post", "/sessions/1/close")):
        assert getattr(bare, method)(path).status_code in (401, 403)


def test_start_answers_201_with_the_opening(client):
    response, body = _start(client)
    assert response.status_code == 201
    assert set(body) == {"session", "messages"}
    session = body["session"]
    assert set(session) == SESSION_KEYS
    assert session["type"] == "nightly_review"
    assert session["purpose"] == "Close out today and plan tomorrow"
    assert session["status"] == "open"
    assert session["title"] is None and session["opened_at"] is None
    assert session["closed_at"] is None and session["summary"] is None
    assert session["for_day"] == database.task_day()
    [opening] = body["messages"]
    assert set(opening) == MESSAGE_KEYS
    assert opening["role"] == "assistant"
    assert opening["text"].startswith("Nothing was on your list for ")
    assert session["last_message_at"] == opening["created_at"]


def test_a_second_start_the_same_day_answers_200_with_the_first(client):
    _, first = _start(client)
    response, again = _start(client)
    assert response.status_code == 200
    assert again["session"]["id"] == first["session"]["id"]
    assert len(client.get("/sessions?status=all").json()["sessions"]) == 1


def test_an_unknown_type_is_400(client):
    response = client.post("/sessions", json={"type": "nonsense"})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_the_list_by_status(client):
    assert client.get("/sessions").json() == {"sessions": []}
    _, body = _start(client)
    listed = client.get("/sessions").json()["sessions"]
    assert [s["id"] for s in listed] == [body["session"]["id"]]
    assert set(listed[0]) == SESSION_KEYS
    assert client.get("/sessions?status=closed").json() == {"sessions": []}
    assert client.get("/sessions?status=missed").json() == {"sessions": []}
    assert len(client.get("/sessions?status=all").json()["sessions"]) == 1
    assert client.get("/sessions?status=bogus").status_code == 422


def test_get_marks_it_opened_and_shows_it(client):
    _, body = _start(client)
    session_id = body["session"]["id"]
    got = client.get(f"/sessions/{session_id}").json()
    assert set(got) == {"session", "messages"}
    assert got["session"]["opened_at"] is not None
    assert client.get(f"/sessions/{session_id}").json()["session"]["opened_at"] == \
        got["session"]["opened_at"]
    assert client.get("/sessions/999").status_code == 404


def test_a_turn_streams_the_chat_stream_events(client, fake_claude):
    _, body = _start(client)
    session_id = body["session"]["id"]
    fake_claude([text_block("One sec."), tool_block("get_tasks_for_date", {"date": "2026-11-24"})],
                [text_block("Nice work.")])
    response = client.post(f"/sessions/{session_id}/turns", json={"message": "did the lab"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _events(response.text) == [
        ("delta", "One "), ("delta", "sec."),
        ("stage", "Reading today's tasks"),
        ("reset", ""),
        ("delta", "Nice "), ("delta", "work."),
        ("done", "Nice work."),
    ]
    texts = [(m["role"], m["text"]) for m in client.get(f"/sessions/{session_id}").json()["messages"]]
    # "One sec." was reset live, so a reload hides it too (decided Nov 24 2026).
    assert texts[1:] == [("user", "did the lab"), ("assistant", "Nice work.")]


def test_closing_in_a_turn_sends_closed_before_done(client, fake_claude):
    _, body = _start(client)
    session_id = body["session"]["id"]
    fake_claude([text_block("Good night."), tool_block("close_session")],
                summary="He planned tomorrow.")
    events = _events(client.post(f"/sessions/{session_id}/turns",
                                 json={"message": "that's it"}).text)
    kinds = [k for k, _ in events]
    assert kinds[-2:] == ["closed", "done"]
    assert events[-1] == ("done", "Good night.")
    assert events[-2][1].endswith("He planned tomorrow.")
    assert client.get(f"/sessions/{session_id}").json()["session"]["status"] == "closed"


def test_a_failed_turn_ends_with_an_error(client, monkeypatch):
    _, body = _start(client)

    def broken():
        raise RuntimeError("Claude is unreachable")
    monkeypatch.setattr(sessions, "_client", broken)
    events = _events(client.post(f"/sessions/{body['session']['id']}/turns",
                                 json={"message": "hi"}).text)
    assert events == [("error", "Claude is unreachable")]


def test_turn_status_codes(client):
    assert client.post("/sessions/999/turns", json={"message": "hi"}).status_code == 404
    _, body = _start(client)
    session_id = body["session"]["id"]
    assert client.post(f"/sessions/{session_id}/turns", json={"message": ""}).status_code == 422
    assert client.post(f"/sessions/{session_id}/turns",
                       json={"message": "x" * 4001}).status_code == 422
    client.post(f"/sessions/{session_id}/close")
    response = client.post(f"/sessions/{session_id}/turns", json={"message": "hi"})
    assert response.status_code == 409
    assert "detail" in response.json()


def test_close_answers_the_closed_session(client):
    _, body = _start(client)
    session_id = body["session"]["id"]
    response = client.post(f"/sessions/{session_id}/close")
    assert response.status_code == 200
    closed = response.json()
    assert set(closed) == {"session"}
    assert set(closed["session"]) == SESSION_KEYS
    assert closed["session"]["status"] == "closed"
    assert closed["session"]["summary"].startswith("Today: 0 done")
    assert closed["session"]["closed_at"] is not None
    assert client.post(f"/sessions/{session_id}/close").status_code == 409
    assert client.post("/sessions/999/close").status_code == 404
    assert [s["id"] for s in client.get("/sessions?status=closed").json()["sessions"]] == [session_id]


def test_a_session_started_by_hand_is_never_pushed(client):
    # Started at 21:30:10, not yet fetched: the timer must still leave it alone.
    import session_schedule as sched
    from session_types import get_type
    _, body = _start(client)
    row = sessions.get(body["session"]["id"])
    assert row["push_sent_at"] is not None and row["reminder_sent_at"] is not None
    schedule = get_type("nightly_review").schedule
    for minute in (30, 35):
        assert sched.decide(datetime(2026, 11, 24, 21, minute), schedule,
                            {**row, "for_day": "2026-11-24", "opened_at": None}, False) is None


def test_purpose_is_never_null_for_an_unregistered_type(client, db):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("INSERT INTO sessions (type, created_at) VALUES ('retired_type', '2026-11-24T20:00:00')")
    conn.commit()
    conn.close()
    listed = client.get("/sessions").json()["sessions"]
    assert [s["purpose"] for s in listed if s["type"] == "retired_type"] == ["retired_type"]
