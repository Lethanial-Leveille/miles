"""/chat/stream sends the turn as it is written, not after it finishes.

Sep 14 2026: the app could only show a thinking indicator until the whole reply
came back, and on a text turn that wait included speaking it aloud."""

import json
import os
import tempfile

import database

# server.py calls init_db() at import. Point the database somewhere throwaway
# first, so importing it here can never touch the real ~/miles/data/miles.db.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402
from brain import TurnResult                     # noqa: E402


@pytest.fixture
def client():
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def _events(body):
    """The (kind, text) pairs in an SSE body, ignoring keep alive comments."""
    events, kind = [], None
    for line in body.splitlines():
        if line.startswith("event: "):
            kind = line[len("event: "):]
        elif line.startswith("data: "):
            events.append((kind, json.loads(line[len("data: "):])["text"]))
    return events


def test_the_pieces_arrive_then_the_finished_reply(client, monkeypatch):
    def fake_turn(message, device=None, channel=None, on_text=None):
        on_text("delta", "Six hours")
        on_text("delta", " and twenty minutes.")
        return TurnResult(text="Six hours and twenty minutes.")

    monkeypatch.setattr(server, "ask_nova", fake_turn)
    response = client.post("/chat/stream", json={"message": "how was my sleep",
                                                 "channel": "text"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _events(response.text) == [
        ("delta", "Six hours"),
        ("delta", " and twenty minutes."),
        ("done", "Six hours and twenty minutes."),
    ]


def test_the_channel_reaches_the_turn(client, monkeypatch):
    """Without this the app would stream a reply and still hear it spoken."""
    seen = {}

    def fake_turn(message, device=None, channel=None, on_text=None):
        seen["channel"] = channel
        return TurnResult(text="Done.")

    monkeypatch.setattr(server, "ask_nova", fake_turn)
    client.post("/chat/stream", json={"message": "hi", "channel": "text"})
    assert seen["channel"] == "text"


def test_a_failed_turn_ends_the_stream_with_an_error(client, monkeypatch):
    def fake_turn(message, device=None, channel=None, on_text=None):
        raise RuntimeError("Claude is unreachable")

    monkeypatch.setattr(server, "ask_nova", fake_turn)
    response = client.post("/chat/stream", json={"message": "hi", "channel": "text"})
    [(kind, text)] = _events(response.text)
    assert kind == "error"
    assert "unreachable" in text


def test_a_newline_in_the_reply_does_not_end_the_event(client, monkeypatch):
    """The blank line is what ends an SSE event, so the text has to be encoded."""
    def fake_turn(message, device=None, channel=None, on_text=None):
        on_text("delta", "One.\n\nTwo.")
        return TurnResult(text="One.\n\nTwo.")

    monkeypatch.setattr(server, "ask_nova", fake_turn)
    response = client.post("/chat/stream", json={"message": "hi", "channel": "text"})
    assert _events(response.text) == [("delta", "One.\n\nTwo."),
                                      ("done", "One.\n\nTwo.")]
