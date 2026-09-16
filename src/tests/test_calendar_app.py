"""The app's calendar screen: the week as data, and tap edits on the MILES
calendar only. Nothing here touches Google."""

import datetime
import os
import tempfile

import httplib2
import pytest
from googleapiclient.errors import HttpError

import calendar_tools as cal
import database

# server.py calls init_db() at import. Point it somewhere throwaway first.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402
from test_calendar_changes import FakeService, _at, _timed   # noqa: E402


class GettableService(FakeService):
    """Adds events().get, which the tap edits use to find an event by id."""

    def events(self):
        service, inner = self, super().events()

        class _WithGet:
            def get(self, calendarId, eventId):
                for event in service.events_by_calendar.get(calendarId, []):
                    if event["id"] == eventId:
                        return type("Call", (), {"execute": lambda self, e=event: e})()
                raise HttpError(httplib2.Response({"status": "404"}), b"{}")

            def __getattr__(self, name):
                return getattr(inner, name)
        return _WithGet()


@pytest.fixture
def google(monkeypatch):
    cal._calendar_cache.clear()
    service = GettableService(
        {
            "miles_id": [_timed("m1", "Charley lesson", _at(16), _at(17)),
                         _timed("m2", "Gym", _at(18), _at(19), recurringEventId="series")],
            "primary": [_timed("p1", "Class", _at(9), _at(10))],
            "club": [{"id": "c1", "summary": "Info session",
                      "start": {"date": "2026-09-14"}, "end": {"date": "2026-09-15"}}],
        },
        calendars=[
            {"id": "miles_id", "summary": "MILES", "selected": True, "accessRole": "owner"},
            {"id": "primary", "summary": "Lethanial", "primary": True, "accessRole": "owner"},
            {"id": "club", "summary": "UF IEEE", "selected": True, "accessRole": "reader"},
        ])
    monkeypatch.setattr(cal, "_service", lambda: service)
    yield service
    cal._calendar_cache.clear()


@pytest.fixture
def client():
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def test_the_week_is_data_and_only_miles_is_editable(google, client):
    week = client.get("/calendar/events?days=7").json()
    assert week["unreadable"] == []
    by_id = {e["id"]: e for e in week["events"]}
    assert [e["id"] for e in week["events"]] == ["c1", "p1", "m1", "m2"]
    assert by_id["m1"]["editable"] and by_id["m1"]["mine"]
    assert not by_id["p1"]["editable"] and by_id["p1"]["mine"]
    assert not by_id["c1"]["mine"] and by_id["c1"]["all_day"]
    assert by_id["c1"]["start"] == "2026-09-14"
    assert by_id["m2"]["repeating"]
    assert by_id["m1"]["start"] == _at(16)


def test_a_broken_calendar_is_named_not_hidden(google, client, monkeypatch):
    real = cal._fetch_events
    monkeypatch.setattr(cal, "_fetch_events", lambda queries: [
        (cid, None, RuntimeError("403")) if cid == "club" else (cid, items, err)
        for cid, items, err in real(queries)])
    assert client.get("/calendar/events").json()["unreadable"] == ["UF IEEE"]


def test_moving_by_tap_keeps_the_length(google, client):
    start = datetime.datetime(2026, 9, 14, 15, 0).astimezone().isoformat()
    response = client.patch("/calendar/events/m1", json={"start": start})
    assert response.json() == {"result": "Updated Charley lesson."}
    assert google.patched == [("miles_id", "m1", {
        "start": {"dateTime": _at(15)}, "end": {"dateTime": _at(16)}})]


def test_a_tap_rename_sends_only_the_title_as_typed(google, client):
    """Not capitalized the way a spoken title is: typed text is already his."""
    client.patch("/calendar/events/m1", json={"title": "  charley lesson (zoom) "})
    assert google.patched == [("miles_id", "m1", {"summary": "charley lesson (zoom)"})]


def test_an_event_on_another_calendar_cannot_be_changed(google, client):
    assert client.patch("/calendar/events/p1", json={"title": "x"}).status_code == 404
    assert client.delete("/calendar/events/p1").status_code == 404
    assert google.patched == [] and google.deleted == []


def test_an_end_before_the_start_is_refused(google, client):
    start = datetime.datetime(2026, 9, 14, 15, 0).astimezone().isoformat()
    end = datetime.datetime(2026, 9, 14, 14, 0).astimezone().isoformat()
    assert client.patch("/calendar/events/m1", json={"start": start, "end": end}).status_code == 400
    assert google.patched == []


def test_a_tap_delete_happens_at_once(google, client):
    assert client.delete("/calendar/events/m1").json() == {"result": "Deleted Charley lesson."}
    assert google.deleted == [("miles_id", "m1")]


def test_a_tap_edit_is_not_added_to_nova_s_undo(google, client):
    """Her undo groups by conversation turn; a tap is not one."""
    cal._recent_changes.clear()
    client.delete("/calendar/events/m1")
    assert cal._recent_changes == {}


def test_google_failing_reads_as_google_failing(client, monkeypatch):
    def expired():
        raise RuntimeError("invalid_grant: Token has been expired or revoked")
    monkeypatch.setattr(cal, "_service", expired)
    response = client.get("/calendar/events")
    assert response.status_code == 502
    assert "expired" in response.json()["detail"]
