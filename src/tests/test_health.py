"""The phone's Health data: what the app may send, and what Nova is told.

Two things here are easy to get wrong silently. A step count is only as fresh as
the last sync, so the summary has to say how old it is. And today's steps are a
half finished day, so they must stay out of the average or every morning's
figure is dragged down by a day that has barely started.
"""

import os
import tempfile
import time
from datetime import datetime

import database
import health_tools

# server.py calls init_db() at import. Point it somewhere throwaway first.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402

NOW = datetime(2026, 10, 30, 16, 0)


def _day(day, steps=None, kcal=None, weight=None):
    return {"day": day, "steps": steps, "active_kcal": kcal, "weight_lb": weight}


def _run(**extra):
    return {"id": "w1", "kind": "running", "started_at": "2026-10-28T07:10:00",
            "minutes": 31.6, "distance_mi": 3.104, "kcal": 312.4, **extra}


# ── the summary ──

def test_today_stays_out_of_the_average():
    days = [_day("2026-10-28", steps=9000), _day("2026-10-29", steps=11000),
            _day("2026-10-30", steps=800)]
    summary = health_tools.summarize_health(days, [], "2026-10-30T09:00:00", NOW, 7)
    assert summary["iphone_average_steps_per_finished_day"] == 10000
    assert summary["finished_days_averaged"] == 2
    assert summary["iphone_steps_today_so_far"] == 800


def test_the_age_of_the_sync_is_stated_and_old_data_is_flagged():
    fresh = health_tools.summarize_health([_day("2026-10-30", steps=800)], [],
                                          "2026-10-30T15:30:00", NOW, 7)
    assert fresh["last_synced_hours_ago"] == 0.5
    assert fresh["may_be_out_of_date"] is False

    stale = health_tools.summarize_health([_day("2026-10-30", steps=800)], [],
                                          "2026-10-30T09:00:00", NOW, 7)
    assert stale["last_synced_hours_ago"] == 7.0
    assert stale["may_be_out_of_date"] is True


def test_weight_change_is_computed_in_code_from_the_first_to_the_latest_weigh_in():
    days = [_day("2026-10-24", weight=185.0), _day("2026-10-27"), _day("2026-10-29", weight=183.6)]
    weight = health_tools.summarize_health(days, [], "2026-10-30T15:00:00", NOW, 7)["body_weight"]
    assert weight["latest_pounds"] == 183.6
    assert weight["weighed_on"] == "2026-10-29"
    assert weight["change_pounds_since_first_weigh_in_in_window"] == -1.4
    assert weight["first_weigh_in_in_window"] == "2026-10-24"


def test_one_weigh_in_has_no_change_to_report():
    weight = health_tools.summarize_health([_day("2026-10-29", weight=183.6)], [],
                                           "2026-10-30T15:00:00", NOW, 7)["body_weight"]
    assert weight["change_pounds_since_first_weigh_in_in_window"] is None


def test_a_run_is_worded_with_its_unit_in_the_field_name():
    summary = health_tools.summarize_health([], [_run()], "2026-10-30T15:00:00", NOW, 7)
    assert summary["workouts_newest_first"] == [{
        "when": "Wednesday October 28", "kind": "run", "minutes": 32,
        "miles": 3.1, "active_calories": 312.0}]


def test_no_weight_and_no_workouts_are_empty_not_invented():
    summary = health_tools.summarize_health([_day("2026-10-30", steps=100)], [],
                                            "2026-10-30T15:00:00", NOW, 7)
    assert summary["body_weight"] is None
    assert summary["workouts_newest_first"] == []


# ── the tool ──

def test_the_tool_says_so_when_nothing_has_ever_been_sent(db):
    answer = health_tools.get_health_summary()
    assert "No health data yet" in answer


def test_the_tool_reads_what_a_sync_stored(db, monkeypatch):
    today = datetime.now().date().isoformat()
    database.save_health_sync(
        [_day(today, steps=4200, kcal=310.0, weight=184.2)], [])
    summary = health_tools.get_health_summary(days=3)
    assert summary["iphone_steps_today_so_far"] == 4200
    assert summary["body_weight"]["latest_pounds"] == 184.2
    assert summary["may_be_out_of_date"] is False


# ── the endpoint ──

@pytest.fixture
def client(db):
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def test_a_sync_stores_days_and_workouts(client, db):
    body = {
        "days": [{"day": "2026-10-29", "steps": 11000, "active_kcal": 540.5, "weight_lb": 183.6},
                 {"day": "2026-10-30", "steps": 800}],
        "workouts": [{"id": "abc", "kind": "running", "started_at": "2026-10-28T07:10:00",
                      "minutes": 31.6, "distance_mi": 3.1, "kcal": 312.0}],
    }
    assert client.post("/health/sync", json=body).json() == {"days": 2, "workouts": 1}
    assert [d["steps"] for d in database.health_days_since("2026-10-29")] == [11000, 800]
    assert database.health_workouts_since("2026-10-01T00:00:00")[0]["id"] == "abc"


def test_syncing_again_replaces_a_day_instead_of_adding_to_it(client, db):
    day = {"day": "2026-10-29", "steps": 5000, "weight_lb": 190.0}
    client.post("/health/sync", json={"days": [day]})
    # He deleted the weigh in and walked more. The second copy is the truth.
    client.post("/health/sync", json={"days": [{"day": "2026-10-29", "steps": 6500}]})
    [row] = database.health_days_since("2026-10-29")
    assert row["steps"] == 6500
    assert row["weight_lb"] is None


def test_a_workout_sent_twice_is_one_row(client, db):
    run = {"id": "abc", "kind": "running", "started_at": "2026-10-28T07:10:00", "minutes": 30}
    client.post("/health/sync", json={"workouts": [run]})
    client.post("/health/sync", json={"workouts": [run]})
    assert len(database.health_workouts_since("2026-10-01T00:00:00")) == 1


@pytest.fixture
def new_york_time(monkeypatch):
    """The process clock in New York for one test. Changing TZ is not enough:
    the C library reads it only at tzset(), so the real zone has to be read
    back in afterwards or every later test runs in New York time. The session
    planner's tests found that out."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_an_instant_is_stored_as_the_pis_local_time(client, db, new_york_time):
    # 11:10 UTC on Oct 28 is 7:10 AM in New York (EDT).
    run = {"id": "abc", "kind": "running", "started_at": "2026-10-28T11:10:00Z", "minutes": 30}
    client.post("/health/sync", json={"workouts": [run]})
    [stored] = database.health_workouts_since("2026-10-01T00:00:00")
    assert stored["started_at"] == "2026-10-28T07:10:00"


@pytest.mark.parametrize("body", [
    {"days": [{"day": "2026-10-29", "weight_lb": 18300}]},
    {"days": [{"day": "2026-10-29", "steps": -5}]},
    {"days": [{"day": "tomorrow", "steps": 5}]},
    {"workouts": [{"id": "", "kind": "running", "started_at": "2026-10-28T07:10:00", "minutes": 5}]},
    {"workouts": [{"id": "a", "kind": "running", "started_at": "2026-10-28T07:10:00", "minutes": 5000}]},
    {"days": [{"day": "2026-10-29"}] * 61},
])
def test_an_implausible_payload_is_refused_whole(client, db, body):
    assert client.post("/health/sync", json=body).status_code == 422
    assert database.last_health_sync() is None
