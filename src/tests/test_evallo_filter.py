"""The Evallo session id filter (docs/EVALLO_FILTER_CONTRACT.md, C1 to C7).
Every database here is a temporary one, and Google is the fake."""

import datetime
import os
import tempfile
from zoneinfo import ZoneInfo

import pytest

import database

database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

from fastapi.testclient import TestClient        # noqa: E402

import auth                                      # noqa: E402
import calendar_tools as cal                     # noqa: E402
import evallo_filter as ef                       # noqa: E402
import nightly_review as nr                      # noqa: E402
import server                                    # noqa: E402
from tests import fake_google                    # noqa: E402

UTC = datetime.timezone.utc
MINE, OTHER, NEW, FRESH2 = "a" * 24, "b" * 24, "c" * 24, "d" * 24
F = datetime.datetime(2026, 11, 25, 18, 0, 0, tzinfo=UTC)
WINDOW = (datetime.date(2026, 11, 18), datetime.date(2027, 1, 24))


def iso(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def state(ids=(MINE,), fetched=F, window=WINDOW, accepted=True):
    return ef.State(ids=frozenset(ids), fetched_at=fetched, fetched_raw=iso(fetched),
                    window=window, accepted=accepted)


def ev(sid=OTHER, title="Kid <> Someone Else | Math", updated=None, start=None, **extra):
    private = {"evalloEvent": "true"}
    if sid is not None:
        private["evalloSessionId"] = sid
    event = {"id": "g" + str(sid), "summary": title,
             "start": {"dateTime": start or "2026-11-30T15:00:00-05:00"},
             "end": {"dateTime": "2026-11-30T16:00:00-05:00"},
             "extendedProperties": {"private": private},
             "updated": updated if updated is not None else iso(F - datetime.timedelta(days=3))}
    event.update(extra)
    return event


NOW = F + datetime.timedelta(hours=1)


# ── 1. each row of C4 ──

def test_row_by_row():
    s = state()
    assert ef.decide({"id": "x", "summary": "Class"}, s, NOW) == (True, 0)
    assert ef.decide(ev(sid=None), s, NOW) == (True, 1)
    assert ef.decide(ev(sid=MINE), s, NOW) == (True, 2)
    assert ef.decide(ev(updated=iso(F)), s, NOW) == (True, 3)
    assert ef.decide(ev(), s, NOW) == (False, 4)
    late = NOW + datetime.timedelta(days=2)
    assert ef.decide(ev(), s, late) == (False, 6)
    named = ev(title="Kid <> Lethanial Leveille | Math")
    assert ef.decide(named, s, late) == (True, 5)
    assert ef.decide(named, s, NOW) == (False, 4)       # fresh and inside the window


def test_a_listed_id_is_kept_even_stale_and_outside_the_window():
    s = state()
    out = ev(sid=MINE, start="2030-01-01T10:00:00-05:00")
    assert ef.keep(out, s, F + datetime.timedelta(days=9))


def test_the_24_hour_boundary():
    s = state()
    day = datetime.timedelta(hours=24)
    assert ef.decide(ev(), s, F + day) == (False, 4)    # exactly 24h is still fresh
    assert ef.decide(ev(), s, F + day + datetime.timedelta(seconds=1)) == (False, 6)
    named = ev(title="Kid / Lethanial Leveille")
    assert ef.decide(named, s, F + day + datetime.timedelta(seconds=1)) == (True, 5)


def test_window_edges():
    s = state()
    first = ev(start="2026-11-18T00:30:00-05:00")
    assert ef.decide(first, s, NOW) == (False, 4)
    before = ev(start="2026-11-17T23:30:00-05:00")
    assert ef.decide(before, s, NOW) == (False, 6)
    last_day = ev(start="2027-01-23T23:30:00-05:00")
    assert ef.decide(last_day, s, NOW) == (False, 4)
    end_day = ev(start="2027-01-24T00:30:00-05:00")     # end is exclusive
    assert ef.decide(end_day, s, NOW) == (False, 6)
    named_out = ev(title="Kid <> Lethanial Leveille", start="2027-02-01T10:00:00-05:00")
    assert ef.decide(named_out, s, NOW) == (True, 5)    # fresh set, outside the window


def test_the_fifteen_minute_cutoff():
    s = state()
    cutoff = F - datetime.timedelta(minutes=15)
    sec = datetime.timedelta(seconds=1)
    assert ef.decide(ev(updated=iso(cutoff)), s, NOW)[0] is False
    assert ef.decide(ev(updated=iso(cutoff + sec)), s, NOW) == (True, 3)
    assert ef.decide(ev(updated=iso(cutoff - sec)), s, NOW)[0] is False


# ── 2. booked after the last push ──

@pytest.mark.parametrize("later", [datetime.timedelta(hours=1), datetime.timedelta(days=3)])
def test_a_session_booked_after_the_push_is_kept_fresh_or_stale(later):
    s = state()
    assert ef.keep(ev(updated=iso(F + datetime.timedelta(minutes=30))), s, F + later)


def test_evallos_clock_ahead_of_googles():
    s = state()
    assert ef.keep(ev(updated=iso(F - datetime.timedelta(minutes=14))), s, NOW)
    assert not ef.keep(ev(updated=iso(F - datetime.timedelta(minutes=16))), s, NOW)


def test_unlisted_old_event_with_full_name_inside_window_is_dropped_when_fresh():
    named = ev(title="Kid <> Lethanial Leveille | Math")
    assert not ef.keep(named, state(), NOW)
    stale_now = F + datetime.timedelta(days=2)
    assert ef.keep(named, state(), stale_now)
    assert not ef.keep(ev(), state(), stale_now)


# ── 3. the name rule ──

@pytest.mark.parametrize("title,matches", [
    ("Taylor <> Lethanial Leveille | Calculus", True),
    ("taylor <> lethanial leveille | calculus", True),
    ("Taylor / Lethanial Leveille", True),
    ("Taylor <> Lethanial | Calculus", False),
    ("Test Prep | Lethanial <> X", False),
    ("Lethanial Leveille <> Taylor | Calculus", False),
    ("Lethanial Leveille / Taylor", False),
    ("Taylor <> Lethanial Leveilleson | x", False),
    (None, False),
])
def test_name_rule(title, matches):
    event = ev()
    if title is None:
        del event["summary"]
    else:
        event["summary"] = title
    assert ef.has_tutor_name(event) is matches


# ── 4. ids and dates ──

@pytest.mark.parametrize("sid", [None, "", "XYZ", "A" * 24, "a" * 23, "a" * 25, 12])
def test_a_missing_or_malformed_id_is_kept(sid):
    event = ev(sid=None)
    if sid is not None:
        event["extendedProperties"]["private"]["evalloSessionId"] = sid
    assert ef.decide(event, state(), NOW) == (True, 1)


def test_a_timed_event_uses_its_new_york_date():
    # 00:30 UTC on Nov 18 is 7:30 PM on Nov 17 in New York: outside the window.
    event = ev(start="2026-11-18T00:30:00Z")
    assert ef.start_date(event) == datetime.date(2026, 11, 17)
    assert ef.decide(event, state(), NOW) == (False, 6)
    assert ef.start_date(ev(start="2026-11-18T05:30:00Z")) == datetime.date(2026, 11, 18)


def test_all_day_uses_start_date_and_unusable_starts_never_take_row_4():
    allday = ev()
    allday["start"] = {"date": "2026-11-20"}
    assert ef.decide(allday, state(), NOW) == (False, 4)
    for start in ({}, {"dateTime": "garbage"}, {"date": "nope"}, None):
        event = ev()
        event["start"] = start
        assert ef.decide(event, state(), NOW)[1] == 6
        event["summary"] = "Kid <> Lethanial Leveille"
        assert ef.decide(event, state(), NOW) == (True, 5)
    del event["start"]
    assert ef.decide(event, state(), NOW) == (True, 5)


def test_a_missing_or_bad_updated_is_kept():
    missing = ev()
    del missing["updated"]
    assert ef.decide(missing, state(), NOW) == (True, 3)
    assert ef.decide(ev(updated="not a time"), state(), NOW) == (True, 3)
    assert ef.decide(ev(updated="2026-11-01T00:00:00"), state(), NOW) == (True, 3)   # no zone


# ── storing pushes, the flag and restarts (5, 6) ──

def push(db, ids=(MINE,), fetched=None, window=WINDOW, now=None):
    return ef.store_snapshot(fetched or F, window[0], window[1], list(ids), now or NOW)


def test_no_status_row_keeps_everything(db):
    s = ef.load_state()
    assert s.readable and not s.on
    assert ef.status(s, NOW)["outcome"] == "off"
    assert ef.keep(ev(), s, NOW)


def test_state_read_failure_keeps_everything_and_says_unreadable(db, monkeypatch):
    push(db)
    monkeypatch.setattr(database, "DB_PATH", os.path.join(tempfile.mkdtemp(), "missing", "x.db"))
    s = ef.load_state()
    assert not s.readable
    assert ef.status(s, NOW)["outcome"] == "unreadable"
    assert ef.keep(ev(), s, NOW) and ef.keep(ev(sid=None), s, NOW)
    assert ef.keep({"id": "x"}, s, NOW)


def test_a_push_stores_ids_and_turns_the_filter_on(db):
    push(db, ids=[MINE, NEW])
    s = ef.load_state()
    assert s.on and s.ids == {MINE, NEW} and s.fetched_at == F and s.window == WINDOW
    assert ef.status(s, F + datetime.timedelta(hours=1)) == {"outcome": "active",
                                                            "verified_at": iso(F)}
    assert not ef.keep(ev(), s, NOW)


def test_restart_turns_it_off_until_a_push_is_accepted(db):
    push(db)
    assert ef.load_state().on
    ef.clear_active()                      # a restart: deploy, rollback, redeploy
    s = ef.load_state()
    assert not s.on and ef.status(s, NOW)["outcome"] == "off"
    assert ef.keep(ev(), s, NOW)                 # Mac offline: every Evallo event kept


def test_an_idempotent_retry_after_a_restart_turns_it_on(db):
    push(db)
    ef.clear_active()
    count, fetched, _ = push(db)
    assert (count, fetched) == (1, iso(F))
    assert ef.load_state().on


def test_a_25_hour_old_row_after_a_restart_is_off_not_stale(db):
    push(db, fetched=F, now=F)
    ef.clear_active()
    s = ef.load_state()
    late = F + datetime.timedelta(hours=25)
    assert ef.status(s, late)["outcome"] == "off"
    assert ef.keep(ev(title="Kid <> Someone"), s, late)
    push(db, fetched=F, now=F)                   # the Mac's retry turns it on again
    assert ef.status(ef.load_state(), late)["outcome"] == "stale"


def test_a_backward_clock_correction_cannot_turn_it_on(db):
    push(db)
    ef.clear_active()
    s = ef.load_state()
    assert ef.status(s, F - datetime.timedelta(hours=5))["outcome"] == "off"
    assert not s.on


def test_older_and_conflicting_snapshots_are_refused_and_change_nothing(db):
    push(db, ids=[MINE])
    newer = F + datetime.timedelta(minutes=10)
    push(db, ids=[NEW], fetched=newer)
    with pytest.raises(ef.Conflict) as old:
        push(db, ids=[MINE], fetched=F)               # a delayed retry of A after B
    assert old.value.reason == "older_snapshot"
    assert ef.load_state().ids == {NEW}
    with pytest.raises(ef.Conflict) as other:
        push(db, ids=[NEW, MINE], fetched=newer)
    assert other.value.reason == "conflicting_snapshot"
    with pytest.raises(ef.Conflict) as window:
        push(db, ids=[NEW], fetched=newer, window=(WINDOW[0], datetime.date(2027, 1, 25)))
    assert window.value.reason == "conflicting_snapshot"
    assert ef.load_state().ids == {NEW}


def test_a_failed_push_keeps_the_old_set(db, monkeypatch):
    push(db, ids=[MINE])
    real = ef.sqlite3.connect

    class Boom:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, *a):
            if sql.startswith("INSERT INTO evallo_verified_sessions"):
                raise RuntimeError("disk")
            return self.conn.execute(sql, *a)

        def executemany(self, sql, rows):
            raise RuntimeError("disk")

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(ef.sqlite3, "connect", lambda *a, **k: Boom(real(*a, **k)))
    with pytest.raises(RuntimeError):
        push(db, ids=[NEW], fetched=F + datetime.timedelta(minutes=5))
    monkeypatch.setattr(ef.sqlite3, "connect", real)
    s = ef.load_state()
    assert s.ids == {MINE} and s.fetched_at == F


# ── 7. through _fetch_events with the fake Google ──

def _event(fake, eid, hour, sid, title, updated, **extra):
    day = datetime.datetime(2026, 11, 30, hour)
    private = {"evalloEvent": "true"}
    if sid:
        private["evalloSessionId"] = sid
    fake.add(fake_google.PRIMARY, eid, title, day, day + datetime.timedelta(hours=1),
             extendedProperties={"private": private}, updated=updated, **extra)


@pytest.fixture
def calendar(db, monkeypatch):
    fake = fake_google.install(monkeypatch)
    # The state is fetched an hour ago on the real clock, so it is fresh.
    fetched = ef.now_utc().replace(microsecond=0) - datetime.timedelta(hours=1)
    ef.store_snapshot(fetched, datetime.date(2026, 11, 1), datetime.date(2027, 3, 1), [MINE])
    old, recent = iso(fetched - datetime.timedelta(days=3)), iso(fetched - datetime.timedelta(minutes=1))
    _event(fake, "mine", 10, MINE, "Kid <> Lethanial Leveille | Math", old)
    _event(fake, "theirs", 11, OTHER, "Kid <> Someone Else | Math", old)
    _event(fake, "overlap", 10, OTHER, "Kid2 <> Someone Else | Art", old)
    _event(fake, "booked", 14, NEW, "Kid3 <> Lethanial Leveille | New", recent)
    _event(fake, "free", 16, FRESH2, "Kid4 <> Lethanial Leveille | Free", recent,
           transparency="transparent")
    _event(fake, "declined", 17, "e" * 24, "Kid5 <> Lethanial Leveille | No", recent,
           attendees=[{"self": True, "responseStatus": "declined"}])
    return fake


DAY = datetime.datetime(2026, 11, 30)


def _busy_hours():
    blocks, _ = cal.busy_blocks(DAY, DAY + datetime.timedelta(days=1))
    return sorted((a.astimezone().hour, b.astimezone().hour) for a, b in blocks)


def test_other_tutors_are_gone_from_every_path_and_his_remain(calendar):
    assert _busy_hours() == [(10, 11), (14, 15)]       # free and declined are not busy
    ids = {e["id"] for e in cal.events_for_app(DAY, DAY + datetime.timedelta(days=1))["events"]}
    assert ids == {"mine", "booked", "free", "declined"}
    assert "evallo" not in cal.events_for_app(DAY, DAY + datetime.timedelta(days=1))
    out = cal.find_schedule_conflicts("2026-11-30", "2026-12-01", now=datetime.datetime(2026, 11, 29))
    assert "No overlaps" in out
    listing = cal.get_upcoming_events("2026-11-30", "2026-12-01", now=datetime.datetime(2026, 11, 29))
    assert "Kid3" in listing and "Kid4" in listing
    assert "Kid <> Someone Else" not in listing and "Kid2" not in listing
    assert "Evallo sessions" not in listing


def test_already_over_is_filtered_too(calendar):
    listing = cal.get_upcoming_events("2026-11-30", "2026-12-01",
                                      now=datetime.datetime(2026, 11, 30, 12, 30))
    assert "Already over" in listing
    assert "Kid <> Lethanial Leveille" in listing
    assert "Someone Else" not in listing


def test_without_a_push_in_this_process_everything_shows_with_the_note(calendar):
    ef.clear_active()
    ids = {e["id"] for e in cal.events_for_app(DAY, DAY + datetime.timedelta(days=1))["events"]}
    assert {"theirs", "overlap"} <= ids
    listing = cal.get_upcoming_events("2026-11-30", "2026-12-01", now=datetime.datetime(2026, 11, 29))
    assert listing.startswith("Evallo sessions not verified since ")


def test_reviewed_events_reports_the_outcome(calendar):
    assert cal.reviewed_events(DAY, DAY + datetime.timedelta(days=1))["evallo"]["outcome"] == "active"


# ── 8. the PUT route ──

@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setenv("MILES_JWT_SECRET", "test secret")
    server.app.dependency_overrides.clear()
    return TestClient(server.app)


def _h(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def agent(client):
    login = auth.create_token("Lethanial")
    return client.post("/auth/agent", headers=_h(login)).json()["access_token"]


def body(**kw):
    now = ef.now_utc()
    today = now.astimezone(ef.NY).date()
    b = {"fetched_at": iso(now - datetime.timedelta(minutes=1)),
         "window_start": (today - datetime.timedelta(days=7)).isoformat(),
         "window_end": (today + datetime.timedelta(days=60)).isoformat(),
         "session_ids": [MINE]}
    b.update(kw)
    return b


def put(client, bearer, **kw):
    return client.put("/agent/evallo/sessions", json=body(**kw), headers=_h(bearer))


def test_auth(client, agent):
    assert client.put("/agent/evallo/sessions", json=body()).status_code in (401, 403)
    assert client.put("/agent/evallo/sessions", json=body(), headers=_h("bad")).status_code == 401
    device = client.post("/auth/device", headers=_h(auth.create_token("Lethanial")))
    if device.status_code == 200:
        assert put(client, device.json()["access_token"]).status_code == 401


def test_validation_is_422(client, agent):
    now = ef.now_utc()
    today = now.astimezone(ef.NY).date()
    assert put(client, agent, token="secret").status_code == 422
    assert put(client, agent, session_ids=["XYZ"]).status_code == 422
    assert put(client, agent, session_ids=[MINE, MINE]).status_code == 422
    assert put(client, agent, session_ids=[f"{i:024x}" for i in range(2001)]).status_code == 422
    assert put(client, agent, session_ids=[f"{i:024x}" for i in range(2000)]).status_code == 200
    ws, we = (today + datetime.timedelta(days=1)).isoformat(), (today + datetime.timedelta(days=2)).isoformat()
    assert put(client, agent, window_start=ws, window_end=we).status_code == 422   # excludes today
    assert put(client, agent, window_start=we, window_end=ws).status_code == 422
    far = (today + datetime.timedelta(days=130)).isoformat()
    assert put(client, agent, window_end=far).status_code == 422                    # over 120 days
    assert put(client, agent, fetched_at="yesterday").status_code == 422
    assert put(client, agent, fetched_at="2026-11-25T18:00:00").status_code == 422  # no zone
    assert put(client, agent, fetched_at=iso(now - datetime.timedelta(hours=24, seconds=5))).status_code == 422


def test_fetched_at_ahead_bounds(client, agent, monkeypatch):
    fixed = ef.now_utc().replace(microsecond=0)
    monkeypatch.setattr(ef, "now_utc", lambda: fixed)
    ahead = lambda s: iso(fixed + datetime.timedelta(seconds=s))        # noqa: E731
    assert put(client, agent, fetched_at=ahead(5 * 60 + 1)).status_code == 422
    assert put(client, agent, fetched_at=ahead(4 * 60)).status_code == 200
    assert put(client, agent, fetched_at=ahead(5 * 60)).status_code == 200


def test_ordering_and_the_response(client, agent):
    now = ef.now_utc()
    a = iso(now - datetime.timedelta(minutes=30))
    b = iso(now - datetime.timedelta(minutes=10))
    r = put(client, agent, fetched_at=b, session_ids=[NEW])
    assert r.status_code == 200
    assert set(r.json()) == {"stored", "fetched_at", "pushed_at"} and r.json()["stored"] == 1
    late = put(client, agent, fetched_at=a, session_ids=[MINE])           # delayed retry of A
    assert late.status_code == 409 and late.json() == {"reason": "older_snapshot"}
    assert ef.load_state().ids == {NEW}
    same = put(client, agent, fetched_at=b, session_ids=[NEW])
    assert same.status_code == 200 and same.json()["fetched_at"] == b
    other = put(client, agent, fetched_at=b, session_ids=[NEW, MINE])
    assert other.status_code == 409 and other.json() == {"reason": "conflicting_snapshot"}
    assert ef.load_state().ids == {NEW}


def test_the_first_accepted_push_turns_the_filter_on(client, agent):
    assert not ef.load_state().on
    assert put(client, agent).status_code == 200
    assert ef.load_state().on


def test_a_delayed_delivery_is_accepted_with_evallos_time(client, agent):
    now = ef.now_utc().replace(microsecond=0)
    fetched = iso(now - datetime.timedelta(minutes=30))
    r = put(client, agent, fetched_at=fetched)
    assert r.status_code == 200 and r.json()["fetched_at"] == fetched
    s = ef.load_state()
    booked = ev(updated=iso(now - datetime.timedelta(minutes=20)))
    assert ef.decide(booked, s, now)[0] and ef.decide(booked, s, now + datetime.timedelta(days=3))[0]


def test_the_route_does_not_log_the_body(client, agent, capsys):
    put(client, agent, session_ids=[MINE])
    out = capsys.readouterr().out
    assert MINE not in out and "count=1" in out


# ── 9. the notes ──

def test_note_wording():
    assert ef.note({"outcome": "active", "verified_at": iso(F)}) is None
    assert ef.note({"outcome": "off", "verified_at": None}) == "Evallo sessions never verified"
    assert ef.note({"outcome": "unreadable", "verified_at": None}) == \
        "Evallo sessions could not be checked"
    # 18:00 UTC on Nov 25 2026 is 1:00 PM in New York.
    assert ef.note({"outcome": "stale", "verified_at": iso(F)}) == \
        "Evallo sessions not verified since Wed Nov 25, 1:00 PM"
    assert ef.worst([{"outcome": "stale"}, {"outcome": "unreadable"}, {"outcome": "off"}]) == \
        {"outcome": "unreadable"}


REVIEW_NOW = datetime.datetime(2026, 11, 30, 21, 30)


def test_the_review_opening_carries_the_note(calendar, monkeypatch):
    text, _ = nr.opening(REVIEW_NOW)
    assert "Evallo sessions" not in text                       # active
    ef.clear_active()
    text, _ = nr.opening(REVIEW_NOW)
    assert text.startswith("Evallo sessions not verified since ")
    monkeypatch.setattr(ef, "load_state", lambda: ef.State(readable=False))
    text, _ = nr.opening(REVIEW_NOW)
    assert text.startswith("Evallo sessions could not be checked.")


def test_a_stale_set_shows_the_stale_note(db, monkeypatch):
    fake_google.install(monkeypatch)
    old = ef.now_utc().replace(microsecond=0) - datetime.timedelta(hours=30)
    ef.store_snapshot(old, datetime.date(2026, 11, 1), datetime.date(2027, 3, 1), [MINE],
                      now=old)
    listing = cal.get_upcoming_events("2026-11-30", "2026-12-01", now=datetime.datetime(2026, 11, 29))
    assert listing.startswith("Evallo sessions not verified since ")


def test_a_voice_answer_from_a_failed_read_says_unreadable_even_if_a_second_read_works(
        calendar, monkeypatch):
    real = ef.load_state
    calls = []

    def flaky():
        calls.append(1)
        return ef.State(readable=False) if len(calls) == 1 else real()
    monkeypatch.setattr(ef, "load_state", flaky)
    listing = cal.get_upcoming_events("2026-11-30", "2026-12-01", now=datetime.datetime(2026, 11, 29))
    assert listing.startswith("Evallo sessions could not be checked.")
    assert len(calls) == 1                                       # one read for the whole answer


def test_the_evallo_keys_match_the_write_guard():
    assert ef.EVALLO_KEYS == cal.EVALLO_KEYS


# ── revision 5, group 1: active lives in the database ──

def _connect():
    import sqlite3
    return sqlite3.connect(database.DB_PATH)


def test_a_second_connection_sees_an_accepted_push(db):
    push(db)
    other = _connect()
    assert other.execute("SELECT active FROM evallo_verification").fetchone() == (1,)
    other.close()
    assert ef.load_state().on


def test_server_startup_clears_active_and_a_reader_restart_does_not(db):
    push(db)
    # a voice or review process starting up only reads
    assert ef.load_state().on and ef.load_state().on
    server._evallo_filter_off_until_a_push()           # miles-server's startup hook
    s = ef.load_state()
    assert not s.on and ef.status(s, NOW)["outcome"] == "off"
    push(db)                                           # the idempotent retry turns it on
    assert ef.load_state().on


def test_startup_clear_without_a_row_is_fine(db):
    ef.clear_active()


def test_a_count_that_does_not_match_the_ids_is_unreadable(db):
    push(db, ids=[MINE, NEW])
    conn = _connect()
    conn.execute("UPDATE evallo_verification SET count = 5")
    conn.commit()
    conn.close()
    s = ef.load_state()
    assert not s.readable and ef.status(s, NOW)["outcome"] == "unreadable"
    assert ef.keep(ev(), s, NOW)


def test_migration_045_is_atomic(tmp_path, monkeypatch):
    import sqlite3
    path = str(tmp_path / "m.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:44])
    database.init_db()
    full = list(database.MIGRATIONS) + [(45, database._migration_045_evallo_filter)]

    def half(conn):
        conn.execute("CREATE TABLE evallo_verified_sessions (session_id TEXT PRIMARY KEY)")
        raise RuntimeError("between the two tables")
    monkeypatch.setattr(database, "MIGRATIONS", database.MIGRATIONS + [(45, half)])
    with pytest.raises(RuntimeError):
        database.init_db()
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "evallo_verified_sessions" not in tables
    assert conn.execute("SELECT version FROM schema_version").fetchone() == (44,)
    conn.close()
    monkeypatch.setattr(database, "MIGRATIONS", full)
    database.init_db()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT version FROM schema_version").fetchone() == (45,)
    assert "active" in [r[1] for r in conn.execute("PRAGMA table_info(evallo_verification)")]
    conn.close()


# ── revision 5, group 2: the filter never raises and never drops a calendar ──

@pytest.mark.parametrize("extended", ["x", 7, ["private"], {"private": "x"}, {"private": [1]},
                                      {"private": None}])
def test_malformed_extended_properties_do_not_raise(extended):
    event = {"id": "m", "summary": "Class", "extendedProperties": extended}
    assert ef.decide(event, state(), NOW) == (True, 0)


def test_an_event_that_cannot_be_evaluated_is_kept_but_flagged(monkeypatch, capsys):
    def boom(event):
        raise RuntimeError("secret title " + event.get("summary", ""))
    monkeypatch.setattr(ef, "start_date", boom)
    assert ef.decide(ev(title="Private name"), state(), NOW) == (True, ef.ERROR_ROW)
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "Private name" not in out and "secret" not in out


def test_the_filter_cannot_make_a_calendar_disappear(calendar, monkeypatch):
    def boom(event):
        raise ValueError("bad event")
    monkeypatch.setattr(ef, "has_tutor_name", boom)
    monkeypatch.setattr(ef, "start_date", boom)
    queries = [{"calendarId": fake_google.PRIMARY, "timeMin": DAY.astimezone().isoformat(),
                "timeMax": (DAY + datetime.timedelta(days=1)).astimezone().isoformat()}]
    (cid, items, error), = cal._fetch_events(queries)
    assert error is None
    assert {e["id"] for e in items} == {"mine", "theirs", "overlap", "booked", "free", "declined"}


# ── revision 5, group 3: whole seconds, a top level 409, a harmless log ──

def test_fractional_fetched_at_is_ordered_by_whole_seconds(client, agent):
    r = put(client, agent, fetched_at=iso(ef.now_utc() - datetime.timedelta(minutes=2))[:-1]
            + ".900Z", session_ids=[MINE])
    assert r.status_code == 200
    stored = r.json()["fetched_at"]
    assert "." not in stored
    later_in_text_earlier_in_fraction = stored[:-1] + ".100Z"
    r2 = put(client, agent, fetched_at=later_in_text_earlier_in_fraction, session_ids=[])
    assert r2.status_code == 409 and r2.json() == {"reason": "conflicting_snapshot"}
    assert ef.load_state().ids == {MINE}                  # never replaced
    r3 = put(client, agent, fetched_at=later_in_text_earlier_in_fraction, session_ids=[MINE])
    assert r3.status_code == 200 and r3.json()["fetched_at"] == stored


def test_a_failing_log_write_after_the_commit_changes_nothing(client, agent, monkeypatch):
    def broken(*a, **k):
        raise OSError("log disk full")
    monkeypatch.setattr(server, "print", broken, raising=False)
    r = put(client, agent)
    assert r.status_code == 200 and r.json()["stored"] == 1
    assert ef.load_state().on                              # the answer and the state agree


def test_a_refused_push_never_turns_the_filter_on(client, agent):
    assert put(client, agent, session_ids=["bad"]).status_code == 422
    assert not ef.load_state().on
    assert put(client, agent).status_code == 200
    ef.clear_active()
    older = iso(ef.now_utc() - datetime.timedelta(hours=3))
    assert put(client, agent, fetched_at=older).status_code == 409
    assert not ef.load_state().on                          # a 409 does not activate


# ── revision 5, group 4: uncertain events come back anonymous ──

SECRET_NAME = "Maribel Quintero"
SECRET_BITS = (SECRET_NAME, "Organic Chem", "zoom.us/j/12345", "Room 204", "maribel@example.com")


def _secret_kwargs():
    return dict(description="Organic Chem with Zoom link https://zoom.us/j/12345",
                location="Room 204",
                attendees=[{"email": "maribel@example.com", "displayName": SECRET_NAME},
                           {"self": True, "responseStatus": "accepted", "email": "me@example.com"}],
                organizer={"displayName": SECRET_NAME}, creator={"email": "maribel@example.com"},
                conferenceData={"entryPoints": [{"uri": "https://zoom.us/j/12345"}]})


@pytest.fixture
def uncertain(calendar):
    fake = calendar
    fetched = ef.load_state().fetched_at
    recent = iso(fetched - datetime.timedelta(minutes=1))
    day = datetime.datetime(2026, 12, 1)
    fake.add(fake_google.PRIMARY, "u1", f"{SECRET_NAME} <> Other Tutor | Organic Chem",
             day.replace(hour=10), day.replace(hour=11),
             extendedProperties={"private": {"evalloEvent": "true", "note": SECRET_NAME}},
             updated=recent, **_secret_kwargs())                         # row 1, no id
    fake.add(fake_google.PRIMARY, "u3", f"{SECRET_NAME} <> Other Tutor | Chem",
             day.replace(hour=13), day.replace(hour=14),
             extendedProperties={"private": {"evalloEvent": "true", "evalloSessionId": "f" * 24}},
             updated=recent, **_secret_kwargs())                         # row 3, new
    kw = _secret_kwargs()
    kw["attendees"][1]["responseStatus"] = "declined"
    fake.add(fake_google.PRIMARY, "u4", f"{SECRET_NAME} <> Other Tutor | Declined",
             day.replace(hour=15), day.replace(hour=16),
             extendedProperties={"private": {"evalloEvent": "true", "evalloSessionId": "9" * 24}},
             updated=recent, **kw)                                       # row 3, he declined
    fake.add(fake_google.PRIMARY, "u5", "Kid <> Lethanial Leveille | Math",
             day.replace(hour=17), day.replace(hour=18),
             extendedProperties={"private": {"evalloEvent": "true", "evalloSessionId": "8" * 24}},
             updated=recent, description="His own notes")                # row 3 with his name
    return fake


D2 = datetime.datetime(2026, 12, 1)


def _clean(text):
    for bit in SECRET_BITS:
        assert bit not in text, bit


def test_uncertain_events_are_anonymous_on_the_app_and_review_paths(uncertain):
    import json
    window = cal.events_for_app(D2, D2 + datetime.timedelta(days=1))
    _clean(json.dumps(window))
    by_id = {e["id"]: e for e in window["events"]}
    for eid in ("u1", "u3", "u4"):
        assert by_id[eid]["title"] == ef.ANONYMOUS_TITLE
        assert not by_id[eid]["description"]
    assert by_id["u4"]["declined"] and not by_id["u1"]["declined"]
    assert by_id["u5"]["title"] == "Kid <> Lethanial Leveille | Math"     # his name: keeps data
    assert by_id["u5"]["description"] == "His own notes"
    reviewed = cal.reviewed_events(D2, D2 + datetime.timedelta(days=1))
    _clean(json.dumps(reviewed, default=str))


def test_anonymous_blocks_are_busy_unless_declined_or_free(uncertain):
    blocks, _ = cal.busy_blocks(D2, D2 + datetime.timedelta(days=1))
    assert sorted((a.astimezone().hour, b.astimezone().hour) for a, b in blocks) == \
        [(10, 11), (13, 14), (17, 18)]


def test_uncertain_events_are_anonymous_in_voice_and_conflicts(uncertain):
    now = datetime.datetime(2026, 11, 30, 9)
    listing = cal.get_upcoming_events("2026-12-01", "2026-12-02", now=now)
    _clean(listing)
    assert ef.ANONYMOUS_TITLE in listing
    over = cal.get_upcoming_events("2026-12-01", "2026-12-02",
                                   now=datetime.datetime(2026, 12, 1, 20))
    _clean(over)
    assert "Already over" in over and ef.ANONYMOUS_TITLE in over
    _clean(cal.find_schedule_conflicts("2026-12-01", "2026-12-02", now=now))


def test_anonymous_blocks_keep_what_the_write_guard_needs(uncertain):
    ids = {e["id"]: e for e in cal._fetch_events([{
        "calendarId": fake_google.PRIMARY, "timeMin": D2.astimezone().isoformat(),
        "timeMax": (D2 + datetime.timedelta(days=1)).astimezone().isoformat()}])[0][1]}
    block = ids["u3"]
    assert block["extendedProperties"]["private"] == {"evalloEvent": "true",
                                                      "evalloSessionId": "f" * 24}
    assert "description" not in block and "organizer" not in block and "location" not in block
    assert block["attendees"] == [{"self": True, "responseStatus": "accepted"}]
    assert ids["u1"]["extendedProperties"]["private"] == {"evalloEvent": "true"}


def test_row_2_and_row_5_keep_their_data():
    row2 = ev(sid=MINE, title="Maribel <> Other | Chem", description="notes")
    assert ef.view(row2, state(), NOW) == (row2, 2)
    row5 = ev(title="Kid <> Lethanial Leveille | Math", start="2027-03-01T10:00:00-05:00")
    assert ef.view(row5, state(), NOW) == (row5, 5)
    row3_named = ev(title="Kid <> Lethanial Leveille", updated=iso(F))
    assert ef.view(row3_named, state(), NOW)[0] is row3_named
    assert ef.view(ev(), state(), NOW) == (None, 4)


def test_nothing_anonymized_reaches_tool_call_log(uncertain, db):
    import sqlite3
    out = cal.get_upcoming_events("2026-12-01", "2026-12-02", now=datetime.datetime(2026, 11, 30, 9))
    database.log_tool_call("get_upcoming_events", {"time_min": "2026-12-01"}, out)
    conn = sqlite3.connect(database.DB_PATH)
    rows = " ".join(" ".join(map(str, r)) for r in conn.execute("SELECT * FROM tool_call_log"))
    conn.close()
    _clean(rows)
    assert ef.ANONYMOUS_TITLE in rows


def test_a_filter_that_is_off_does_not_anonymize(uncertain):
    ef.clear_active()
    window = cal.events_for_app(D2, D2 + datetime.timedelta(days=1))
    assert {e["title"] for e in window["events"]} >= {f"{SECRET_NAME} <> Other Tutor | Organic Chem"}


# ── revision 5, group 5: three inputs that used to drop a real session ──

@pytest.mark.parametrize("sid", [MINE + "\n", "\n" + MINE, MINE + " "])
def test_an_id_with_a_trailing_newline_is_not_usable(sid):
    event = ev(sid=None)
    event["extendedProperties"]["private"]["evalloSessionId"] = sid
    assert ef.session_id(event) is None
    assert ef.decide(event, state(), NOW) == (True, 1)


def test_the_push_refuses_an_id_with_a_trailing_newline(client, agent):
    r = put(client, agent, session_ids=[MINE + "\n"])
    assert r.status_code == 422 and MINE not in r.text


def test_an_event_in_another_zone_uses_its_new_york_date():
    # 00:30 in Tokyo on Nov 18 is 10:30 AM on Nov 17 in New York: outside the window.
    tokyo = ev(start="2026-11-18T00:30:00+09:00")
    assert ef.start_date(tokyo) == datetime.date(2026, 11, 17)
    assert ef.decide(tokyo, state(), NOW) == (False, 6)
    naive = ev()
    naive["start"] = {"dateTime": "2026-11-18T00:30:00", "timeZone": "Asia/Tokyo"}
    assert ef.start_date(naive) == datetime.date(2026, 11, 17)
    naive["start"] = {"dateTime": "2026-11-18T00:30:00", "timeZone": "America/New_York"}
    assert ef.start_date(naive) == datetime.date(2026, 11, 18)
    naive["start"] = {"dateTime": "2026-11-18T00:30:00", "timeZone": "Not/AZone"}
    assert ef.start_date(naive) is None                              # never guesses New York


@pytest.mark.parametrize("title", [
    "Taylor <> Lethanial\u00a0Leveille | Calculus",
    "Taylor <>  Lethanial \t Leveille | Calculus",
    "Taylor\u00a0/\u00a0Lethanial\u00a0\u00a0Leveille",
    "  Taylor <> LETHANIAL LEVEILLE",
])
def test_whitespace_runs_and_non_breaking_spaces_match_the_name(title):
    event = ev(title=title)
    assert ef.has_tutor_name(event)
    assert ef.decide(event, state(), NOW + datetime.timedelta(days=2)) == (True, 5)


# ── revision 5, group 6: direct lookups go through the filter too ──

def _miles_event(fake, eid, title, hour, private, day=datetime.datetime(2026, 12, 2), **extra):
    fake.add(fake_google.MILES, eid, title, day.replace(hour=hour), day.replace(hour=hour + 1),
             extendedProperties={"private": private}, updated=extra.pop("updated", None), **extra)


@pytest.fixture
def lookups(calendar):
    fetched = ef.load_state().fetched_at
    recent = iso(fetched - datetime.timedelta(minutes=1))
    old = iso(fetched - datetime.timedelta(days=3))
    # uncertain (row 1) with a name that must never be quoted, and one the filter drops (row 4)
    _miles_event(calendar, "anon", f"{SECRET_NAME} Chem", 9, {"evalloEvent": "true"},
                 updated=recent, **_secret_kwargs())
    _miles_event(calendar, "gone", f"{SECRET_NAME} Physics", 11,
                 {"evalloEvent": "true", "evalloSessionId": "7" * 24}, updated=old)
    _miles_event(calendar, "mine", "Gym", 13, {}, updated=old)
    return calendar


def test_failed_title_search_does_not_list_hidden_names(lookups):
    now = datetime.datetime(2026, 12, 1, 8)
    service = cal._service()
    with pytest.raises(cal.EventLookupError) as err:
        cal._find_miles_event(service, "Yoga", "2026-12-02", now)
    text = str(err.value)
    _clean(text)
    assert "Gym" in text and ef.ANONYMOUS_TITLE in text and "Physics" not in text
    with pytest.raises(cal.EventLookupError) as near:
        cal._find_miles_event(service, "Chem", "2026-12-03", now, nearby=True)
    _clean(str(near.value))


def test_rename_and_block_lookups_use_the_filter(lookups):
    now = datetime.datetime(2026, 12, 1, 8)
    with pytest.raises(cal.EventLookupError):
        cal.rename_calendar_events("Physics", "Math", "2026-12-02", "2026-12-03", now=now)
    with pytest.raises(cal.EventLookupError):
        cal.rename_calendar_events(SECRET_NAME, "X", "2026-12-02", "2026-12-03", now=now)
    block = cal.get_miles_event("anon")
    assert block["title"] == ef.ANONYMOUS_TITLE
    assert cal.get_miles_event("gone") is None
    assert cal.get_miles_event("mine")["title"] == "Gym"


def test_the_write_guard_refusal_never_repeats_a_hidden_title(lookups):
    for eid in ("anon", "gone"):
        with pytest.raises(cal.EvalloProtectedError) as err:
            cal._write_patch(fake_google.MILES, eid, {"summary": "x"})
        _clean(str(err.value))
        assert "Evallo tutoring session" in str(err.value)
    with pytest.raises(cal.EvalloProtectedError) as ok:
        cal._write_delete(fake_google.PRIMARY, "mine")
    assert "Kid <> Lethanial Leveille | Math" in str(ok.value)       # his own keeps its name
    assert lookups.written() == []


# ── the read only survivors script ──

def _survivors():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evallo_survivors", os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                                         "evallo_survivors.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_survivors_script_lists_id_updated_and_row(tmp_path):
    mod = _survivors()
    events = [{"id": "plain", "summary": "Class"}, ev(sid=MINE), ev(sid=None), ev(),
              ev(updated=iso(F)), ev(updated="Tuesday-ish"), ev(updated="")]
    lines = mod.survivor_lines(events, state(), NOW)
    assert [line.split()[-1] for line in lines] == ["2", "1", "3", "3", "3"]
    assert lines[0].startswith(MINE) and lines[1].startswith("none ")
    assert lines[3].split()[1] == "unparseable" and "Tuesday" not in " ".join(lines)
    assert lines[4].split()[1] == "none"
    assert lines[2].split()[1] == iso(F)


def test_survivors_file_is_atomic_private_and_never_follows_a_symlink(tmp_path):
    mod = _survivors()
    out = tmp_path / "survivors.txt"
    out.write_text("old contents\n")
    mod.write_private(str(out), ["outcome=off", "a row 1"])
    assert out.read_text() == "outcome=off\na row 1\n"
    assert oct(out.stat().st_mode & 0o777) == "0o600"
    assert [p.name for p in tmp_path.iterdir()] == ["survivors.txt"]       # no temp left behind
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    link = tmp_path / "link.txt"
    link.symlink_to(victim)
    mod.write_private(str(link), ["x"])
    assert victim.read_text() == "keep me" and not link.is_symlink()


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    mod = _survivors()
    out = tmp_path / "survivors.txt"
    out.write_text("complete old file\n")
    monkeypatch.setattr(mod.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError):
        mod.write_private(str(out), ["new"])
    assert out.read_text() == "complete old file\n"
    assert [p.name for p in tmp_path.iterdir()] == ["survivors.txt"]


def test_survivors_main_reads_the_real_flag_and_labels_the_outcome(db, monkeypatch, tmp_path):
    mod = _survivors()
    push(db)
    ef.clear_active()                                     # the server restarted: off
    monkeypatch.setattr(mod, "raw_events", lambda a, b: [ev(), ev(sid=MINE)])
    out = tmp_path / "s.txt"
    assert mod.main(["2026-11-23", "2026-12-07", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "outcome=off" and len(lines) == 3          # nothing was filtered
    push(db)                                              # accepted again
    mod.main(["2026-11-23", "2026-12-07", "--out", str(out)])
    lines = out.read_text().splitlines()
    assert lines[0].startswith("outcome=") and lines[0] != "outcome=off"
    assert all("row 4" not in line for line in lines)


# ── round 3, group 1: an offset-less dateTime with its own timeZone ──

TOKYO = datetime.datetime(2026, 11, 26, 0, 30, tzinfo=ZoneInfo("Asia/Tokyo"))


class _Reshaped:
    """Hands the filter the way Google can send an event: dateTime without an
    offset, plus a separate timeZone."""

    def __init__(self, service, shapes):
        self._service, self._shapes = service, shapes

    def events(self):
        inner = self._service.events()
        shapes = self._shapes

        def list_(**query):
            call = inner.list(**query)

            class Run:
                def execute(self):
                    page = call.execute()
                    items = []
                    for item in page["items"]:
                        if item["id"] in shapes:
                            item = {**item, **shapes[item["id"]]}
                        items.append(item)
                    return {**page, "items": items}
            return Run()
        return type("E", (), {"list": staticmethod(list_), "get": inner.get,
                              "patch": inner.patch, "delete": inner.delete,
                              "insert": inner.insert})()

    def __getattr__(self, name):
        return getattr(self._service, name)


@pytest.fixture
def tokyo(calendar, monkeypatch):
    fetched = ef.load_state().fetched_at
    old = iso(fetched - datetime.timedelta(days=3))
    end = TOKYO + datetime.timedelta(hours=1)
    calendar.add(fake_google.PRIMARY, "tokyo", "Kid <> Lethanial Leveille | Tokyo", TOKYO, end,
                 extendedProperties={"private": {"evalloEvent": "true", "evalloSessionId": MINE}},
                 updated=old)
    shapes = {"tokyo": {"start": {"dateTime": "2026-11-26T00:30:00", "timeZone": "Asia/Tokyo"},
                        "end": {"dateTime": "2026-11-26T01:30:00", "timeZone": "Asia/Tokyo"}}}
    inner = cal._service
    monkeypatch.setattr(cal, "_service", lambda: _Reshaped(inner(), shapes))
    # an ordinary lesson at the same real instant, to make a conflict
    calendar.add(fake_google.PRIMARY, "same", "Calc lab", TOKYO, end)
    return calendar


def test_a_timezone_event_is_placed_at_its_real_instant_everywhere(tokyo):
    day = datetime.datetime(2026, 11, 25)
    window = (day - datetime.timedelta(days=1), day + datetime.timedelta(days=3))
    blocks, _ = cal.busy_blocks(*window)
    assert any(a == TOKYO and b == TOKYO + datetime.timedelta(hours=1) for a, b in blocks)
    shown = {e["id"]: e for e in cal.events_for_app(*window)["events"]}
    assert datetime.datetime.fromisoformat(shown["tokyo"]["start"]) == TOKYO
    assert datetime.datetime.fromisoformat(shown["tokyo"]["end"]) == TOKYO + datetime.timedelta(hours=1)
    local = TOKYO.astimezone()
    now = datetime.datetime(2026, 11, 24, 9)
    listing = cal.get_upcoming_events("2026-11-25", "2026-11-28", now=now)
    assert "Tokyo" in listing and cal._spoken(local.replace(tzinfo=None)) in listing
    conflicts = cal.find_schedule_conflicts("2026-11-25", "2026-11-28", now=now)
    assert "Tokyo" in conflicts and "Calc lab" in conflicts


def test_the_event_is_kept_by_its_new_york_date_not_its_wall_clock():
    # 00:30 Tokyo on Nov 26 is Nov 25 at 10:30 AM Eastern.
    event = ev(sid=MINE)
    event["start"] = {"dateTime": "2026-11-26T00:30:00", "timeZone": "Asia/Tokyo"}
    assert ef.start_date(event) == datetime.date(2026, 11, 25)
    placed = ef.place_in_time(event)
    assert datetime.datetime.fromisoformat(placed["start"]["dateTime"]) == TOKYO
    assert event["start"]["dateTime"] == "2026-11-26T00:30:00"      # the original is untouched


def test_an_unrecognized_zone_is_unusable_and_takes_row_5_never_row_4():
    event = ev()
    event["start"] = {"dateTime": "2026-11-26T00:30:00", "timeZone": "Mars/Olympus"}
    assert ef.start_date(event) is None
    assert ef.decide(event, state(), NOW) == (False, 6)
    named = dict(event, summary="Kid <> Lethanial Leveille")
    assert ef.decide(named, state(), NOW) == (True, 5)
    assert ef.place_in_time(event)["start"] == event["start"]        # old behavior kept
    no_zone = dict(event, start={"dateTime": "2026-11-26T00:30:00"})
    assert ef.start_date(no_zone) is None


def test_an_offset_wins_over_a_conflicting_zone():
    event = ev()
    event["start"] = {"dateTime": "2026-11-26T00:30:00+09:00", "timeZone": "America/New_York"}
    assert ef.start_date(event) == datetime.date(2026, 11, 25)
    assert ef.place_in_time(event)["start"] == event["start"]


# ── round 3, group 2: the startup reset cannot fail silently ──

@pytest.fixture
def quick_clear(monkeypatch):
    monkeypatch.setattr(ef, "CLEAR_WAIT", 0.05)
    monkeypatch.setattr(ef, "CLEAR_BUSY_TIMEOUT", 0.05)


def _hold_the_writer_lock():
    import sqlite3
    holder = sqlite3.connect(database.DB_PATH, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    return holder


def test_the_reset_retries_while_another_writer_holds_the_database(db, quick_clear):
    import threading
    push(db)
    holder = _hold_the_writer_lock()
    threading.Timer(0.3, lambda: (holder.execute("COMMIT"), holder.close())).start()
    ef.clear_active()                                      # waits, retries, succeeds
    assert not ef.load_state().on


def test_a_reset_that_never_gets_the_lock_fails_startup(db, quick_clear):
    push(db)
    holder = _hold_the_writer_lock()
    try:
        with pytest.raises(RuntimeError):
            server._evallo_filter_off_until_a_push()        # the hook: the server must not start
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert ef.load_state().on                              # nothing was cleared, and it was loud


# ── round 3, group 3: voice free/busy when a blocking calendar is unreadable ──

def _failing(monkeypatch, calendar_ids):
    inner = cal._service

    class Down(_Reshaped):
        def events(self):
            real = self._service.events()

            def list_(calendarId, **query):
                if calendarId in calendar_ids:
                    return type("Boom", (), {"execute": staticmethod(
                        lambda: (_ for _ in ()).throw(RuntimeError("down")))})()
                return real.list(calendarId=calendarId, **query)
            return type("E", (), {"list": staticmethod(list_), "get": real.get,
                                  "patch": real.patch, "delete": real.delete,
                                  "insert": real.insert})()
    monkeypatch.setattr(cal, "_service", lambda: Down(inner(), {}))


def _freebusy():
    return cal.check_calendar_freebusy("2026-11-30 8am", "2026-11-30 8pm")


def test_everything_readable_keeps_the_free_list(calendar):
    result = _freebusy()
    assert result["free"] and "cannot_tell_free" not in result and "could_not_read" not in result


def test_a_failed_primary_calendar_gives_no_free_list(calendar, monkeypatch):
    from config import CALENDAR_ROLES
    assert CALENDAR_ROLES[fake_google.PRIMARY] == "blocking"
    _failing(monkeypatch, {fake_google.PRIMARY})
    result = _freebusy()
    assert "free" not in result
    assert result["could_not_read"] == ["primary.fake@example.com"]
    assert "cannot tell when he is free" in result["cannot_tell_free"]
    assert "busy" in result


def test_only_a_context_calendar_failing_keeps_the_free_list_and_names_it(calendar, monkeypatch):
    from config import CALENDAR_ROLES
    club = [cid for cid in (fake_google.ACM, fake_google.NSBE, fake_google.HOLIDAYS)
            if CALENDAR_ROLES.get(cid, "blocking") != "blocking"]
    assert club, "the fake's context calendars must be classified"
    _failing(monkeypatch, {club[0]})
    result = _freebusy()
    assert result["free"] and "cannot_tell_free" not in result
    assert result["could_not_read"]


def test_the_tool_description_tells_the_model_not_to_offer_free_times():
    from tools import registry
    text = registry.get("check_calendar_freebusy").description
    assert "never offer or guess free times" in text


# ── round 3, group 4: a filter failure must not reveal an event ──

def test_an_event_the_filter_fails_on_comes_back_anonymized(monkeypatch, capsys):
    def boom(event):
        raise RuntimeError("bad")
    monkeypatch.setattr(ef, "start_date", boom)
    event = ev(title=f"{SECRET_NAME} <> Other | Chem", **_secret_kwargs())
    shown, row = ef.view(event, state(), NOW)
    assert row == ef.ERROR_ROW and shown["summary"] == ef.ANONYMOUS_TITLE
    _clean(str(shown))
    assert "description" not in shown and "organizer" not in shown
    assert shown["attendees"] == [{"self": True, "responseStatus": "accepted"}]
    _clean(capsys.readouterr().out)


def test_a_failed_event_that_is_positively_his_keeps_its_data(monkeypatch):
    monkeypatch.setattr(ef, "start_date", lambda e: (_ for _ in ()).throw(RuntimeError("bad")))
    listed = ev(sid=MINE, title="Maribel <> Other | Chem")
    # row 2 returns before start_date is needed, so force the failure at the top
    monkeypatch.setattr(ef, "_decide", lambda *a: (_ for _ in ()).throw(RuntimeError("bad")))
    assert ef.view(listed, state(), NOW) == (listed, ef.ERROR_ROW)       # id is in the set
    named = ev(title="Kid <> Lethanial Leveille | Math")
    assert ef.view(named, state(), NOW) == (named, ef.ERROR_ROW)         # full name
    stranger = ev(title="Maribel <> Other | Chem")
    assert ef.view(stranger, state(), NOW)[0]["summary"] == ef.ANONYMOUS_TITLE


def test_view_never_raises_even_when_anonymizing_fails(monkeypatch, capsys):
    monkeypatch.setattr(ef, "_decide", lambda *a: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setattr(ef, "anonymize", lambda e: (_ for _ in ()).throw(RuntimeError("worse")))
    event = ev(title=SECRET_NAME, **_secret_kwargs())
    shown, _ = ef.view(event, state(), NOW)
    assert set(shown) <= {"id", "start", "end", "transparency"} and "id" in shown
    _clean(str(shown))
    monkeypatch.setattr(ef, "decide", lambda *a: (_ for _ in ()).throw(RuntimeError("worst")))
    shown, _ = ef.view(event, state(), NOW)
    assert set(shown) <= {"id", "start", "end", "transparency"}
    assert SECRET_NAME not in capsys.readouterr().out


def test_an_anonymous_block_copies_only_valid_evallo_properties():
    event = ev(sid=None, **_secret_kwargs())
    event["extendedProperties"]["private"] = {
        "evalloEvent": "true", "evalloSessionId": SECRET_NAME, "orgId": "Maribel Quintero org",
        "note": SECRET_NAME}
    block = ef.anonymize(event)
    private = block["extendedProperties"]["private"]
    assert set(private) == {"evalloEvent", "evalloSessionId", "orgId"}
    assert private == {"evalloEvent": "true", "evalloSessionId": "", "orgId": ""}
    _clean(str(block))
    good = ev(sid=MINE)
    assert ef.anonymize(good)["extendedProperties"]["private"]["evalloSessionId"] == MINE
