"""The Evallo session id filter (docs/EVALLO_FILTER_CONTRACT.md, C1 to C7).
Every database here is a temporary one, and Google is the fake."""

import datetime
import os
import tempfile

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
    assert late.status_code == 409 and late.json()["detail"] == {"reason": "older_snapshot"}
    assert ef.load_state().ids == {NEW}
    same = put(client, agent, fetched_at=b, session_ids=[NEW])
    assert same.status_code == 200 and same.json()["fetched_at"] == b
    other = put(client, agent, fetched_at=b, session_ids=[NEW, MINE])
    assert other.status_code == 409 and other.json()["detail"] == {"reason": "conflicting_snapshot"}
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


# ── the read only survivors script ──

def test_survivors_script_lists_id_updated_and_row(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evallo_survivors", os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                                         "evallo_survivors.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    events = [{"id": "plain", "summary": "Class"}, ev(sid=MINE), ev(sid=None), ev(), ev(updated=iso(F))]
    lines = mod.survivor_lines(events, state(), NOW)
    assert [line.split()[-1] for line in lines] == ["2", "1", "3"]
    assert lines[0].startswith(MINE) and lines[1].startswith("none ")
    out = tmp_path / "survivors.txt"
    mod.write_private(str(out), lines)
    assert oct(out.stat().st_mode & 0o777) == "0o600"


def test_validation_errors_never_echo_what_was_sent(client, agent, caplog, capsys):
    import logging
    caplog.set_level(logging.DEBUG)
    secret = "SECRET-fake-token-9f8e7d"
    capsys.readouterr()
    bodies = [
        body(token=secret),
        body(session_ids=[secret]),
        body(window_start=secret),
        body(window_end=secret, fetched_at=secret),
        body(session_ids=[MINE, MINE, secret]),
    ]
    for payload in bodies:
        r = client.put("/agent/evallo/sessions", json=payload, headers=_h(agent))
        assert r.status_code == 422
        assert secret not in r.text
        assert "input" not in r.text
        assert all(set(e) == {"loc", "type"} for e in r.json()["detail"]
                   ) if isinstance(r.json()["detail"], list) else True
    raw = client.put("/agent/evallo/sessions", content=secret, headers=_h(agent))
    assert raw.status_code == 422 and secret not in raw.text
    out = capsys.readouterr()
    assert secret not in out.out and secret not in out.err
    assert secret not in caplog.text


def test_other_routes_keep_the_default_422(client, agent):
    r = client.post("/agent/jobs/report", headers=_h(agent),
                    json={"job": "lunch", "day": "2026-11-24", "ok": True})
    assert r.status_code == 422 and "input" in r.json()["detail"][0]


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


def test_startup_clear_without_a_row_or_table_does_not_raise(db, monkeypatch):
    ef.clear_active()
    monkeypatch.setattr(database, "DB_PATH", os.path.join(tempfile.mkdtemp(), "none", "x.db"))
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
