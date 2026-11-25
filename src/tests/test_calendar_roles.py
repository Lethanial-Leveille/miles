"""Calendar roles, conflicts, breaks, the week view, and the Evallo guard.

The pure rules run on plain event dicts; the busy time and the write guard run
against the fake Google in tests/fake_google.py."""

import datetime
from datetime import date, datetime as dt

import pytest

import calendar_roles as roles
import calendar_tools
import fake_google
import review_plan as rp
from fake_google import ACM, HOLIDAYS, IMPORTED, MILES, NSBE, PRIMARY

NOW = dt(2026, 11, 24, 12, 0)


def ev(title, start, end, calendar=PRIMARY, event_id=None, repeating=False, series=None,
       all_day=False, **extra):
    """An event as events_for_app returns it, from naive local datetimes."""
    event_id = event_id or title.replace(" ", "_") + start.isoformat()
    if all_day:
        first, last = start.date().isoformat(), end.date().isoformat()
    else:
        first, last = start.astimezone().isoformat(), end.astimezone().isoformat()
    return {"id": event_id, "calendar": "x", "calendar_id": calendar, "title": title,
            "start": first, "end": last, "all_day": all_day, "mine": True, "editable": False,
            "repeating": repeating, "series_id": series or event_id, "transparent": False,
            "declined": False, **extra}


def d(day, hour, minute=0, month=11):
    return dt(2026, month, day, hour, minute)


# ── roles ──

def test_roles_by_calendar_id():
    assert roles.role_of(PRIMARY) == "blocking"
    assert roles.role_of(MILES, miles_id=MILES) == "blocking"
    for club in (ACM, NSBE, "fake-calendar-08@example.com", "fake-calendar-07@example.com"):
        assert roles.role_of(club) == "club"
    assert roles.role_of(HOLIDAYS) == "holiday"


def test_an_unclassified_calendar_counts_as_busy():
    # The imported "Calendar" is not listed, so it blocks until someone says otherwise.
    assert roles.role_of(IMPORTED) == "blocking"
    assert roles.role_of("fake-calendar-02@example.com") == "blocking"


def test_club_and_holiday_events_are_marked_and_not_busy():
    events = roles.annotate([
        ev("GBM", d(18, 18), d(18, 19), calendar=NSBE),
        ev("Thanksgiving Day", d(26, 0), d(27, 0), calendar=HOLIDAYS, all_day=True),
        ev("Lab", d(18, 14), d(18, 15))])
    assert [(e["title"], e["role"], e["busy"], e["mark"]) for e in events] == [
        ("GBM", "club", False, "(club)"),
        ("Thanksgiving Day", "holiday", False, "(holiday)"),
        ("Lab", "blocking", True, "")]


# ── conflicts (bug 4) ──

def test_overlap_is_flagged_with_its_overlap():
    events = roles.annotate([ev("Lab", d(18, 14), d(18, 16)), ev("Dentist", d(18, 15), d(18, 17))])
    [conflict] = roles.find_conflicts(events)
    assert (conflict["a"]["title"], conflict["b"]["title"]) == ("Lab", "Dentist")
    assert (conflict["start"], conflict["end"]) == (d(18, 15), d(18, 16))


def test_touching_is_not_a_conflict():
    events = roles.annotate([ev("Lab", d(18, 14), d(18, 16)), ev("Dentist", d(18, 16), d(18, 17))])
    assert roles.find_conflicts(events) == []


def test_all_day_events_are_ignored():
    events = roles.annotate([ev("Trip", d(18, 0), d(19, 0), all_day=True),
                             ev("Lab", d(18, 14), d(18, 16))])
    assert roles.find_conflicts(events) == []


def test_context_events_never_conflict():
    events = roles.annotate([ev("Lab", d(18, 14), d(18, 16)),
                             ev("GBM", d(18, 14), d(18, 16), calendar=NSBE),
                             ev("Seminar", d(18, 15), d(18, 17), calendar="fake-calendar-08@example.com")])
    assert roles.find_conflicts(events) == []


def test_an_attended_club_event_conflicts():
    gbm = ev("Design Team", d(18, 14), d(18, 16), calendar=NSBE, series="design_series")
    events = roles.annotate([ev("Lab", d(18, 15), d(18, 17)), gbm],
                            attending={(NSBE, "design_series")})
    [conflict] = roles.find_conflicts(events)
    assert {conflict["a"]["title"], conflict["b"]["title"]} == {"Lab", "Design Team"}
    assert [e["busy"] for e in events] == [True, True]


# ── breaks and likely cancelled ──

def test_a_class_on_a_break_day_is_likely_cancelled_and_not_busy():
    dsa = ev("DSA", d(25, 10, 40), d(25, 11, 30), repeating=True)
    other = ev("Dentist", d(25, 11), d(25, 12))
    events = roles.annotate([dsa, other])
    assert events[0]["mark"] == "(likely cancelled, Thanksgiving break)"
    assert events[0]["busy"] is False
    assert roles.find_conflicts(events) == []      # DSA and the dentist overlap, but DSA is off


def test_a_class_on_a_normal_day_is_busy():
    [dsa] = roles.annotate([ev("DSA", d(17, 13, 55), d(17, 14, 45), repeating=True)])
    assert dsa["busy"] and dsa["mark"] == ""


def test_class_titles_match_whole_titles_only():
    exam = roles.annotate([ev("DSA Final Exam", d(25, 15), d(25, 16))])[0]
    tutoring = roles.annotate([ev("Jordan <> Lethanial Leveille | Test Prep | 1-1",
                                  d(25, 12, 30), d(25, 14))])[0]
    assert exam["busy"] and tutoring["busy"]       # not classes, so a break does not cancel them


def test_a_holiday_calendar_all_day_event_makes_a_break_day():
    holiday = ev("Veterans Day", d(11, 0), d(12, 0), calendar=HOLIDAYS, all_day=True)
    dsa = ev("DSA", d(11, 13, 55), d(11, 14, 45), repeating=True)
    [_, cancelled] = roles.annotate([holiday, dsa])
    assert cancelled["mark"] == "(likely cancelled, Veterans Day)"


def test_a_recurring_club_event_on_a_break_is_a_stale_repeat():
    # Friday Fishbowl, UF ACM Events, Friday Nov 27 during Thanksgiving break.
    fishbowl = ev("Friday Fishbowl", d(27, 13), d(27, 14), calendar=ACM, repeating=True,
                  series="fishbowl_series")
    lab = ev("Lab", d(27, 13), d(27, 14))
    [marked, _] = roles.annotate([fishbowl, lab])
    assert marked["mark"] == "(likely cancelled, Thanksgiving break)"
    assert marked["busy"] is False and marked["cancelled"] is True
    # Even attended: the series is his, but the repeat on a break is not.
    [attended, _] = roles.annotate([fishbowl, lab], attending={(ACM, "fishbowl_series")})
    assert attended["busy"] is False
    assert roles.find_conflicts(roles.annotate([fishbowl, lab],
                                               attending={(ACM, "fishbowl_series")})) == []


def test_the_same_fishbowl_off_a_break_is_context_only():
    fishbowl = ev("Friday Fishbowl", d(20, 13), d(20, 14), calendar=ACM, repeating=True)
    [marked] = roles.annotate([fishbowl])
    assert marked["mark"] == "(club)" and marked["busy"] is False


def test_a_stale_repeat_is_never_suggested():
    fishbowl = roles.annotate([ev("Friday Fishbowl exam", d(27, 13), d(27, 14), calendar=ACM,
                                  repeating=True)])
    assert rp.suggest(date(2026, 11, 24), [], fishbowl) == []


# ── busy time, shared with the voice tool ──

@pytest.fixture
def google(monkeypatch):
    return fake_google.install(monkeypatch)


def _busy_hours(fake_busy):
    return [(a.astimezone().replace(tzinfo=None), b.astimezone().replace(tzinfo=None))
            for a, b in fake_busy]


def test_busy_blocks_apply_roles(google, db):
    google.add(PRIMARY, "lab", "Lab", d(18, 14), d(18, 15))
    google.add(NSBE, "talk", "Disney Info Session", d(18, 16), d(18, 17))
    busy, unreadable = calendar_tools.busy_blocks(d(18, 0), d(19, 0))
    assert _busy_hours(busy) == [(d(18, 14), d(18, 15))]
    assert unreadable == []


def test_an_attended_club_series_counts_as_busy(google, db):
    google.add(NSBE, "design1", "Design Team Meeting (Members Only)", d(30, 17), d(30, 18),
               recurringEventId="design_series")
    assert calendar_tools.busy_blocks(d(30, 0), d(30, 23, 59))[0] == []
    db.add_attending(NSBE, "design_series", "Design Team Meeting (Members Only)")
    busy, _ = calendar_tools.busy_blocks(d(30, 0), d(30, 23, 59))
    assert _busy_hours(busy) == [(d(30, 17), d(30, 18))]


def test_a_class_on_a_break_is_not_busy(google, db):
    google.add(PRIMARY, "dsa", "DSA", d(25, 10, 40), d(25, 11, 30), recurringEventId="dsa")
    assert calendar_tools.busy_blocks(d(25, 0), d(25, 23, 59))[0] == []


def test_a_transparent_event_is_not_busy(google, db):
    google.add(PRIMARY, "free", "Free time", d(18, 14), d(18, 15), transparency="transparent")
    assert calendar_tools.busy_blocks(d(18, 0), d(19, 0))[0] == []


# ── the week view ──

def test_the_week_view_marks_conflicts_and_open_days():
    events = roles.annotate([
        ev("DSA", d(25, 10, 40), d(25, 11, 30), repeating=True),
        ev("Jordan <> Lethanial Leveille | Test Prep | 1-1", d(25, 12, 30), d(25, 14)),
        ev("Test Prep | Lethanial <> Alex", d(27, 15), d(27, 16)),
        ev("Dentist", d(27, 15, 30), d(27, 16, 30)),
        ev("Friday Fishbowl", d(27, 13), d(27, 14), calendar=ACM, repeating=True),
        ev("GBM", d(26, 18), d(26, 19), calendar=NSBE),
        ev("Thanksgiving Day", d(26, 0), d(27, 0), calendar=HOLIDAYS, all_day=True)])
    text = rp.week_view(events, "2026-11-25", 4)
    assert text == "\n".join([
        "Wednesday Nov 25",
        "• 10:40 to 11:30 AM  DSA (likely cancelled, Thanksgiving break)",
        "• 12:30 to 2 PM  Test Prep with Jordan",
        "",
        "Thursday Nov 26",
        "• All day  Thanksgiving Day (holiday)",
        "• 6 to 7 PM  GBM (club)",
        "",
        "Friday Nov 27",
        "• 1 to 2 PM  Friday Fishbowl (likely cancelled, Thanksgiving break)",
        "• 3 to 4 PM  Test Prep with Alex",
        "• 3:30 to 4:30 PM  Dentist",
        "CONFLICT: Test Prep with Alex overlaps Dentist, 3:30 to 4 PM",
        "",
        "Saturday Nov 28",
        "Open"])


# ── the Evallo guard ──

EVALLO_KEYS = ("evalloEvent", "evalloSessionId", "orgId")


@pytest.fixture
def miles(google, db):
    """A MILES calendar with one plain block and one block per Evallo key."""
    google.add(MILES, "plain", "Study", d(25, 14), d(25, 15))
    for key in EVALLO_KEYS:
        google.add(MILES, f"evallo_{key}", f"Tutoring {key}", d(25, 16), d(25, 17),
                   extendedProperties={"private": {key: "123"}})
    return google


def _unchanged(fake):
    assert fake.written() == []


@pytest.mark.parametrize("key", EVALLO_KEYS)
def test_move_and_delete_refuse_an_evallo_event(miles, key):
    event_id = f"evallo_{key}"
    with pytest.raises(calendar_tools.EvalloProtectedError, match="Evallo"):
        calendar_tools.move_miles_event(event_id, d(25, 18), d(25, 19))
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools.delete_miles_event(event_id)
    _unchanged(miles)


@pytest.mark.parametrize("key", EVALLO_KEYS)
def test_the_app_routes_refuse_an_evallo_event(miles, key):
    event_id = f"evallo_{key}"
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools.change_event_for_app(event_id, title="Renamed")
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools.delete_event_for_app(event_id)
    _unchanged(miles)


@pytest.mark.parametrize("key", EVALLO_KEYS)
def test_the_voice_tools_refuse_an_evallo_event(miles, key):
    title = f"Tutoring {key}"
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools.delete_calendar_event(title, "tomorrow", now=NOW)
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools.update_calendar_event(title, "tomorrow", new_title="Renamed", now=NOW)
    _unchanged(miles)


@pytest.mark.parametrize("key", EVALLO_KEYS)
def test_undo_refuses_an_evallo_event(miles, key):
    # The event is added by Nova (so the undo knows it), then gains Evallo's
    # property in Google before the undo runs; the guard still stops the delete.
    calendar_tools._recent_changes.clear()
    calendar_tools._insert_event("Study block", d(25, 19), d(25, 20))
    [created] = [e for e in miles.events[MILES] if e.startswith("evt")]
    miles.events[MILES][created]["extendedProperties"] = {"private": {key: "1"}}
    said = calendar_tools.undo_last_change()
    assert "Evallo" in said
    assert created in miles.events[MILES]
    _unchanged(miles)


def test_a_miles_event_without_the_keys_still_works(miles):
    calendar_tools.move_miles_event("plain", d(25, 18), d(25, 19))
    calendar_tools.change_event_for_app("plain", title="Study more")
    assert calendar_tools.delete_miles_event("plain") is True
    assert [w[0] for w in miles.written()] == ["patch", "patch", "delete"]


def test_the_guard_runs_before_any_write_call(miles):
    # Every refusal above leaves written() empty; and the event was fetched.
    with pytest.raises(calendar_tools.EvalloProtectedError):
        calendar_tools.delete_miles_event("evallo_orgId")
    assert (MILES, "evallo_orgId") in miles.gets
    _unchanged(miles)


def test_inserting_stays_on_miles_only(miles):
    calendar_id, _ = calendar_tools.add_miles_event("Study", d(25, 19), d(25, 20))
    assert calendar_id == MILES
    assert [w[1] for w in miles.writes if w[0] == "insert"] == [MILES]
