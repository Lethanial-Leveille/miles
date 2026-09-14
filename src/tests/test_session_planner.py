"""The session planner. Placement is pure; the tool is driven through a fake
calendar. Nothing here reaches Google."""

import datetime

import pytest

import calendar_tools as cal
import pending_action as pa

from test_calendar_changes import FakeService, _timed

# Sunday afternoon; the week runs Monday 14 to Sunday 20.
NOW = datetime.datetime(2026, 9, 13, 15, 52)
DAYS = [datetime.date(2026, 9, 14) + datetime.timedelta(days=i) for i in range(7)]
WEEKEND = {d for d in DAYS if d.weekday() >= 5}
END = datetime.datetime(2026, 9, 20, 23, 59)


def _session(name, count, minutes=90, earliest=(15, 0), latest=(20, 0)):
    return {"name": name, "count": count, "minutes": minutes,
            "earliest": datetime.time(*earliest), "latest_end": datetime.time(*latest)}


def _at(day, hour, minute=0):
    return datetime.datetime.combine(datetime.date(2026, 9, day), datetime.time(hour, minute))


def _plan(sessions, busy=(), days=DAYS, no_school=WEEKEND, not_before=NOW):
    return cal.plan_sessions_on(sessions, list(busy), days, no_school, not_before, END)


@pytest.fixture(autouse=True)
def fresh():
    pa._pending, pa._turn = None, 0
    cal._calendar_cache.clear()
    yield
    pa._pending, pa._turn = None, 0
    cal._calendar_cache.clear()


def test_never_before_the_students_earliest_time():
    """He said after three; she offered two."""
    placed, _ = _plan([_session("Isaiah lesson", 1, earliest=(15, 30))])
    assert placed == [("Isaiah lesson", _at(14, 15, 30), _at(14, 17, 0))]


def test_two_sessions_spread_across_the_week():
    placed, _ = _plan([_session("Andrew lesson", 2)])
    assert [p[1].date() for p in placed] == [DAYS[0], DAYS[6]]


def test_weekends_and_holidays_prefer_the_morning():
    placed, _ = _plan([_session("Andrew lesson", 1)], days=[DAYS[5]])
    assert placed[0][1] == _at(19, 9, 0)
    holiday = DAYS[2]
    placed, _ = _plan([_session("Andrew lesson", 1)], days=[holiday], no_school={holiday})
    assert placed[0][1] == _at(16, 9, 0)


def test_a_busy_morning_falls_back_to_the_afternoon():
    placed, _ = _plan([_session("Andrew lesson", 1)], busy=[(_at(19, 8), _at(19, 12))], days=[DAYS[5]])
    assert placed[0][1] == _at(19, 12, 0)


def test_nothing_lands_on_his_own_calendar():
    """She put a lesson in the class he had just described."""
    placed, _ = _plan([_session("Andrew lesson", 1)], busy=[(_at(14, 14), _at(14, 17))], days=[DAYS[0]])
    assert placed[0][1] == _at(14, 17, 0)


def test_the_same_student_is_never_twice_in_a_day():
    placed, unplaced = _plan([_session("Charlie lesson", 3)], days=[DAYS[0]])
    assert len(placed) == 1
    assert unplaced == [("Charlie lesson", 2)]


def test_different_students_can_run_back_to_back():
    placed, _ = _plan([_session("Isaiah lesson", 1, latest=(18, 0)),
                       _session("Andrew lesson", 1, latest=(18, 0))], days=[DAYS[0]])
    starts = sorted(p[1] for p in placed)
    assert starts == [_at(14, 15, 0), _at(14, 16, 30)]


def test_no_two_sessions_ever_overlap():
    """Two of her seven lessons overlapped each other."""
    placed, _ = _plan([_session("Isaiah lesson", 2), _session("Andrew lesson", 2),
                       _session("Charlie lesson", 3)])
    for i, (_, a_start, a_end) in enumerate(placed):
        for _, b_start, b_end in placed[i + 1:]:
            assert a_end <= b_start or b_end <= a_start
    assert len(placed) == 7


def test_what_does_not_fit_is_reported_not_squeezed():
    placed, unplaced = _plan([_session("Isaiah lesson", 1, earliest=(19, 0), latest=(20, 0))], days=[DAYS[0]])
    assert placed == []
    assert unplaced == [("Isaiah lesson", 1)]


def test_the_tool_plans_around_his_calendar_and_adds_at_once(monkeypatch):
    at = lambda day, h, m=0: datetime.datetime(2026, 9, day, h, m).astimezone().isoformat()
    service = FakeService(
        {
            "primary": [_timed("dsa", "DSA", at(15, 13), at(15, 17))],
            "ieee": [_timed("club", "Club meeting", at(14, 15), at(14, 20))],
            "en.usa#holiday@group.v.calendar.google.com": [],
        },
        calendars=[
            {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"},
            {"id": "ieee", "summary": "UF IEEE Calendar", "selected": True, "accessRole": "reader"},
            {"id": "en.usa#holiday@group.v.calendar.google.com", "summary": "Holidays", "selected": True, "accessRole": "reader"},
        ],
    )
    monkeypatch.setattr(cal, "_service", lambda: service)
    created = []
    monkeypatch.setattr(cal, "_insert_event", lambda name, s, e: created.append((name, s)) or f"Added {name}.")

    pa.begin_turn()
    cal.plan_sessions([{"name": "Isaiah lesson", "count": 2, "minutes": 90}],
                      time_min="monday", time_max="sunday", now=NOW)
    # Not Monday, which the club meeting covers from 3 to 8, and not Tuesday
    # before DSA ends at 5. Club events are avoided whenever there is room.
    assert pa.words_for_turn() == "Added 2 sessions: Isaiah lesson Tuesday at 5 PM and Sunday September 20 at 9 AM, all 90 minutes."
    assert len(created) == 2, "added at once, without a question"


def test_a_session_without_a_length_is_refused(monkeypatch):
    monkeypatch.setattr(cal, "_service", lambda: pytest.fail("should not reach Google"))
    with pytest.raises(cal.WhenError):
        cal.plan_sessions([{"name": "Isaiah lesson", "count": 2}], now=NOW)



def test_spreading_survives_an_unusable_last_day():
    """Planned just after midnight, the last day of the range had no usable time
    and two lessons for one student landed today and tomorrow."""
    days = DAYS + [datetime.date(2026, 9, 21)]
    not_after = datetime.datetime(2026, 9, 21, 0, 5)
    placed, _ = cal.plan_sessions_on([_session("Isaiah lesson", 2)], [], days, WEEKEND,
                                     datetime.datetime(2026, 9, 14, 0, 5), not_after)
    assert [p[1].date() for p in placed] == [DAYS[0], DAYS[6]]


def test_three_sessions_are_spread_not_bunched():
    placed, _ = _plan([_session("Charlie lesson", 3)])
    gaps = [(b[1].date() - a[1].date()).days for a, b in zip(placed, placed[1:])]
    assert min(gaps) >= 2, gaps


def test_a_spoken_block_is_respected(monkeypatch):
    """He said: don't schedule anything on Monday from 1 to 6:30, career fair."""
    service = FakeService({"primary": []}, calendars=[
        {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    monkeypatch.setattr(cal, "_insert_event", lambda *a: "Added.")
    pa.begin_turn()
    reply = cal.plan_sessions([{"name": "Andrew lesson", "count": 1, "minutes": 90}],
                              blocked=[{"from": "monday 1pm", "until": "6:30pm"}],
                              time_min="monday", time_max="monday", now=NOW)
    assert "Andrew lesson tomorrow at 6:30 PM" in reply



def test_a_block_naming_today_means_today(monkeypatch):
    """Monday, just after midnight: "monday 1pm" had resolved to the next Monday,
    and three lessons went into the career fair he had ruled out."""
    monday_early = datetime.datetime(2026, 9, 14, 0, 10)
    service = FakeService({"primary": []}, calendars=[
        {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    monkeypatch.setattr(cal, "_insert_event", lambda *a: "Added.")
    pa.begin_turn()
    reply = cal.plan_sessions([{"name": "Andrew lesson", "count": 1, "minutes": 90}],
                              blocked=[{"from": "monday 1pm", "until": "6:30pm"}],
                              time_max="today at 11pm", now=monday_early)
    assert "Andrew lesson today at 6:30 PM" in reply


def test_a_block_saying_tomorrow_still_means_tomorrow(monkeypatch):
    service = FakeService({"primary": []}, calendars=[
        {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    monkeypatch.setattr(cal, "_insert_event", lambda *a: "Added.")
    pa.begin_turn()
    reply = cal.plan_sessions([{"name": "Andrew lesson", "count": 1, "minutes": 90}],
                              blocked=[{"from": "tomorrow at 3pm", "until": "8pm"}],
                              time_min="tomorrow", time_max="tomorrow", now=NOW)
    assert "Nothing fits" in reply



# ── club events: avoided when there is room ──

def _club(day, start, end, title="Club event"):
    return (_at(day, *start), _at(day, *end), title)


def test_a_club_event_pushes_a_session_later_the_same_day():
    placed, _ = _plan([_session("Andrew lesson", 1)], days=[DAYS[0]])
    assert placed[0][1] == _at(14, 15, 0)
    placed, _ = cal.plan_sessions_on([_session("Andrew lesson", 1)], [], [DAYS[0]], set(), NOW, END,
                                     soft_busy=[_club(14, (16, 0), (17, 0))])
    assert placed[0][1] == _at(14, 17, 0)


def test_a_day_clear_of_club_events_is_preferred_over_overlapping_one():
    placed, _ = cal.plan_sessions_on([_session("Andrew lesson", 1)], [], DAYS[:2], set(), NOW, END,
                                     soft_busy=[_club(14, (15, 0), (20, 0))])
    assert placed[0][1].date() == DAYS[1]


def test_a_forced_overlap_is_placed_and_can_be_reported():
    """Better a lesson over an optional club event than no lesson at all."""
    club = [_club(14, (15, 0), (20, 0), "Solidigm Info Session")]
    placed, unplaced = cal.plan_sessions_on([_session("Andrew lesson", 1)], [], [DAYS[0]], set(), NOW, END,
                                            soft_busy=club)
    assert unplaced == [] and placed[0][1] == _at(14, 15, 0)
    assert cal._club_overlaps(placed, club) == [("Andrew lesson", _at(14, 15, 0), "Solidigm Info Session")]


def test_a_commitment_is_never_overlapped_even_when_club_events_fill_the_week():
    placed, _ = cal.plan_sessions_on([_session("Andrew lesson", 1)], [(_at(14, 15), _at(14, 18))], [DAYS[0]],
                                     set(), NOW, END, soft_busy=[_club(14, (18, 0), (20, 0))])
    assert placed[0][1] == _at(14, 18, 0)



def test_overlaps_are_one_short_note_per_session():
    """A two session plan once read out five notes before its question."""
    start = _at(16, 17, 30)
    overlaps = [("Charlie lesson", start, "TI Info Session"), ("Charlie lesson", start, "GBM #2"),
                ("Charlie lesson", start, "Study Night #1")]
    assert cal._overlap_notes(overlaps, NOW) == [
        "Charlie lesson Wednesday at 5:30 PM overlaps TI Info Session, GBM #2 and 1 more"]


def test_what_did_not_fit_is_said_with_the_plan(monkeypatch):
    """The question is spoken by code, so a note meant for after his answer was
    never heard."""
    service = FakeService({"primary": []}, calendars=[
        {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    monkeypatch.setattr(cal, "_insert_event", lambda *a: "Added.")
    pa.begin_turn()
    cal.plan_sessions([{"name": "Charlie lesson", "count": 3, "minutes": 90}],
                      time_min="monday", time_max="monday", now=NOW)
    assert pa.words_for_turn() == ("Did not fit: Charlie lesson, 2 more. "
                                   "Added 1 session: Charlie lesson tomorrow at 3 PM, all 90 minutes.")
