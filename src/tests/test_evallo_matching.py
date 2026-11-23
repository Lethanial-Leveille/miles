"""Which Evallo event a request picks. Pure data: no browser, no network."""

from datetime import datetime, timezone

import pytest

from evallo_matching import (CalendarEvent, EVALLO_TZ, MatchError, check_status,
                             find_event, title_matches)

TUE_4PM = datetime(2026, 11, 24, 16, 0, tzinfo=EVALLO_TZ)
TUE_5PM = datetime(2026, 11, 24, 17, 0, tzinfo=EVALLO_TZ)


def ev(title, start=TUE_4PM, canceled=False):
    return CalendarEvent(title=title, start=start, canceled=canceled)


@pytest.mark.parametrize("title, name, expected", [
    ("Taylor Chen <> Lethanial Leveille", "Taylor", True),
    ("taylor chen <>  Lethanial Leveille", "TAYLOR", True),   # spacing and case
    ("Taylor <> Lethanial Leveille", "Taylor", True),               # first name only
    ("Mayan Ortiz <> Lethanial Leveille", "Taylor", False),       # longer name
    ("Taylor Chen <> Jordan Price", "Taylor", False),               # another tutor
    ("Chen Taylor <> Lethanial Leveille", "Taylor", False),         # not at the start
    ("Mary-Kate Lee <> Lethanial Leveille", "Mary-Kate", True),
    ("Mary-Kate Lee <> Lethanial Leveille", "Mary", False),
    ("Taylor Chen <> Lethanial Leveille", "", False),
])
def test_title_rule(title, name, expected):
    assert title_matches(title, name) is expected


def test_one_match_is_returned():
    want = ev("Taylor Chen <> Lethanial Leveille")
    events = [want, ev("Taylor Chen <> Lethanial Leveille", TUE_5PM),
              ev("Liam Park <> Lethanial Leveille")]
    assert find_event(events, "Taylor", TUE_4PM) is want


def test_no_match_names_what_was_nearby():
    events = [ev("Taylor Chen <> Lethanial Leveille", TUE_5PM)]
    with pytest.raises(MatchError) as err:
        find_event(events, "Taylor", TUE_4PM)
    assert "No session for Taylor" in str(err.value)
    assert "5:00 PM" in str(err.value)


def test_two_matches_refuse_to_guess():
    events = [ev("Taylor Chen <> Lethanial Leveille"), ev("Taylor Ruiz <> Lethanial Leveille")]
    with pytest.raises(MatchError, match="2 sessions match"):
        find_event(events, "Taylor", TUE_4PM)


def test_canceled_copy_is_skipped_not_counted():
    # Canceled and rebooked at the same time: Evallo shows both with one title.
    live = ev("Taylor Chen <> Lethanial Leveille")
    events = [ev("Taylor Chen <> Lethanial Leveille", canceled=True), live]
    assert find_event(events, "Taylor", TUE_4PM) is live


def test_only_a_canceled_session_is_no_match():
    events = [ev("Taylor Chen <> Lethanial Leveille", canceled=True)]
    with pytest.raises(MatchError, match="No session"):
        find_event(events, "Taylor", TUE_4PM)


def test_time_matches_to_the_minute_across_zones():
    want = ev("Taylor Chen <> Lethanial Leveille")
    # 4 PM Eastern in November is 21:00 UTC.
    assert find_event([want], "Taylor", datetime(2026, 11, 24, 21, 0, tzinfo=timezone.utc)) is want
    # A naive time is read as Eastern.
    assert find_event([want], "Taylor", datetime(2026, 11, 24, 16, 0)) is want
    with pytest.raises(MatchError):
        find_event([want], "Taylor", datetime(2026, 11, 24, 16, 30, tzinfo=EVALLO_TZ))


@pytest.mark.parametrize("given, spelled", [
    ("Completed", "Completed"), ("no show", "No Show"), ("CANCELED", "Canceled")])
def test_status_spelling(given, spelled):
    assert check_status(given) == spelled


@pytest.mark.parametrize("bad", ["Scheduled", "Cancelled", "done", ""])
def test_unknown_status_is_refused(bad):
    with pytest.raises(ValueError):
        check_status(bad)


# ── Reading cards and the students list ──────────────────────────────────

from evallo_matching import parse_card, panel_date, panel_time, student_on_list  # noqa: E402


def test_card_is_read_into_title_start_and_length():
    e = parse_card("2026-11-24", "Taylor <> Lethanial Leveille | Test Prep | 1-1\n04:15 PM - 05:45 PM",
                   index=2)
    assert e.title == "Taylor <> Lethanial Leveille | Test Prep | 1-1"
    assert e.start == datetime(2026, 11, 24, 16, 15, tzinfo=EVALLO_TZ)
    assert e.minutes == 90 and e.index == 2 and not e.canceled


def test_card_without_a_time_is_ignored():
    assert parse_card("2026-11-24", "Taylor <> Lethanial Leveille", canceled=True) is None


def test_card_then_match_end_to_end():
    cards = [("2026-11-21", "Ava <> Lethanial Leveille | Test Prep | 1-1 12:00 PM - 01:00 PM", True),
             ("2026-11-22", "Ava <> Lethanial Leveille | Test Prep | 1-1 12:00 PM - 01:00 PM", False)]
    events = [parse_card(d, t, c) for d, t, c in cards]
    with pytest.raises(MatchError, match="No session"):   # the struck through one
        find_event(events, "Ava", datetime(2026, 11, 21, 12, 0))
    assert find_event(events, "Ava", datetime(2026, 11, 22, 12, 0)) is events[1]


def test_panel_formats():
    t = datetime(2026, 11, 4, 16, 5, tzinfo=EVALLO_TZ)
    assert panel_date(t) == "4 Nov, 2026"
    assert panel_time(t) == "04:05 PM"


NAMES = ["Owen Ross", "Joaquín Diaz", "Anne-Marie Cole", "Lena Park"]


@pytest.mark.parametrize("name, found", [
    ("Lena", ["Lena Park"]),
    ("lena", ["Lena Park"]),
    ("Joaquin", ["Joaquín Diaz"]),          # speech drops the accent
    ("Anne-Marie", ["Anne-Marie Cole"]),
    ("Ow", []),                              # not a prefix match
    ("Zelda", []),                              # not mapped yet
])
def test_students_list(name, found):
    assert student_on_list(NAMES, name) == found
