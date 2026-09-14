"""Calendar time handling and staging. Nothing here touches Google."""

import datetime

import pytest

import calendar_tools as cal
import pending_action as pa

# Sunday, the afternoon these bugs were found.
NOW = datetime.datetime(2026, 9, 13, 15, 52)


@pytest.fixture(autouse=True)
def fresh_pending():
    pa._pending, pa._turn = None, 0
    yield
    pa._pending, pa._turn = None, 0


def test_a_day_name_means_the_next_one_not_the_last():
    assert cal.parse_when("monday", NOW)[0].date() == datetime.date(2026, 9, 14)


def test_a_bare_day_covers_the_whole_day():
    """"tomorrow" at 3:52 PM used to start the listing at 3:52 PM tomorrow."""
    start, end = cal.resolve_window("tomorrow", None, NOW)
    assert start == datetime.datetime(2026, 9, 14, 0, 0)
    assert end == datetime.datetime(2026, 9, 14, 23, 59, 59)


def test_an_explicit_time_is_kept():
    start, end = cal.resolve_window("today at 8am", "today at 10pm", NOW)
    assert (start.hour, end.hour) == (8, 22)


def test_no_bounds_means_from_now_onward():
    assert cal.resolve_window(None, None, NOW) == (NOW, None)


def test_an_inverted_range_is_refused():
    with pytest.raises(cal.WhenError):
        cal.resolve_window("monday at 5pm", "monday at 3pm", NOW)


def test_an_unreadable_phrase_is_refused():
    with pytest.raises(cal.WhenError):
        cal.parse_when("gibberish words", NOW)


def test_merge_and_gaps():
    t = lambda h: datetime.datetime(2026, 9, 14, h, tzinfo=datetime.timezone.utc)
    busy = cal.merge_busy([(t(13), t(15)), (t(9), t(10)), (t(14), t(16))])
    assert busy == [(t(9), t(10)), (t(13), t(16))]
    assert cal.free_gaps(busy, t(8), t(18)) == [(t(8), t(9)), (t(10), t(13)), (t(16), t(18))]


def test_create_adds_at_once_and_reads_it_back(monkeypatch):
    """Sep 14 2026: he found a question before every addition too much."""
    writes = []
    monkeypatch.setattr(cal, "_insert_event", lambda *a: writes.append(a) or "Added.")
    pa.begin_turn()
    cal.create_calendar_event("LeetCode", "tomorrow at 10am", 90, now=NOW)
    assert len(writes) == 1
    assert pa.words_for_turn() == "Added LeetCode tomorrow from 10 AM to 11:30 AM."


def test_create_refuses_a_day_without_a_time():
    with pytest.raises(cal.WhenError):
        cal.create_calendar_event("gym", "tomorrow", 60, now=NOW)


def test_create_refuses_the_past():
    with pytest.raises(cal.WhenError):
        cal.create_calendar_event("gym", "today at 8am", 60, now=NOW)


@pytest.mark.parametrize("day,expected", [
    (13, "today"), (14, "tomorrow"), (16, "Wednesday"), (21, "Monday September 21"),
])
def test_days_are_named_the_way_a_person_says_them(day, expected):
    assert cal._day_words(datetime.datetime(2026, 9, day, 9), NOW) == expected


def test_clock_drops_empty_minutes():
    assert cal._clock(datetime.datetime(2026, 9, 14, 16)) == "4 PM"
    assert cal._clock(datetime.datetime(2026, 9, 14, 16, 30)) == "4:30 PM"



def test_several_additions_in_one_turn_are_all_read_back(monkeypatch):
    writes = []
    monkeypatch.setattr(cal, "_insert_event", lambda summary, *a: writes.append(summary) or "Added.")
    pa.begin_turn()
    cal.create_calendar_event("Isaiah lesson", "tuesday at 3:30pm", 90, now=NOW)
    cal.create_calendar_event("Andrew lesson", "thursday at 4pm", 90, now=NOW)
    assert writes == ["Isaiah lesson", "Andrew lesson"]
    assert pa.words_for_turn() == ("Added Isaiah lesson on Tuesday from 3:30 PM to 5 PM. "
                                   "Added Andrew lesson on Thursday from 4 PM to 5:30 PM.")


def test_todays_weekday_means_today_while_its_time_is_ahead():
    """Just after midnight on Monday Sep 14 2026, "monday at 1pm" became the 21st
    and the career fair went on the wrong week."""
    monday_early = datetime.datetime(2026, 9, 14, 0, 10)
    assert cal.parse_when("monday at 1pm", monday_early)[0] == datetime.datetime(2026, 9, 14, 13, 0)


def test_a_bare_weekday_said_that_day_still_means_next_week():
    """"Plan through sunday", said on a Sunday, is the Sunday coming."""
    sunday = datetime.datetime(2026, 9, 13, 15, 52)
    assert cal.parse_when("sunday", sunday)[0].date() == datetime.date(2026, 9, 20)


def test_todays_weekday_after_its_time_rolls_to_next_week():
    monday_afternoon = datetime.datetime(2026, 9, 14, 15, 0)
    assert cal.parse_when("monday at 1pm", monday_afternoon)[0].date() == datetime.date(2026, 9, 21)


def test_next_monday_is_refused_rather_than_guessed():
    """dateparser cannot read "next monday at 1pm" at all. A clear refusal asks
    for a date; a guess would put the event somewhere he did not say."""
    monday_early = datetime.datetime(2026, 9, 14, 0, 10)
    with pytest.raises(cal.WhenError):
        cal.parse_when("next monday at 1pm", monday_early)


def test_a_new_event_is_read_back_as_a_time_range(monkeypatch):
    """"for 300 minutes" made him do arithmetic to know when it ended."""
    monkeypatch.setattr(cal, "_insert_event", lambda *a: "Added.")
    pa.begin_turn()
    cal.create_calendar_event("career fair", "tomorrow at 1pm", 300, now=NOW)
    assert pa.words_for_turn() == "Added Career Fair tomorrow from 1 PM to 6 PM."


def test_only_an_all_lowercase_title_is_capitalized():
    assert cal._title("career fair") == "Career Fair"
    assert cal._title("Isaiah lesson") == "Isaiah lesson"
    assert cal._title("LeetCode") == "LeetCode"


def test_long_lengths_are_said_in_hours():
    assert cal._length_words(90) == "90 minutes"
    assert cal._length_words(300) == "5 hours"
    assert cal._length_words(150) == "2 and a half hours"
