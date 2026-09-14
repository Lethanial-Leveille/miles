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


def test_create_stages_and_writes_nothing(monkeypatch):
    writes = []
    monkeypatch.setattr(cal, "_insert_event", lambda *a: writes.append(a) or "Created.")
    pa.begin_turn()
    reply = cal.create_calendar_event("LeetCode", "tomorrow at 10am", 90, now=NOW)
    assert reply.startswith("Nothing has changed yet")
    assert "Add LeetCode tomorrow at 10 AM for 90 minutes?" in reply
    assert writes == []
    pa.begin_turn()
    assert pa.resolve(True) == "Created."
    assert len(writes) == 1


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



def test_several_lessons_proposed_in_one_turn_are_all_created_by_one_yes(monkeypatch):
    writes = []
    monkeypatch.setattr(cal, "_insert_event", lambda summary, *a: writes.append(summary) or f"Added {summary}.")
    pa.begin_turn()
    cal.create_calendar_event("Isaiah lesson", "tuesday at 3:30pm", 90, now=NOW)
    reply = cal.create_calendar_event("Andrew lesson", "thursday at 4pm", 90, now=NOW)
    assert "Add Isaiah lesson on Tuesday at 3:30 PM for 90 minutes, and add Andrew lesson on Thursday at 4 PM for 90 minutes?" in reply
    pa.begin_turn()
    assert pa.resolve(True) == "Added Isaiah lesson. Added Andrew lesson."
    assert writes == ["Isaiah lesson", "Andrew lesson"]
