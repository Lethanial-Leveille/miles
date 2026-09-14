"""Edit and delete, against a fake Google service. Nothing here touches Google."""

import datetime

import pytest

import calendar_tools as cal
import pending_action as pa

# Sunday afternoon. The events below are on Monday the 14th.
NOW = datetime.datetime(2026, 9, 13, 15, 52)


def _at(hour, minute=0, day=14):
    """An API style dateTime for local wall clock time, so the tests hold in any
    timezone the suite runs in."""
    return datetime.datetime(2026, 9, day, hour, minute).astimezone().isoformat()


def _timed(eid, title, start, end, **extra):
    return {"id": eid, "summary": title, "start": {"dateTime": start},
            "end": {"dateTime": end}, **extra}


class _Call:
    def __init__(self, result):
        self.result = result

    def execute(self):
        return self.result


class FakeService:
    """Just enough of the Calendar client. It has no calendars() method, so any
    attempt to create a calendar fails the test loudly."""

    def __init__(self, events, calendars=None):
        self.events_by_calendar = events
        self.calendar_items = calendars if calendars is not None else [
            {"id": "miles_id", "summary": "MILES"},
            {"id": "primary", "summary": "Lethanial", "primary": True},
        ]
        self.listed, self.deleted, self.patched = [], [], []

    def calendarList(self):
        service = self

        class _List:
            def list(self, pageToken=None):
                return _Call({"items": service.calendar_items})
        return _List()

    def events(self):
        service = self

        class _Events:
            def list(self, **query):
                service.listed.append(query)
                return _Call({"items": service.events_by_calendar.get(query["calendarId"], [])})

            def delete(self, calendarId, eventId):
                service.deleted.append((calendarId, eventId))
                return _Call("")

            def patch(self, calendarId, eventId, body):
                service.patched.append((calendarId, eventId, body))
                return _Call({})
        return _Events()


@pytest.fixture(autouse=True)
def fresh_pending():
    pa._pending, pa._turn = None, 0
    yield
    pa._pending, pa._turn = None, 0


@pytest.fixture
def google(monkeypatch):
    service = FakeService({
        "miles_id": [
            _timed("e1", "LeetCode session", _at(10), _at(11, 30)),
            _timed("e2", "Gym", _at(15), _at(16)),
            _timed("e3", "Gym", _at(18), _at(19)),
        ],
        "primary": [_timed("p1", "Class", _at(9), _at(10))],
    })
    monkeypatch.setattr(cal, "_service", lambda: service)
    return service


def _confirm():
    pa.begin_turn()
    return pa.resolve(True)


# ── delete ──

def test_delete_waits_for_the_next_turn(google):
    pa.begin_turn()
    reply = cal.delete_calendar_event("leetcode", "monday", now=NOW)
    assert "Delete LeetCode session tomorrow at 10 AM?" in reply
    assert google.deleted == []
    assert _confirm().startswith("Deleted")
    assert google.deleted == [("miles_id", "e1")]


def test_only_the_miles_calendar_is_searched(google):
    pa.begin_turn()
    with pytest.raises(cal.EventLookupError):
        cal.delete_calendar_event("class", "monday", now=NOW)
    assert {q["calendarId"] for q in google.listed} == {"miles_id"}


def test_two_matches_ask_which(google):
    with pytest.raises(cal.EventLookupError, match="2 events"):
        cal.delete_calendar_event("gym", "monday", now=NOW)


def test_a_time_picks_between_two(google):
    pa.begin_turn()
    cal.delete_calendar_event("gym", "monday at 6pm", now=NOW)
    _confirm()
    assert google.deleted == [("miles_id", "e3")]


def test_no_match_says_what_is_there(google):
    with pytest.raises(cal.EventLookupError, match="LeetCode session"):
        cal.delete_calendar_event("dentist", "monday", now=NOW)


def test_no_miles_calendar_is_never_created(monkeypatch):
    service = FakeService({}, calendars=[{"id": "primary", "summary": "Lethanial", "primary": True}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    with pytest.raises(cal.EventLookupError, match="no MILES calendar"):
        cal.delete_calendar_event("gym", "monday", now=NOW)


def test_a_repeating_event_says_only_one_occurrence(monkeypatch):
    service = FakeService({"miles_id": [
        _timed("r1", "Gym", _at(15), _at(16), recurringEventId="series")]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    reply = cal.delete_calendar_event("gym", "monday", now=NOW)
    assert "Delete Gym tomorrow at 3 PM, just that one time?" in reply


# ── update ──

def test_a_bare_time_stays_on_the_events_day(google):
    pa.begin_turn()
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="4pm", now=NOW)
    assert "Move LeetCode session tomorrow from 10 AM to 4 PM?" in reply
    assert google.patched == []
    assert _confirm() == "Updated LeetCode session."
    _, eid, body = google.patched[0]
    assert eid == "e1"
    assert body["start"]["dateTime"] == _at(16)
    assert body["end"]["dateTime"] == _at(17, 30)


def test_a_bare_day_keeps_the_time(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="tuesday", now=NOW)
    assert "Move LeetCode session from tomorrow at 10 AM to Tuesday at 10 AM?" in reply


def test_tomorrow_counts_from_today_not_from_the_event(google):
    """Said on Sunday, tomorrow is Monday, even though the event is on Monday."""
    reply = cal.update_calendar_event("leetcode", "monday",
                                      new_start_time="tomorrow at 2pm", now=NOW)
    assert "Move LeetCode session tomorrow from 10 AM to 2 PM?" in reply


def test_a_new_length_moves_only_the_end(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_duration_minutes=30, now=NOW)
    assert "Make LeetCode session tomorrow 30 minutes long?" in reply
    _confirm()
    body = google.patched[0][2]
    assert body["start"]["dateTime"] == _at(10)
    assert body["end"]["dateTime"] == _at(10, 30)


def test_a_rename_sends_only_the_title(google):
    pa.begin_turn()
    reply = cal.update_calendar_event("leetcode", "monday", new_title="LeetCode grind", now=NOW)
    assert "Rename LeetCode session tomorrow to LeetCode grind?" in reply
    _confirm()
    assert google.patched[0][2] == {"summary": "LeetCode grind"}


def test_an_update_needs_a_change(google):
    with pytest.raises(cal.WhenError):
        cal.update_calendar_event("leetcode", "monday", now=NOW)


def test_moving_into_the_past_is_refused(google):
    with pytest.raises(cal.WhenError, match="already passed"):
        cal.update_calendar_event("leetcode", "monday", new_start_time="today at 8am", now=NOW)


def test_an_all_day_event_can_only_be_renamed(monkeypatch):
    service = FakeService({"miles_id": [{
        "id": "a1", "summary": "Hackathon",
        "start": {"date": "2026-09-14"}, "end": {"date": "2026-09-15"}}]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    with pytest.raises(cal.WhenError, match="all day"):
        cal.update_calendar_event("hackathon", "monday", new_start_time="4pm", now=NOW)


def test_several_changes_make_one_question(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="4pm",
                                      new_title="LeetCode grind", now=NOW)
    assert "Move LeetCode session tomorrow from 10 AM to 4 PM, and rename it to LeetCode grind?" in reply


def test_the_spoken_question_stays_short(google):
    """The regression this exists for: a read back of both versions with full
    dates, sixty words before the question mark."""
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="4pm", now=NOW)
    question = reply.split("before or after: ")[1].split("?")[0]
    assert len(question.split()) <= 12



# ── listing: now onward, his events first ──

def _listing_service():
    at = lambda h, m=0, day=13: datetime.datetime(2026, 9, day, h, m).astimezone().isoformat()
    return FakeService(
        {
            "primary": [
                {"id": "b1", "summary": "David's birthday",
                 "start": {"date": "2026-09-14"}, "end": {"date": "2026-09-15"}},
                _timed("p2", "PBP", at(12, day=14), at(13, day=14)),
            ],
            "ieee": [_timed("c1", "Renesas Tech Workshop", at(18, 30, day=14), at(20, day=14))],
        },
        calendars=[
            {"id": "primary", "summary": "leveillelethanial@gmail.com", "primary": True, "accessRole": "owner"},
            {"id": "ieee", "summary": "UF IEEE Calendar", "selected": True, "accessRole": "reader"},
        ],
    )


def test_upcoming_never_starts_before_now(monkeypatch):
    """Sep 13 2026, 7:34 PM: "today" resolved to midnight and she read him
    sessions he had already been to."""
    service = _listing_service()
    monkeypatch.setattr(cal, "_service", lambda: service)
    cal.get_upcoming_events("today", "sunday september 20", now=NOW)
    started = {datetime.datetime.fromisoformat(q["timeMin"].replace("Z", "+00:00")) for q in service.listed}
    assert started == {NOW.astimezone().astimezone(datetime.timezone.utc)}


def test_a_range_entirely_in_the_past_says_so(monkeypatch):
    monkeypatch.setattr(cal, "_service", lambda: pytest.fail("should not reach Google"))
    assert cal.get_upcoming_events("today at 8am", "today at 9am", now=NOW) == "That whole range has already passed."


def test_his_events_come_first_and_followed_calendars_are_marked(monkeypatch):
    service = _listing_service()
    monkeypatch.setattr(cal, "_service", lambda: service)
    reply = cal.get_upcoming_events(now=NOW)
    mine, followed = reply.split("not commitments")
    assert "PBP" in mine and "Renesas" not in mine
    assert "Renesas Tech Workshop (UF IEEE Calendar)" in followed


def test_an_all_day_event_is_a_day_not_a_duration(monkeypatch):
    service = _listing_service()
    monkeypatch.setattr(cal, "_service", lambda: service)
    reply = cal.get_upcoming_events(now=NOW)
    assert "Monday September 14: David's birthday" in reply
    assert "all day" not in reply
    assert "gmail.com" not in reply, "his own calendar's name is his email address"
