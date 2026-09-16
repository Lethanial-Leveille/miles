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
        self.listed, self.deleted, self.patched, self.inserted = [], [], [], []

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

            def insert(self, calendarId, body):
                service.inserted.append((calendarId, body))
                return _Call({"id": f"new{len(service.inserted)}"})
        return _Events()


@pytest.fixture(autouse=True)
def fresh_calendar_list():
    """The calendar list is remembered for minutes, so each test's fake service
    has to start from an empty memory."""
    cal._calendar_cache.clear()
    cal._recent_changes.clear()
    yield
    cal._calendar_cache.clear()
    cal._recent_changes.clear()


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

def test_a_delete_happens_at_once_and_is_read_back(google):
    """Since Sep 16 2026. It waited for a yes on the next turn, and he found a
    question before every change too much; undo now covers it."""
    pa.begin_turn()
    reply = cal.delete_calendar_event("leetcode", "monday", now=NOW)
    assert reply.startswith("Deleted LeetCode session tomorrow at 10 AM.")
    assert google.deleted == [("miles_id", "e1")]
    assert pa.words_for_turn() == "Deleted LeetCode session tomorrow at 10 AM."


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
    assert service.deleted == [], "undo could not put it back into its series, so it still asks"


# ── update ──

def test_a_bare_time_stays_on_the_events_day(google):
    pa.begin_turn()
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="4pm", now=NOW)
    assert reply.startswith("Moved LeetCode session tomorrow from 10 AM to 4 PM.")
    _, eid, body = google.patched[0]
    assert eid == "e1"
    assert body["start"]["dateTime"] == _at(16)
    assert body["end"]["dateTime"] == _at(17, 30)


def test_a_bare_day_keeps_the_time(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="tuesday", now=NOW)
    assert reply.startswith("Moved LeetCode session from tomorrow at 10 AM to Tuesday at 10 AM.")


def test_tomorrow_counts_from_today_not_from_the_event(google):
    """Said on Sunday, tomorrow is Monday, even though the event is on Monday."""
    reply = cal.update_calendar_event("leetcode", "monday",
                                      new_start_time="tomorrow at 2pm", now=NOW)
    assert reply.startswith("Moved LeetCode session tomorrow from 10 AM to 2 PM.")


def test_a_new_length_moves_only_the_end(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_duration_minutes=30, now=NOW)
    assert reply.startswith("Made LeetCode session tomorrow 30 minutes long.")
    body = google.patched[0][2]
    assert body["start"]["dateTime"] == _at(10)
    assert body["end"]["dateTime"] == _at(10, 30)


def test_a_rename_sends_only_the_title(google):
    pa.begin_turn()
    reply = cal.update_calendar_event("leetcode", "monday", new_title="LeetCode grind", now=NOW)
    assert reply.startswith("Renamed LeetCode session tomorrow to LeetCode grind.")
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


def test_several_changes_make_one_sentence(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="4pm",
                                      new_title="LeetCode grind", now=NOW)
    assert reply.startswith(
        "Moved LeetCode session tomorrow from 10 AM to 4 PM, and renamed it to LeetCode grind.")


def test_the_spoken_read_back_stays_short(google):
    """The regression this exists for: a read back of both versions with full
    dates, sixty words long."""
    pa.begin_turn()
    cal.update_calendar_event("leetcode", "monday", new_start_time="4pm", now=NOW)
    assert len(pa.words_for_turn().split()) <= 12



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
    now_utc = NOW.astimezone().astimezone(datetime.timezone.utc)
    instant = lambda raw: datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    upcoming = [q for q in service.listed if instant(q.get("timeMax", "9999-01-01T00:00:00Z")) != now_utc]
    assert {instant(q["timeMin"]) for q in upcoming} == {now_utc}
    # What is already over is looked up separately, and only up to now.
    assert all(instant(q["timeMax"]) == now_utc
               for q in service.listed if q not in upcoming)


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



# ── conflicts ──

def _t(hour, minute=0, day=17):
    return datetime.datetime(2026, 9, day, hour, minute)


def test_overlapping_events_are_grouped():
    groups = cal.find_overlaps([(_t(16), _t(18), "DSA"), (_t(17), _t(18, 30), "Workshop")])
    assert [[g[2] for g in group] for group in groups] == [["DSA", "Workshop"]]


def test_back_to_back_is_not_a_conflict():
    """He said lessons can run back to back online."""
    assert cal.find_overlaps([(_t(15), _t(16, 30), "Isaiah"), (_t(16, 30), _t(18), "Andrew")]) == []


def test_a_chain_of_overlaps_is_one_decision():
    groups = cal.find_overlaps([(_t(15), _t(17), "A"), (_t(16), _t(18), "B"), (_t(17, 30), _t(19), "C")])
    assert len(groups) == 1 and [g[2] for g in groups[0]] == ["A", "B", "C"]


def test_conflicts_name_whose_event_it_is(monkeypatch):
    at = lambda h, m=0: datetime.datetime(2026, 9, 17, h, m).astimezone().isoformat()
    service = FakeService(
        {
            "primary": [
                _timed("dsa", "DSA", at(16), at(18)),
                {"id": "bd", "summary": "David's birthday",
                 "start": {"date": "2026-09-17"}, "end": {"date": "2026-09-18"}},
            ],
            "ieee": [_timed("ws", "Renesas Tech Workshop", at(17), at(18, 30))],
            "en.usa#holiday@group.v.calendar.google.com": [_timed("h", "Should never be read", at(16), at(20))],
        },
        calendars=[
            {"id": "primary", "summary": "leveillelethanial@gmail.com", "primary": True, "accessRole": "owner"},
            {"id": "ieee", "summary": "UF IEEE Calendar", "selected": True, "accessRole": "reader"},
            {"id": "en.usa#holiday@group.v.calendar.google.com", "summary": "Holidays", "selected": True, "accessRole": "reader"},
        ],
    )
    monkeypatch.setattr(cal, "_service", lambda: service)
    reply = cal.find_schedule_conflicts(now=NOW)
    assert reply == "Thursday September 17 at 4:00 PM until 6:30 PM: DSA (his own) and Renesas Tech Workshop (UF IEEE Calendar) overlap"
    assert "#holiday@" not in {q["calendarId"] for q in service.listed}


def test_no_conflicts_says_so(monkeypatch):
    service = FakeService({"primary": []}, calendars=[
        {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    assert cal.find_schedule_conflicts(now=NOW) == "No overlaps in that range."



# ── speed ──

def test_calendars_are_fetched_at_the_same_time(monkeypatch):
    """Nine calendars one after another took 1621ms. Every calendar's request
    has to be in flight before any of them returns."""
    import threading
    calendars = [{"id": f"c{i}", "summary": f"Calendar {i}", "selected": True, "accessRole": "reader"}
                 for i in range(4)]
    service = FakeService({}, calendars=calendars)
    all_started = threading.Barrier(4, timeout=2)
    original_events = service.events

    def events():
        real = original_events()

        class Waiting:
            def list(self, **query):
                all_started.wait()   # raises if the requests ran one at a time
                return real.list(**query)
        return Waiting()

    monkeypatch.setattr(service, "events", events)
    monkeypatch.setattr(cal, "_service", lambda: service)
    reply = cal.get_upcoming_events(now=NOW)
    assert "Could not read" not in reply
    assert {q["calendarId"] for q in service.listed} == {"c0", "c1", "c2", "c3"}


def test_the_calendar_list_is_remembered_between_questions(monkeypatch):
    service = FakeService({"primary": []}, calendars=[
        {"id": "primary", "summary": "me", "primary": True, "accessRole": "owner"}])
    lookups = []
    original = service.calendarList
    monkeypatch.setattr(service, "calendarList", lambda: (lookups.append(1), original())[1])
    monkeypatch.setattr(cal, "_service", lambda: service)
    cal.get_upcoming_events(now=NOW)
    cal.find_schedule_conflicts(now=NOW)
    assert len(lookups) == 1



# ── renaming, and spelling what cannot be heard ──

def test_similar_sounding_names_share_a_sound_code():
    assert cal._soundex("Charlie") == cal._soundex("Charley") == "C640"
    assert cal._soundex("session") != cal._soundex("grind")


def test_a_rename_that_sounds_the_same_says_only_what_changed():
    """"Rename Charlie to Charley" sounded like nothing changed, and spelling the
    whole name out was more than he wanted to hear."""
    assert cal._spelling_note("Charlie lesson", "Charley lesson") == ", E Y instead of I E"
    assert cal._spelling_note("LeetCode session", "LeetCode grind") == ""


@pytest.mark.parametrize("old,new,said", [
    ("Charlie", "Charley", "E Y instead of I E"),
    ("Jon", "John", "with an added H"),
    ("Isaac", "Isac", "without the A"),
    ("Kristin", "Christine", "spelled C H R I S T I N E"),
])
def test_the_difference_is_said_as_briefly_as_possible(old, new, said):
    assert cal._spelled_difference(old, new) == said


def test_renaming_one_event_spells_a_homophone(google):
    google.events_by_calendar["miles_id"].append(_timed("c1", "Charlie lesson", _at(12), _at(13)))
    reply = cal.update_calendar_event("charlie", "monday", new_title="Charley lesson", now=NOW)
    assert reply.startswith("Renamed Charlie lesson tomorrow to Charley lesson, E Y instead of I E.")


def test_every_event_with_the_name_is_renamed_in_one_question(monkeypatch):
    """Sep 14 2026: fixing Charlie to Charley on three lessons took five read
    backs and three yeses, one event at a time."""
    at = lambda day, h: datetime.datetime(2026, 9, day, h).astimezone().isoformat()
    service = FakeService({"miles_id": [
        _timed("m", "Charlie lesson", at(14, 18), at(14, 19)),
        _timed("w", "Charlie lesson", at(16, 16), at(16, 17)),
        _timed("f", "Charlie lesson", at(18, 16), at(18, 17)),
        _timed("i", "Isaiah lesson", at(18, 17), at(18, 18)),
    ]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    pa.begin_turn()
    reply = cal.rename_calendar_events("Charlie", "Charley", now=NOW)
    assert "Rename Charlie to Charley, E Y instead of I E, on 3 events: tomorrow, Wednesday and Friday?" in reply
    assert service.patched == []
    pa.begin_turn()
    assert pa.resolve(True) == "Renamed 3 events."
    assert {(eid, body["summary"]) for _, eid, body in service.patched} == {
        ("m", "Charley lesson"), ("w", "Charley lesson"), ("f", "Charley lesson")}


def test_a_name_nobody_has_is_refused(google):
    with pytest.raises(cal.EventLookupError):
        cal.rename_calendar_events("Zelda", "Zelena", now=NOW)



# ── undo ──

def test_undo_removes_what_was_just_added(google):
    pa.begin_turn()
    cal.create_calendar_event("career fair", "tomorrow at 1pm", 300, now=NOW)
    assert len(google.inserted) == 1
    pa.begin_turn()
    cal.undo_last_change()
    assert google.deleted == [("miles_id", "new1")]
    assert pa.words_for_turn() == "Removed Career Fair."


def test_undo_takes_back_a_whole_plan_at_once(google):
    pa.begin_turn()
    cal.plan_sessions([{"name": "Charlie lesson", "count": 3, "minutes": 60}],
                      time_min="tuesday", time_max="friday", now=NOW)
    assert len(google.inserted) == 3
    pa.begin_turn()
    cal.undo_last_change()
    assert len(google.deleted) == 3
    assert pa.words_for_turn() == "Removed 3 events."


def test_only_the_latest_addition_is_undone(google):
    pa.begin_turn()
    cal.create_calendar_event("Gym", "tomorrow at 7am", 60, now=NOW)
    pa.begin_turn()
    cal.create_calendar_event("Study", "tomorrow at 8pm", 60, now=NOW)
    pa.begin_turn()
    cal.undo_last_change()
    assert google.deleted == [("miles_id", "new2")]


def test_nothing_to_undo_is_a_failure_she_explains(google):
    with pytest.raises(LookupError):
        cal.undo_last_change()


def test_undo_puts_a_moved_event_back(google):
    pa.begin_turn()
    cal.update_calendar_event("leetcode", "monday", new_start_time="4pm",
                              new_title="LeetCode grind", now=NOW)
    pa.begin_turn()
    cal.undo_last_change()
    assert google.patched[-1] == ("miles_id", "e1", {
        "summary": "LeetCode session",
        "start": {"dateTime": _at(10)}, "end": {"dateTime": _at(11, 30)}})
    assert pa.words_for_turn() == "Put back LeetCode session."


def test_undo_restores_a_deleted_event_as_it_was(monkeypatch):
    service = FakeService({"miles_id": [_timed(
        "e1", "Dentist", _at(10), _at(11), location="Archer Rd",
        description="bring insurance card", htmlLink="https://calendar/x")]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    pa.begin_turn()
    cal.delete_calendar_event("dentist", "monday", now=NOW)
    pa.begin_turn()
    cal.undo_last_change()
    assert service.inserted == [("miles_id", {
        "summary": "Dentist", "description": "bring insurance card",
        "location": "Archer Rd", "start": {"dateTime": _at(10)},
        "end": {"dateTime": _at(11)}})]
    assert pa.words_for_turn() == "Restored Dentist."


def test_several_moves_on_one_turn_undo_together(google):
    pa.begin_turn()
    cal.update_calendar_event("gym", "monday at 3pm", new_start_time="2pm", now=NOW)
    cal.update_calendar_event("gym", "monday at 6pm", new_start_time="7pm", now=NOW)
    assert pa.words_for_turn() == (
        "Moved Gym tomorrow from 3 PM to 2 PM. Moved Gym tomorrow from 6 PM to 7 PM.")
    pa.begin_turn()
    cal.undo_last_change()
    assert [eid for _, eid, _ in google.patched[2:]] == ["e3", "e2"], "newest first"
    assert pa.words_for_turn() == "Put back Gym and Gym."


def test_undo_says_each_kind_of_change(google):
    pa.begin_turn()
    cal.create_calendar_event("Study", "tomorrow at 8pm", 60, now=NOW)
    cal.delete_calendar_event("leetcode", "monday", now=NOW)
    pa.begin_turn()
    cal.undo_last_change()
    assert pa.words_for_turn() == "Restored LeetCode session. Removed Study."


# ── bare clock times and the wrong day (Sep 15 2026) ──
# One rescheduling conversation took eight turns and four failed calls. "3:30"
# became 3:30 AM, "wednesday at 4" became April 15 2027 because dateparser reads
# a bare number as a month, and a lesson on Thursday said to be on Wednesday was
# reported as not on the calendar at all.

def _instant(raw):
    return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))


class RangedService(FakeService):
    """Honors timeMin and timeMax. The week wide search depends on the range, and
    the plain fake returns every event for any range, which would let that
    search pass without doing anything."""

    def events(self):
        service, inner = self, super().events()

        class _Ranged:
            def list(self, **query):
                service.listed.append(query)
                low, high = _instant(query["timeMin"]), _instant(query["timeMax"])
                items = [e for e in service.events_by_calendar.get(query["calendarId"], [])
                         if low <= _instant(e["start"]["dateTime"]) <= high]
                return _Call({"items": items})

            def __getattr__(self, name):
                return getattr(inner, name)
        return _Ranged()


@pytest.fixture
def lessons(monkeypatch):
    """Monday has a gym session; the Andrew lesson is on Tuesday."""
    service = RangedService({"miles_id": [
        _timed("g1", "Gym", _at(15), _at(16)),
        _timed("a1", "Andrew lesson", _at(16, day=15), _at(17, day=15)),
    ]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    return service


def test_a_bare_hour_is_a_clock_time_not_a_month():
    when, has_time = cal.parse_when("wednesday at 4", NOW)
    assert (when.date(), has_time) == (datetime.date(2026, 9, 16), True)
    when, _ = cal.parse_when("5", NOW)
    assert (when.year, when.month) == (2026, 9)


def test_saying_am_or_pm_is_never_second_guessed():
    for phrase in ("4pm", "at 9am", "9 AM", "tomorrow morning at 8", "noon"):
        assert not cal._bare_clock(phrase), phrase
    assert not cal._bare_clock("in 30 minutes"), "a relative time has no half of the day"
    assert cal._bare_clock("wednesday at 4") and cal._bare_clock("3:30")


def test_a_bare_time_stays_in_the_same_half_of_the_day(google):
    reply = cal.update_calendar_event("gym", "monday at 3pm", new_start_time="2:30", now=NOW)
    assert reply.startswith("Moved Gym tomorrow from 3 PM to 2:30 PM.")


def test_a_morning_event_moved_to_a_bare_time_stays_in_the_morning(google):
    reply = cal.update_calendar_event("leetcode", "monday", new_start_time="9", now=NOW)
    assert reply.startswith("Moved LeetCode session tomorrow from 10 AM to 9 AM.")


def test_an_explicit_am_on_an_afternoon_event_is_kept(google):
    reply = cal.update_calendar_event("gym", "monday at 3pm", new_start_time="11am", now=NOW)
    assert "from 3 PM to 11 AM." in reply


def test_a_bare_time_in_the_day_finds_the_afternoon_event(google):
    pa.begin_turn()
    cal.delete_calendar_event("gym", "monday at 6", now=NOW)
    assert google.deleted == [("miles_id", "e3")]


def test_a_change_finds_the_event_on_the_day_it_is_really_on(lessons):
    pa.begin_turn()
    reply = cal.update_calendar_event("andrew", "monday", new_start_time="4:30", now=NOW)
    assert "Andrew lesson" in reply and "Tuesday" in reply and "to 4:30 PM" in reply
    _, eid, body = lessons.patched[0]
    assert eid == "a1"
    assert body["start"]["dateTime"] == _at(16, 30, day=15)


def test_several_matches_that_week_ask_which(monkeypatch):
    service = RangedService({"miles_id": [
        _timed("a1", "Andrew lesson", _at(16, day=15), _at(17, day=15)),
        _timed("a2", "Andrew lesson", _at(16, day=17), _at(17, day=17)),
    ]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    with pytest.raises(cal.EventLookupError, match="2 that week"):
        cal.update_calendar_event("andrew", "monday", new_start_time="5", now=NOW)


def test_a_delete_never_reaches_another_day(lessons):
    with pytest.raises(cal.EventLookupError, match="no event matching"):
        cal.delete_calendar_event("andrew", "monday", now=NOW)
    assert lessons.deleted == []


def test_a_new_event_at_a_bare_hour_is_in_the_afternoon(google):
    pa.begin_turn()
    reply = cal.create_calendar_event("Lesson", "wednesday at 4", 60, now=NOW)
    assert "from 4 PM to 5 PM" in reply
    start = _instant(google.inserted[0][1]["start"]["dateTime"]).astimezone()
    assert (start.day, start.hour) == (16, 16)


def test_a_new_event_at_a_bare_morning_hour_stays_in_the_morning(google):
    pa.begin_turn()
    reply = cal.create_calendar_event("Study", "wednesday at 9", 60, now=NOW)
    assert "from 9 AM to 10 AM" in reply


# ── a named weekday is the one in the event's own week (Sep 16 2026) ──

def _wednesday_lesson(monkeypatch):
    service = FakeService({"miles_id": [
        _timed("w1", "Andrew lesson", _at(16, day=16), _at(17, day=16))]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    return service


def test_a_weekday_before_the_event_stays_in_its_week(monkeypatch):
    """A Thursday lesson moved to "wednesday" went to the Wednesday after."""
    service = _wednesday_lesson(monkeypatch)
    cal.update_calendar_event("andrew", "wednesday", new_start_time="tuesday at 4:30pm", now=NOW)
    assert service.patched[0][2]["start"]["dateTime"] == _at(16, 30, day=15)


def test_a_weekday_after_the_event_stays_in_its_week(monkeypatch):
    service = _wednesday_lesson(monkeypatch)
    cal.update_calendar_event("andrew", "wednesday", new_start_time="friday at 5", now=NOW)
    assert service.patched[0][2]["start"]["dateTime"] == _at(17, day=18)


def test_a_nearest_day_already_gone_means_the_next_one(monkeypatch):
    """Said on Sunday the 13th, "saturday" for a Wednesday lesson is not the
    12th, which has passed."""
    service = _wednesday_lesson(monkeypatch)
    cal.update_calendar_event("andrew", "wednesday", new_start_time="saturday", now=NOW)
    assert service.patched[0][2]["start"]["dateTime"] == _at(16, day=19)


def test_an_explicit_date_beats_the_weekday(monkeypatch):
    service = _wednesday_lesson(monkeypatch)
    cal.update_calendar_event("andrew", "wednesday",
                              new_start_time="tuesday september 22 at 4pm", now=NOW)
    assert service.patched[0][2]["start"]["dateTime"] == _at(16, day=22)


def test_next_weekday_is_refused_rather_than_guessed(monkeypatch):
    """dateparser cannot read "next tuesday" in any form, so Nova is told to
    give a date. A guess here would be the wrong week half the time."""
    service = _wednesday_lesson(monkeypatch)
    with pytest.raises(cal.WhenError):
        cal.update_calendar_event("andrew", "wednesday", new_start_time="next tuesday", now=NOW)
    assert service.patched == []


# ── a listing names what is already over, and says when it stopped (Sep 15 2026) ──

def test_a_lesson_already_missed_today_is_listed_as_over(monkeypatch):
    """At 11:33 PM he asked to move the lesson he had missed that evening, and
    the listing, starting at now, told Nova there was only one that week."""
    service = RangedService({"miles_id": [
        _timed("a1", "Andrew lesson", _at(17, 30, day=15), _at(19, day=15)),
        _timed("a2", "Andrew lesson", _at(16, day=17), _at(17, 30, day=17)),
    ]}, calendars=[{"id": "miles_id", "summary": "MILES", "selected": True,
                    "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    late = datetime.datetime(2026, 9, 15, 23, 33)
    out = cal.get_upcoming_events("today", "sunday", now=late)
    over, upcoming = out.split("His events:")
    assert "Already over" in over and "Andrew lesson" in over and "5:30 PM" in over
    assert "Thursday" in upcoming and "5:30 PM" not in upcoming


def test_nothing_over_adds_nothing(monkeypatch):
    service = RangedService({"miles_id": [
        _timed("a2", "Andrew lesson", _at(16, day=17), _at(17, 30, day=17))]},
        calendars=[{"id": "miles_id", "summary": "MILES", "selected": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    out = cal.get_upcoming_events("today", "sunday", now=datetime.datetime(2026, 9, 15, 23, 33))
    assert out.startswith("His events:")


def test_a_long_listing_says_it_stopped_short(monkeypatch):
    many = [_timed(f"e{i}", f"Thing {i}", _at(8 + i % 12, day=14 + i // 12), _at(8 + i % 12, 30, day=14 + i // 12))
            for i in range(cal._MAX_EVENTS + 3)]
    service = FakeService({"miles_id": many}, calendars=[
        {"id": "miles_id", "summary": "MILES", "selected": True, "accessRole": "owner"}])
    monkeypatch.setattr(cal, "_service", lambda: service)
    out = cal.get_upcoming_events(now=NOW)
    assert "There are more of his events in this range than listed" in out
    assert out.count("Thing ") == cal._MAX_EVENTS


def test_several_matches_let_nova_change_each_without_asking(google, monkeypatch):
    """He had said which lesson went where; the result made her ask anyway."""
    with pytest.raises(cal.EventLookupError, match="If he asked for each of them to change"):
        cal.update_calendar_event("gym", "monday", new_start_time="5pm", now=NOW)
    service = RangedService({"miles_id": [
        _timed("a1", "Andrew lesson", _at(16, day=15), _at(17, day=15)),
        _timed("a2", "Andrew lesson", _at(16, day=17), _at(17, day=17))]})
    monkeypatch.setattr(cal, "_service", lambda: service)
    with pytest.raises(cal.EventLookupError, match="If he asked for each of them to change"):
        cal.update_calendar_event("andrew", "monday", new_start_time="5pm", now=NOW)
