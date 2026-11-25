"""A fake Google Calendar for the nightly review's tests: calendars, events
with the fields the review reads, and a record of every write and every read
of one event. Nothing here touches Google."""

import datetime
from types import SimpleNamespace

import httplib2
from googleapiclient.errors import HttpError

PRIMARY = "primary.fake@example.com"
MILES = "fake-calendar-06@example.com"
ACM = "fake-calendar-05@example.com"
NSBE = "fake-calendar-04@example.com"
HOLIDAYS = "en.usa#holiday@group.v.calendar.google.com"
IMPORTED = "fake-calendar-09@example.com"


def http_error(status):
    return HttpError(httplib2.Response({"status": status}), b"")


class _Exec:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def execute(self):
        if self.error:
            raise self.error
        return self.value


def _aware(value):
    return value.astimezone() if value.tzinfo is None else value


def _bounds(event):
    if "date" in event["start"]:
        return (_aware(datetime.datetime.fromisoformat(event["start"]["date"])),
                _aware(datetime.datetime.fromisoformat(event["end"]["date"])))
    return (datetime.datetime.fromisoformat(event["start"]["dateTime"]),
            datetime.datetime.fromisoformat(event["end"]["dateTime"]))


class FakeCalendar:
    CALENDARS = [
        (PRIMARY, "primary.fake@example.com", "owner", True),
        (MILES, "MILES", "owner", False),
        (ACM, "UF ACM Events", "reader", False),
        (NSBE, "General NSBE Member Events", "reader", False),
        (HOLIDAYS, "Holidays in United States", "reader", False),
        (IMPORTED, "Calendar", "reader", False),
    ]

    def __init__(self):
        self.events = {cid: {} for cid, *_ in self.CALENDARS}
        self.writes = []          # (verb, calendar id, event id) of every write
        self.inserts = []         # (calendar id, body) of every insert
        self.gets = []
        self._n = 0

    # building the calendar
    def add(self, calendar_id, event_id, title, start, end, **extra):
        """A timed event, from naive local datetimes."""
        self.events[calendar_id][event_id] = {
            "id": event_id, "summary": title,
            "start": {"dateTime": _aware(start).isoformat()},
            "end": {"dateTime": _aware(end).isoformat()}, **extra}
        return event_id

    def add_all_day(self, calendar_id, event_id, title, first, last_exclusive, **extra):
        self.events[calendar_id][event_id] = {
            "id": event_id, "summary": title,
            "start": {"date": first.isoformat()}, "end": {"date": last_exclusive.isoformat()},
            **extra}
        return event_id

    def remove(self, calendar_id, event_id):
        """He deletes it in Google, outside this code."""
        del self.events[calendar_id][event_id]

    def written(self, verbs=("patch", "delete")):
        return [w for w in self.writes if w[0] in verbs]

    # the client
    def calendarList(self):
        items = [{"id": cid, "summary": name, "selected": True, "accessRole": role,
                  **({"primary": True} if primary else {})}
                 for cid, name, role, primary in self.CALENDARS]
        return SimpleNamespace(list=lambda pageToken=None: _Exec({"items": items}))

    def calendars(self):
        raise AssertionError("MILES exists; nothing should create a calendar")

    def events_(self):
        fake = self

        def list_(calendarId, timeMin, timeMax, **_):
            lo = datetime.datetime.fromisoformat(timeMin)
            hi = datetime.datetime.fromisoformat(timeMax)
            found = [e for e in fake.events[calendarId].values()
                     if _bounds(e)[0] < hi and lo < _bounds(e)[1]]
            return _Exec({"items": found})

        def get(calendarId, eventId):
            fake.gets.append((calendarId, eventId))
            event = fake.events.get(calendarId, {}).get(eventId)
            return _Exec(event, None if event else http_error(404))

        def insert(calendarId, body):
            fake._n += 1
            event_id = f"evt{fake._n}"
            fake.writes.append(("insert", calendarId, event_id))
            fake.inserts.append((calendarId, body))
            fake.events[calendarId][event_id] = {"id": event_id, **body}
            return _Exec({"id": event_id})

        def patch(calendarId, eventId, body):
            fake.writes.append(("patch", calendarId, eventId))
            fake.events[calendarId][eventId].update(body)
            return _Exec({})

        def delete(calendarId, eventId):
            fake.writes.append(("delete", calendarId, eventId))
            fake.events[calendarId].pop(eventId, None)
            return _Exec("")

        return SimpleNamespace(list=list_, get=get, insert=insert, patch=patch, delete=delete)


class _Service:
    """What calendar_tools._service() returns: the fake's client."""

    def __init__(self, fake):
        self._fake = fake

    def calendarList(self):
        return self._fake.calendarList()

    def calendars(self):
        return self._fake.calendars()

    def events(self):
        return self._fake.events_()


def install(monkeypatch, fake=None):
    """Make calendar_tools talk to a FakeCalendar. Returns it."""
    import calendar_tools
    fake = fake or FakeCalendar()
    calendar_tools._calendar_cache.clear()
    monkeypatch.setattr(calendar_tools, "_service", lambda: _Service(fake))
    return fake
