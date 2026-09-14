"""Google Calendar, as tools.

Reads cover every calendar he has selected. Writes go only to a calendar named
MILES, so anything Nova creates is easy to find, and a bad write cannot land in
a calendar someone else shares with him. Writes are also confirmed first; see
pending_action.py.

Times arrive as phrases like "monday at 3pm" and are resolved here, in code. Two
rules came from running real phrases through dateparser on Sep 13 2026:

Prefer the future. Without it "monday", said on a Sunday, resolved to the Monday
before, so "am I free monday" checked last week while "book monday" wrote to
next week.

A day with no time of day means the whole day. dateparser fills a missing time
with the current one, so "tomorrow" said at 3:52 PM meant 3:52 PM tomorrow, and a
listing starting there silently dropped the morning. Listings snap a bare day to
its start and end. A new event with no time is refused rather than invented.

Failures raise. A swallowed exception here used to turn an expired token into
"No upcoming events found", which is Nova confidently telling him his calendar
is empty. The executor turns a raise into an is_error result instead.
"""

import datetime
import os
import re

import dateparser
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

import pending_action
from tools import Permission, tool

TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "token.json")
_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_WRITE_CALENDAR = "MILES"
_MAX_EVENTS = 10

# Anything that pins a time of day. A phrase with none of these names a day.
_HAS_TIME = re.compile(
    r"\d\s*(am|pm)\b|\d:\d\d|\bnoon\b|\bmidnight\b|\bnow\b|\bhours?\b|\bminutes?\b|\dt\d",
    re.IGNORECASE)


# Words that make a new time relative to today rather than to the event being
# moved. Without one, "4pm" means 4pm on the event's own day.
_RELATIVE_TO_NOW = re.compile(r"\b(today|tonight|tomorrow|now|next|this|in \d+)\b",
                              re.IGNORECASE)


class WhenError(ValueError):
    """A time phrase that could not be used. Raised so Nova hears why."""


class EventLookupError(ValueError):
    """No single event matched. The message says what is there instead, so Nova
    can ask rather than guess."""


# ── time, resolved in code ──

def parse_when(phrase, now=None):
    """(naive local datetime, whether the phrase gave a time of day).

    Naive local because every phrase he says is local; conversion to UTC
    happens once, at the API boundary."""
    now = now or datetime.datetime.now()
    parsed = dateparser.parse(phrase, settings={"PREFER_DATES_FROM": "future",
                                                "RELATIVE_BASE": now})
    if parsed is None:
        raise WhenError(f"could not read {phrase!r} as a time; use a day and a "
                        f"clock time, like 'monday at 3pm'")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed, bool(_HAS_TIME.search(phrase))


def _start_of_day(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt):
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def resolve_window(time_min, time_max, now=None):
    """Start and end of a listing, as naive local datetimes. end may be None,
    meaning no upper bound."""
    now = now or datetime.datetime.now()
    start, start_has_time = (parse_when(time_min, now) if time_min else (now, True))
    if not start_has_time:
        start = _start_of_day(start)

    if time_max:
        end, end_has_time = parse_when(time_max, now)
        if not end_has_time:
            end = _end_of_day(end)
    elif not start_has_time:
        # "what's on tomorrow" is about tomorrow, not tomorrow onward.
        end = _end_of_day(start)
    else:
        end = None

    if end is not None and end <= start:
        raise WhenError("the end of that range is not after its start; give "
                        "both ends in full, like 'monday at 9am' and 'monday at 5pm'")
    return start, end


def _utc(dt):
    return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _spoken(dt):
    """Local and readable, so the model never converts a UTC offset in its head."""
    return dt.astimezone().strftime("%A %B %-d at %-I:%M %p")


def _clock(dt):
    """"4 PM", "4:30 PM". The minutes only when they carry information."""
    return dt.strftime("%-I:%M %p").replace(":00", "")


def _day_words(dt, now):
    """The shortest unambiguous name for a day, as a person would say it."""
    days = (dt.date() - now.date()).days
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if 1 < days < 7:
        return dt.strftime("%A")
    return dt.strftime("%A %B %-d")


def _on_day(dt, now):
    words = _day_words(dt, now)
    return words if words in ("today", "tomorrow") else f"on {words}"


def _ask(question):
    """What a proposal returns to the model.

    The question is built here, not left to the model. Asked to "read back the
    old and the new", it read back both in full, with dates, after already
    announcing the change before the call: a confirmation nobody wanted to sit
    through, recorded Sep 13 2026."""
    return (f"Nothing has changed yet. Ask Lethanial exactly this, with nothing "
            f"added before or after: {question} Then end your turn. When he "
            f"answers, call confirm_pending_action. If you proposed several changes "
            f"this turn, this question covers all of them; ask only the one from "
            f"your last result.")


def _event_start(event):
    """An aware datetime for sorting. All day events carry a bare date, which is
    local midnight; comparing raw strings misorders events whose offsets differ."""
    raw = event["start"].get("dateTime") or event["start"].get("date")
    return datetime.datetime.fromisoformat(raw).astimezone()


def _event_line(event, calendar_name=None):
    """One event as the model sees it.

    An all day event gets its day and nothing else. Labelled "all day", Nova
    read the label out: "David's birthday is all day". A birthday is not a block
    of time, it is a thing true of that day. The calendar name is only kept for
    calendars he follows, where "UF IEEE" is useful context; on his own calendar
    it was his email address, read aloud as noise."""
    title = event.get("summary", "Untitled")
    source = f" ({calendar_name})" if calendar_name else ""
    if "date" in event["start"]:
        return f"{_event_start(event).strftime('%A %B %-d')}: {title}{source}"
    return f"{_spoken(_event_start(event))}: {title}{source}"


def merge_busy(blocks):
    """Overlapping (start, end) blocks collapsed into one. With nine calendars,
    a class and a meeting in the same hour are two blocks for one busy hour."""
    merged = []
    for start, end in sorted(blocks):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def free_gaps(busy, start, end):
    """The complement of merged busy blocks inside [start, end]. Arithmetic, so
    it is done here rather than left to the model."""
    gaps, cursor = [], start
    for busy_start, busy_end in busy:
        if busy_start > cursor:
            gaps.append((cursor, min(busy_start, end)))
        cursor = max(cursor, busy_end)
    if cursor < end:
        gaps.append((cursor, end))
    return [(a, b) for a, b in gaps if b > a]


# ── Google ──

def _service():
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError("no Google token; run scripts/google_auth.py")
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, _SCOPES)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _calendars(service):
    """Every calendar on his list, paged, as {id: (name, selected, owned)}.

    owned is Google's own accessRole, not a list of names kept here. His primary
    calendar and MILES are "owner"; club and event calendars he subscribed to
    are "reader". That line is exactly the one between things he has committed
    to and things he could go to, so a new club calendar lands on the right side
    without anyone editing this file."""
    found, page = {}, None
    while True:
        result = service.calendarList().list(pageToken=page).execute()
        for item in result.get("items", []):
            selected = item.get("selected", False) or item.get("primary", False)
            owned = item.get("accessRole") == "owner"
            found[item["id"]] = (item.get("summary", "Unknown calendar"), selected, owned)
        page = result.get("nextPageToken")
        if not page:
            return found


def _selected(service):
    return {cid: name for cid, (name, selected, _) in _calendars(service).items() if selected}


def _miles_calendar_id(service):
    for cid, (name, _, _) in _calendars(service).items():
        if name == _WRITE_CALENDAR:
            return cid
    return None


def _write_calendar_id(service):
    """Created on first use by create. Edit and delete never create it: with no
    MILES calendar there is nothing of Nova's to change."""
    return (_miles_calendar_id(service)
            or service.calendars().insert(body={"summary": _WRITE_CALENDAR}).execute()["id"])


def _words(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


def _event_bounds(event):
    """(start, end, all_day) as naive local datetimes."""
    if "date" in event["start"]:
        return (datetime.datetime.fromisoformat(event["start"]["date"]),
                datetime.datetime.fromisoformat(event["end"]["date"]), True)
    local = lambda raw: datetime.datetime.fromisoformat(raw).astimezone().replace(tzinfo=None)
    return local(event["start"]["dateTime"]), local(event["end"]["dateTime"]), False


def _label(event):
    """A full description, for lookup errors the model reasons over rather than
    speaks. Spoken confirmations use the short questions instead."""
    start, end, all_day = _event_bounds(event)
    title = event.get("summary", "Untitled")
    if all_day:
        return f"'{title}' on {start.strftime('%A %B %-d')}, all day"
    label = f"'{title}' on {_spoken(start)} until {end.strftime('%-I:%M %p')}"
    if event.get("recurringEventId"):
        # Only the instance id is ever used, so only that occurrence changes.
        # Said out loud, because "delete gym monday" could mean the series.
        label += ", only that one occurrence of a repeating event"
    return label


def _find_miles_event(service, title, day, now):
    """The one MILES calendar event matching a title on a day.

    Searched by title and day rather than by event id, because only what Nova
    says reaches the conversation history. The ids in a listing are gone by the
    next turn, so a tool that took an id would be asking the model to invent
    one. A time in the day phrase narrows two events with the same title."""
    when, has_time = parse_when(day, now)
    calendar_id = _miles_calendar_id(service)
    if calendar_id is None:
        raise EventLookupError("there is no MILES calendar yet, so there is no event Nova can change")

    items = service.events().list(
        calendarId=calendar_id, timeMin=_utc(_start_of_day(when)),
        timeMax=_utc(_end_of_day(when)), singleEvents=True, orderBy="startTime",
    ).execute().get("items", [])

    wanted = set(_words(title))
    matches = [e for e in items if wanted and wanted <= set(_words(e.get("summary", "")))]
    if has_time:
        matches = [e for e in matches
                   if not _event_bounds(e)[2] and _event_bounds(e)[0].time() == when.time()]

    if len(matches) == 1:
        return calendar_id, matches[0]

    day_name = when.strftime("%A %B %-d")
    if not matches:
        there = "; ".join(_label(e) for e in items) or "nothing"
        raise EventLookupError(
            f"no event matching {title!r} on the MILES calendar on {day_name}. "
            f"That day it has: {there}. Only MILES calendar events can be changed.")
    options = "; ".join(_label(e) for e in matches)
    raise EventLookupError(
        f"{len(matches)} events match {title!r} on {day_name}: {options}. Ask him "
        f"which one, then call again with its time, like 'monday at 3pm'.")


def _resolve_new_start(phrase, old_start, now):
    """Where a moved event lands.

    Read relative to the event's own day unless the phrase names today,
    tomorrow and the like, so "4pm" stays on that day instead of meaning the
    next 4pm from now. A day with no time keeps the event's time, so "move it
    to tuesday" does not land at midnight."""
    base = now if _RELATIVE_TO_NOW.search(phrase) else _start_of_day(old_start)
    when, has_time = parse_when(phrase, base)
    if not has_time:
        when = datetime.datetime.combine(when.date(), old_start.time())
    return when


# ── tools ──

_WINDOW_HELP = (
    "Give times as a day and a clock time, like 'monday at 9am'. A bare day like "
    "'tomorrow' means that whole day."
)


@tool(
    name="get_upcoming_events",
    description=(
        "Lethanial's upcoming events from now on, soonest first, with local "
        "times already worked out. His own events come first; events from club "
        "and event calendars he follows come separately and are not "
        "commitments. Call this when he asks what is on his calendar, what he "
        "has today or this week, or when something is. " + _WINDOW_HELP
    ),
    input_schema={
        "type": "object",
        "properties": {
            "time_min": {"type": "string", "description": "Optional start. Defaults to now."},
            "time_max": {"type": "string", "description": "Optional end."},
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_upcoming_events(time_min=None, time_max=None, now=None):
    now = now or datetime.datetime.now()
    start, end = resolve_window(time_min, time_max, now)
    # Upcoming means from now. Asked about his week at 7:34 PM, Nova passed the
    # bare date for today, which resolves to midnight, and read him the 3 PM and
    # 4:30 PM sessions he had already been to as if they were still ahead.
    # Google still returns an all day event that is under way, so today's
    # birthday survives the clamp.
    start = max(start, now)
    if end is not None and end <= start:
        return "That whole range has already passed."
    service = _service()

    mine, followed, unreadable = [], [], []
    calendars = {cid: (name, owned) for cid, (name, selected, owned)
                 in _calendars(service).items() if selected}
    for cid, (name, owned) in calendars.items():
        query = {"calendarId": cid, "timeMin": _utc(start), "maxResults": _MAX_EVENTS,
                 "singleEvents": True, "orderBy": "startTime"}
        if end is not None:
            query["timeMax"] = _utc(end)
        try:
            items = service.events().list(**query).execute().get("items", [])
        except Exception:
            # One broken shared calendar should not hide the others, but it is
            # named in the answer rather than dropped.
            unreadable.append(name)
            continue
        for event in items:
            if owned:
                mine.append((_event_start(event), _event_line(event)))
            else:
                followed.append((_event_start(event), _event_line(event, name)))

    if calendars and len(unreadable) == len(calendars):
        raise RuntimeError("could not read any calendar")

    mine.sort(key=lambda pair: pair[0])
    followed.sort(key=lambda pair: pair[0])
    lines = ["His events:"]
    lines += [line for _, line in mine[:_MAX_EVENTS]] or ["Nothing on his own calendar."]
    if followed:
        # Separated rather than dropped. He keeps club calendars so there is
        # something to go to when he wants it, not as a schedule to be read.
        lines += ["", "On calendars he follows, not commitments. Mention these only "
                      "if he asks what is going on or what he could do:"]
        lines += [line for _, line in followed[:_MAX_EVENTS]]
    if unreadable:
        lines.append(f"Could not read: {', '.join(unreadable)}.")
    return "\n".join(lines)


@tool(
    name="check_calendar_freebusy",
    description=(
        "When Lethanial is busy and when he is free between two times, across "
        "every calendar he has selected, with overlapping events merged and the "
        "free gaps already computed. Call this when he asks if he is free, or "
        "before proposing a time for something new. Use the free list as given "
        "rather than working out gaps yourself. " + _WINDOW_HELP
    ),
    input_schema={
        "type": "object",
        "properties": {
            "time_min": {"type": "string", "description": "Start of the range."},
            "time_max": {"type": "string", "description": "End of the range."},
        },
        "required": ["time_min", "time_max"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def check_calendar_freebusy(time_min, time_max):
    start, end = resolve_window(time_min, time_max)
    service = _service()
    # Public holiday calendars cannot be asked for busy time (Google answers
    # notFound), and a holiday is not busy time anyway. Left in, every answer
    # carried a failure Nova would feel obliged to mention.
    calendars = {cid: name for cid, name in _selected(service).items()
                 if "#holiday@" not in cid}

    body = {"timeMin": _utc(start), "timeMax": _utc(end),
            "items": [{"id": cid} for cid in calendars]}
    answer = service.freebusy().query(body=body).execute().get("calendars", {})

    blocks, unreadable = [], []
    for cid, data in answer.items():
        if data.get("errors"):
            unreadable.append(calendars.get(cid, cid))
            continue
        blocks += [(datetime.datetime.fromisoformat(b["start"]),
                    datetime.datetime.fromisoformat(b["end"])) for b in data.get("busy", [])]

    if calendars and len(unreadable) == len(calendars):
        raise RuntimeError("could not read any calendar")

    busy = merge_busy(blocks)
    free = free_gaps(busy, start.astimezone(), end.astimezone())
    result = {
        "range": f"{_spoken(start)} to {_spoken(end)}",
        "busy": [{"from": _spoken(a), "until": _spoken(b)} for a, b in busy],
        "free": [{"from": _spoken(a), "until": _spoken(b)} for a, b in free],
    }
    if unreadable:
        result["could_not_read"] = unreadable
    return result


@tool(
    name="create_calendar_event",
    description=(
        "Propose a new event on Lethanial's MILES calendar. This does not create "
        "it. It returns one short question to ask him. Say nothing before calling "
        "this. Call this when he asks "
        "you to schedule, book, or add something. start_time needs a day and a "
        "clock time, like 'tomorrow at 2pm'. If he has already been asked about "
        "the event and agrees, call confirm_pending_action instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Title of the event."},
            "start_time": {"type": "string", "description": "A day and a clock time."},
            "duration_minutes": {"type": "integer", "description": "Length in minutes."},
        },
        "required": ["summary", "start_time", "duration_minutes"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def create_calendar_event(summary, start_time, duration_minutes, now=None):
    now = now or datetime.datetime.now()
    start, has_time = parse_when(start_time, now)
    if not has_time:
        raise WhenError(f"{start_time!r} names a day but not a time; ask him what time")
    if start < now:
        raise WhenError(f"{_spoken(start)} has already passed")
    if not isinstance(duration_minutes, int) or not 0 < duration_minutes <= 24 * 60:
        raise WhenError("duration_minutes must be between 1 and 1440")

    end = start + datetime.timedelta(minutes=duration_minutes)
    question = f"Add {summary} {_on_day(start, now)} at {_clock(start)} for {duration_minutes} minutes?"
    return _ask(pending_action.propose(question.rstrip("?"),
                                       lambda: _insert_event(summary, start, end)))


def _insert_event(summary, start, end):
    """The actual write, reached only through confirm_pending_action."""
    service = _service()
    body = {"summary": summary,
            "start": {"dateTime": start.astimezone().isoformat()},
            "end": {"dateTime": end.astimezone().isoformat()}}
    service.events().insert(calendarId=_write_calendar_id(service), body=body).execute()
    return f"Added {summary}."


_FIND_SCHEMA = {
    "title": {"type": "string",
              "description": "Words from the event's title, as he said them."},
    "day": {"type": "string",
            "description": "The day the event is on, like 'monday'. Add its time, "
                           "like 'monday at 3pm', to pick between two with the same title."},
}


@tool(
    name="delete_calendar_event",
    description=(
        "Propose deleting one event from Lethanial's MILES calendar, the one you "
        "create events on. This does not delete it. It finds the event by title "
        "and day and returns one short question to ask him. Say nothing before "
        "calling this. Call this when he asks you to delete or remove an event. "
        "Events on his other calendars cannot be changed; tell him so. If he did "
        "not say the day, find it with get_upcoming_events first. If he has "
        "already been asked about the deletion and agrees, call "
        "confirm_pending_action instead."
    ),
    input_schema={"type": "object", "properties": _FIND_SCHEMA,
                  "required": ["title", "day"]},
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def delete_calendar_event(title, day, now=None):
    now = now or datetime.datetime.now()
    calendar_id, event = _find_miles_event(_service(), title, day, now)
    start, _, all_day = _event_bounds(event)
    name = event.get("summary", "Untitled")
    if all_day:
        question = f"Delete the all day event {name} {_on_day(start, now)}{_once(event)}?"
    else:
        question = f"Delete {name} {_on_day(start, now)} at {_clock(start)}{_once(event)}?"
    return _ask(pending_action.propose(question.rstrip("?"),
                                       lambda: _delete_event(calendar_id, event["id"], name)))


def _once(event):
    """Only the instance id is ever used, so only that occurrence changes. Said
    out loud, because "delete gym monday" could be heard as the whole series."""
    return ", just that one time" if event.get("recurringEventId") else ""


@tool(
    name="update_calendar_event",
    description=(
        "Propose changing one event on Lethanial's MILES calendar: its title, its "
        "start time, its length, or any of those. This does not change it. It "
        "returns one short question to ask him. Say nothing before calling this. "
        "Call this when he asks you to "
        "move, reschedule, rename, shorten or lengthen an event. A new time alone, "
        "like '4pm', stays on the event's day; a new day alone, like 'tuesday', "
        "keeps its time. Events on his other calendars cannot be changed. If he "
        "has already been asked about the change and agrees, call "
        "confirm_pending_action instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            **_FIND_SCHEMA,
            "new_title": {"type": "string", "description": "Optional new title."},
            "new_start_time": {"type": "string",
                               "description": "Optional new start: a time, a day, or both."},
            "new_duration_minutes": {"type": "integer",
                                     "description": "Optional new length in minutes."},
        },
        "required": ["title", "day"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def update_calendar_event(title, day, new_title=None, new_start_time=None,
                          new_duration_minutes=None, now=None):
    if new_title is None and new_start_time is None and new_duration_minutes is None:
        raise WhenError("nothing to change; give a new title, start time, or duration")
    now = now or datetime.datetime.now()
    calendar_id, event = _find_miles_event(_service(), title, day, now)
    start, end, all_day = _event_bounds(event)
    if all_day and (new_start_time or new_duration_minutes is not None):
        raise WhenError("that is an all day event; only its title can be changed")

    new_start = _resolve_new_start(new_start_time, start, now) if new_start_time else start
    if new_duration_minutes is not None:
        if not isinstance(new_duration_minutes, int) or not 0 < new_duration_minutes <= 24 * 60:
            raise WhenError("new_duration_minutes must be between 1 and 1440")
        new_end = new_start + datetime.timedelta(minutes=new_duration_minutes)
    else:
        # Moving keeps the length. "Move it to 4" should not also shorten it.
        new_end = new_start + (end - start)

    if not all_day and new_start != start and new_start < now:
        raise WhenError(f"{_spoken(new_start)} has already passed")

    body = {}
    if new_title and new_title != event.get("summary"):
        body["summary"] = new_title
    if not all_day and (new_start, new_end) != (start, end):
        body["start"] = {"dateTime": new_start.astimezone().isoformat()}
        body["end"] = {"dateTime": new_end.astimezone().isoformat()}
    if not body:
        raise WhenError("that is already how the event is set")

    name = event.get("summary", "Untitled")
    clauses = []
    # Only what changes is said. "Move it to 4" does not need the date read back
    # twice, and a rename does not need the time at all.
    if new_start != start:
        if new_start.date() == start.date():
            clauses.append(f"move {name} {_on_day(start, now)} from {_clock(start)} "
                           f"to {_clock(new_start)}")
        else:
            clauses.append(f"move {name} from {_day_words(start, now)} at {_clock(start)} "
                           f"to {_day_words(new_start, now)} at {_clock(new_start)}")
    if new_end - new_start != end - start:
        minutes = int((new_end - new_start).total_seconds() // 60)
        clauses.append(f"make it {minutes} minutes long" if clauses
                       else f"make {name} {_on_day(start, now)} {minutes} minutes long")
    if "summary" in body:
        clauses.append(f"rename it to {new_title}" if clauses
                       else f"rename {name} {_on_day(start, now)} to {new_title}")

    sentence = ", and ".join(clauses) + _once(event)
    question = sentence[0].upper() + sentence[1:] + "?"
    return _ask(pending_action.propose(question.rstrip("?"),
                                       lambda: _patch_event(calendar_id, event["id"], body, new_title or name)))


def _delete_event(calendar_id, event_id, name):
    """The actual delete, reached only through confirm_pending_action."""
    _service().events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return f"Deleted {name}."


def _patch_event(calendar_id, event_id, body, name):
    """The actual edit, reached only through confirm_pending_action. patch, not
    update, so fields this tool never touches, like attendees and reminders, are
    left exactly as they were."""
    _service().events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute()
    return f"Updated {name}."


def find_overlaps(events):
    """Groups of events whose times overlap, in time order.

    events are (start, end, label). Touching is not overlapping: a lesson ending
    at 4:30 and another starting at 4:30 is back to back, which he said is fine.
    A chain where A overlaps B and B overlaps C is one group, because choosing
    what to skip among the three is one decision, not three."""
    groups, current, current_end = [], [], None
    for start, end, label in sorted(events, key=lambda e: (e[0], e[1])):
        if current and start < current_end:
            current.append((start, end, label))
            current_end = max(current_end, end)
            continue
        if len(current) > 1:
            groups.append(current)
        current, current_end = [(start, end, label)], end
    if len(current) > 1:
        groups.append(current)
    return groups


def _names(labels):
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


@tool(
    name="find_schedule_conflicts",
    description=(
        "Where Lethanial's events overlap, with each overlap's time already "
        "worked out: his own events against each other and against events on "
        "club and event calendars he follows. Covers the next seven days unless "
        "given a range. Call this when he asks what conflicts, what overlaps, or "
        "what he should skip. Then suggest what to keep: his classes and his own "
        "commitments come before events on calendars he follows, which are "
        "optional. Changes nothing. " + _WINDOW_HELP
    ),
    input_schema={
        "type": "object",
        "properties": {
            "time_min": {"type": "string", "description": "Optional start. Defaults to now."},
            "time_max": {"type": "string", "description": "Optional end. Defaults to seven days later."},
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def find_schedule_conflicts(time_min=None, time_max=None, now=None):
    now = now or datetime.datetime.now()
    start, end = resolve_window(time_min, time_max, now)
    # Same reasoning as get_upcoming_events: a conflict that is already over is
    # not something he can still decide about.
    start = max(start, now)
    if end is None:
        end = start + datetime.timedelta(days=7)
    if end <= start:
        return "That whole range has already passed."
    service = _service()

    timed, unreadable = [], []
    calendars = {cid: (name, owned) for cid, (name, selected, owned)
                 in _calendars(service).items() if selected and "#holiday@" not in cid}
    for cid, (name, owned) in calendars.items():
        try:
            items = service.events().list(
                calendarId=cid, timeMin=_utc(start), timeMax=_utc(end), maxResults=50,
                singleEvents=True, orderBy="startTime").execute().get("items", [])
        except Exception:
            unreadable.append(name)
            continue
        for event in items:
            # A birthday or a holiday is a fact about the day, not a block of
            # time, so it cannot conflict with anything.
            if "dateTime" not in event["start"]:
                continue
            whose = "his own" if owned else name
            timed.append((datetime.datetime.fromisoformat(event["start"]["dateTime"]).astimezone(),
                          datetime.datetime.fromisoformat(event["end"]["dateTime"]).astimezone(),
                          f"{event.get('summary', 'Untitled')} ({whose})"))

    if calendars and len(unreadable) == len(calendars):
        raise RuntimeError("could not read any calendar")

    lines = []
    for group in find_overlaps(timed):
        first = min(g[0] for g in group)
        last = max(g[1] for g in group)
        lines.append(f"{_spoken(first)} until {last.strftime('%-I:%M %p')}: "
                     f"{_names([g[2] for g in group])} overlap")
    if not lines:
        lines = ["No overlaps in that range."]
    if unreadable:
        lines.append(f"Could not read: {', '.join(unreadable)}.")
    return "\n".join(lines)
