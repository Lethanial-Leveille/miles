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
import string
import time
from concurrent.futures import ThreadPoolExecutor

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
    has_time = bool(_HAS_TIME.search(phrase))

    # Today's weekday with a time still ahead means today. Just after midnight on
    # Monday Sep 14 2026, "monday at 1pm" preferred the future and became the
    # 21st, and a career fair went on the wrong week. Only with a time: a bare
    # "sunday" said on a Sunday still means next week, which is what "plan
    # through sunday" needs. A time already gone today still rolls forward.
    today_name = now.strftime("%A").lower()
    if (has_time and (parsed.date() - now.date()).days == 7
            and not re.search(r"\bnext\b", phrase, re.I)
            and re.search(rf"\b{today_name}\b", phrase, re.I)):
        same_day = parsed - datetime.timedelta(days=7)
        if same_day >= now:
            parsed = same_day
    return parsed, has_time


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


def _soundex(word):
    """The classic four character sound code: similar sounding names share one.

    Used only to decide when a rename has to be spelled out loud. "Charlie" and
    "Charley" are both C640, and read back as "rename Charlie to Charley" they
    sounded identical, so he could not hear what was being confirmed."""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    letters = [ch for ch in word.lower() if ch.isalpha()]
    if not letters:
        return ""
    result, last = letters[0].upper(), codes.get(letters[0], "")
    for ch in letters[1:]:
        code = codes.get(ch, "")
        if code and code != last:
            result += code
        if ch not in "hw":
            last = code
    return (result + "000")[:4]


def _letters(text):
    return " ".join(ch.upper() for ch in text if ch.isalpha())


def _spelled_difference(old, new):
    """Only the letters that changed: "E Y instead of I E" for Charlie to Charley.

    Spelling the whole word, "C, H, A, R, L, E, Y", was more than he needed to
    hear. The shared start and end are left out; a change longer than four
    letters is spelled whole, since a description of it would be no shorter."""
    o, n = old.casefold(), new.casefold()
    start = 0
    while start < min(len(o), len(n)) and o[start] == n[start]:
        start += 1
    end = 0
    while end < min(len(o), len(n)) - start and o[len(o) - 1 - end] == n[len(n) - 1 - end]:
        end += 1
    old_mid, new_mid = old[start:len(old) - end], new[start:len(new) - end]
    if len(new_mid) > 4 or len(old_mid) > 4:
        return f"spelled {_letters(new)}"
    if not old_mid:
        return f"with an added {_letters(new_mid)}"
    if not new_mid:
        return f"without the {_letters(old_mid)}"
    return f"{_letters(new_mid)} instead of {_letters(old_mid)}"


def _spelling_note(old_title, new_title):
    """", E Y instead of I E" for each changed word that sounds like the word it
    replaced, and nothing when the change can be heard."""
    old_words, new_words = old_title.split(), new_title.split()
    notes = [_spelled_difference(old, new) for old, new in zip(old_words, new_words)
             if old.casefold() != new.casefold() and _soundex(old) == _soundex(new)]
    return "".join(f", {note}" for note in notes)


def _title(text):
    """"career fair" becomes "Career Fair". Only an all lowercase title changes;
    one he capitalized himself, like "Isaiah lesson", is his to keep."""
    return string.capwords(text) if text == text.lower() else text


def _length_words(minutes):
    """"90 minutes", but "5 hours" rather than "300 minutes", which is arithmetic
    he should not have to do by ear."""
    if minutes < 120:
        return f"{minutes} minutes"
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return f"{hours} hours"
    if rest == 30:
        return f"{hours} and a half hours"
    return f"{hours} hours {rest} minutes"


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


# The calendar list barely changes and cost 497ms a call, every calendar question.
_CALENDAR_LIST_TTL_S = 300
_calendar_cache = {}


def _calendars(service):
    """Every calendar on his list, as {id: (name, selected, owned)}, remembered
    for a few minutes. See _fetch_calendars for what the entries mean."""
    now = time.monotonic()
    if _calendar_cache and now - _calendar_cache["at"] < _CALENDAR_LIST_TTL_S:
        return _calendar_cache["value"]
    value = _fetch_calendars(service)
    _calendar_cache.update(at=now, value=value)
    return value


def _fetch_calendars(service):
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
    existing = _miles_calendar_id(service)
    if existing:
        return existing
    created = service.calendars().insert(body={"summary": _WRITE_CALENDAR}).execute()["id"]
    # The remembered list does not have it yet.
    _calendar_cache.clear()
    return created


def _fetch_events(queries):
    """Run several events().list queries at once, as (calendar id, items, error).

    One after another, nine calendars took 1621ms, measured Sep 13 2026, against
    279ms for the slowest alone. The Google client is not safe to share between
    threads, because its HTTP connection is not, so each worker builds its own,
    which costs about 3ms."""
    def run(query):
        try:
            items = _service().events().list(**query).execute().get("items", [])
            return query["calendarId"], items, None
        except Exception as exc:
            return query["calendarId"], None, exc

    if not queries:
        return []
    with ThreadPoolExecutor(max_workers=min(len(queries), 10)) as pool:
        return list(pool.map(run, queries))


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
    queries = []
    for cid in calendars:
        query = {"calendarId": cid, "timeMin": _utc(start), "maxResults": _MAX_EVENTS,
                 "singleEvents": True, "orderBy": "startTime"}
        if end is not None:
            query["timeMax"] = _utc(end)
        queries.append(query)
    for cid, items, error in _fetch_events(queries):
        name, owned = calendars[cid]
        if error is not None:
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
        "Add a new event to Lethanial's MILES calendar right away. It is added "
        "immediately and read back to him word for word, and he can say undo. "
        "Say nothing before or after calling this. Call this when he asks you to "
        "schedule, book, or add something. start_time needs a day and a clock "
        "time, like 'tomorrow at 2pm'."
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
    summary = _title(summary)
    # A time range rather than a length: "for 300 minutes" left him working out
    # when the career fair ended.
    finish = _clock(end) if end.date() == start.date() else f"{_day_words(end, now)} at {_clock(end)}"
    # Added at once since Sep 14 2026: he found a question before every addition
    # too much. Adding is the one change undo fully reverses, so moving,
    # renaming and deleting still ask.
    _insert_event(summary, start, end)
    said = f"Added {summary} {_on_day(start, now)} from {_clock(start)} to {finish}."
    pending_action.announce(said)
    return f"{said} This was read back to him word for word; he can say undo."


# What the most recent addition put on the calendar, so "undo that" can take it
# back as a whole: one event, or every session of one plan.
_UNDO_WINDOW_S = 30 * 60
_recent_additions = {}


def _record_addition(calendar_id, event_id, title):
    """Everything added on one turn is one addition, so a whole plan undoes
    together while an earlier, separate addition is left alone."""
    turn = pending_action.current_turn()
    if _recent_additions.get("turn") != turn:
        _recent_additions.update(turn=turn, at=time.monotonic(), events=[])
    _recent_additions["events"].append((calendar_id, event_id, title))


def _insert_event(summary, start, end):
    """The actual write, remembered so it can be undone."""
    service = _service()
    body = {"summary": summary,
            "start": {"dateTime": start.astimezone().isoformat()},
            "end": {"dateTime": end.astimezone().isoformat()}}
    calendar_id = _write_calendar_id(service)
    created = service.events().insert(calendarId=calendar_id, body=body).execute()
    _record_addition(calendar_id, created.get("id"), summary)
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
        "move, reschedule, rename, shorten or lengthen one event. To fix a name on "
        "every event that has it, use rename_calendar_events instead. A new time alone, "
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
    new_title = _title(new_title) if new_title else new_title
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
        spelled = _spelling_note(name, new_title)
        clauses.append(f"rename it to {new_title}{spelled}" if clauses
                       else f"rename {name} {_on_day(start, now)} to {new_title}{spelled}")

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
    queries = [{"calendarId": cid, "timeMin": _utc(start), "timeMax": _utc(end),
                "maxResults": 50, "singleEvents": True, "orderBy": "startTime"}
               for cid in calendars]
    for cid, items, error in _fetch_events(queries):
        name, owned = calendars[cid]
        if error is not None:
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


# ── session planner ──
#
# Placing sessions is arithmetic, so code does it. On Sep 13 2026 Nova placed
# seven tutoring lessons in her head: offered 2 PM after being told "after
# three", put one in a class he had just described, and overlapped two others.
# Here the model only turns what he said into sessions; plan_sessions_on puts
# them on the calendar and nothing it returns can break a rule it was given.

# Defaults for a session given no times. Each is overridden per request, because
# they differ by person: a student's earliest start "depends on when they get out
# of school" and the latest "depends on the kid too, but not too late".
_SCHOOL_DAY_START = datetime.time(15, 0)
_LATEST_END = datetime.time(20, 0)
# He said weekends are fine but he would rather they be mornings. Holidays are
# treated the same, since the afternoon rule exists only because of school.
_NO_SCHOOL_MORNING = (datetime.time(9, 0), datetime.time(12, 0))
_SLOT_STEP = datetime.timedelta(minutes=15)


def _time_of(phrase, default):
    if not phrase:
        return default
    when, has_time = parse_when(phrase, datetime.datetime(2000, 1, 3))
    if not has_time:
        raise WhenError(f"{phrase!r} is not a time of day")
    return when.time()


def _first_free(day, session, taken, no_school, not_before, not_after):
    length = datetime.timedelta(minutes=session["minutes"])
    if no_school:
        windows = [_NO_SCHOOL_MORNING, (_NO_SCHOOL_MORNING[1], session["latest_end"])]
    else:
        windows = [(session["earliest"], session["latest_end"])]
    for open_at, close_at in windows:
        start = datetime.datetime.combine(day, open_at)
        close = min(datetime.datetime.combine(day, close_at), not_after)
        while start + length <= close:
            end = start + length
            # Touching is not overlapping: he tutors online, so one lesson can
            # start the minute another ends.
            if start >= not_before and all(end <= b_start or start >= b_end
                                           for b_start, b_end in taken):
                return start, end
            start += _SLOT_STEP
    return None


def plan_sessions_on(sessions, busy, days, no_school, not_before, not_after, soft_busy=()):
    """Place sessions. Pure: no Google and no clock, so every rule is testable.

    sessions are dicts of name, count, minutes, earliest and latest_end, the last
    two as datetime.time. busy is [(start, end)] he is committed to. days are the
    dates in range; no_school is the subset that are weekends or holidays.

    Returns (placed, unplaced): placed is [(name, start, end)] in time order,
    unplaced is [(name, how many did not fit)]. A session that does not fit is
    reported, never squeezed in against a rule.

    One session per name per day, which is the simplest rule that guarantees he
    never teaches the same student back to back. The sessions with the most
    repeats are placed first, because they have the least room to move.

    Each further session goes on the workable day farthest from the days that
    name already has, earliest on a tie. Spreading by fixed positions in the
    range failed on his real calendar: planned just after midnight, the range's
    last day had no usable time, the pick fell back to the next day in order, and
    two lessons for the same student landed today and tomorrow.

    soft_busy are events on calendars he follows. A session goes clear of them
    whenever any day has room, and only overlaps one when none does. He keeps
    club calendars so there is something to go to, and on Sep 14 2026 a plan
    that ignored them put lessons across a mini career fair, three info
    sessions and a workshop he might have wanted."""
    taken, placed, unplaced = sorted(busy), [], []
    soft = sorted((event[0], event[1]) for event in soft_busy)
    for session in sorted(sessions, key=lambda s: -s["count"]):
        used_days = []
        for _ in range(session["count"]):
            best = None
            # Clear of club events first; only if no day allows that, around
            # his commitments alone.
            for avoid in (taken + soft, taken):
                for day in days:
                    if day in used_days:
                        continue
                    slot = _first_free(day, session, avoid, day in no_school, not_before, not_after)
                    if slot is None:
                        continue
                    distance = min((abs((day - used).days) for used in used_days), default=0)
                    if best is None or distance > best[0]:
                        best = (distance, day, slot)
                if best is not None:
                    break
            if best is None:
                break
            _, day, slot = best
            placed.append((session["name"], *slot))
            taken = sorted(taken + [slot])
            used_days.append(day)
        if len(used_days) < session["count"]:
            unplaced.append((session["name"], session["count"] - len(used_days)))
    return sorted(placed, key=lambda p: p[1]), unplaced


def _club_overlaps(placed, soft_busy):
    """(session name, start, club event title) for each placed session that had
    to overlap an event on a calendar he follows."""
    return [(name, start, label) for name, start, end in placed
            for s_start, s_end, label in soft_busy if start < s_end and s_start < end]


def _overlap_notes(overlaps, now):
    """One short note per session that overlaps club events, naming at most two.

    Listed one overlap at a time, a two session plan read out five notes and a
    sixty word event title before the question."""
    by_session = {}
    for name, start, label in overlaps:
        by_session.setdefault((name, start), []).append(label)
    notes = []
    for (name, start), labels in by_session.items():
        shown = labels[:2]
        named = shown[0] if len(shown) == 1 else f"{shown[0]} and {shown[1]}"
        if len(labels) > 2:
            named = f"{shown[0]}, {shown[1]} and {len(labels) - 2} more"
        notes.append(f"{name} {_day_words(start, now)} at {_clock(start)} overlaps {named}")
    return notes


def _plan_sentence(placed, now):
    """The whole plan as one sentence, grouped by name so it can be followed by
    ear: "Added 4 sessions: Isaiah lesson Monday at 3 PM and Thursday at 3 PM;
    Andrew lesson..., all 90 minutes."."""
    by_name = {}
    for name, start, end in placed:
        by_name.setdefault(name, []).append(f"{_day_words(start, now)} at {_clock(start)}")
    groups = [f"{name} {times[0] if len(times) == 1 else ', '.join(times[:-1]) + ' and ' + times[-1]}"
              for name, times in by_name.items()]
    lengths = {int((end - start).total_seconds() // 60) for _, start, end in placed}
    suffix = f", all {_length_words(lengths.pop())}" if len(lengths) == 1 else ""
    count = len(placed)
    return f"Added {count} session{'' if count == 1 else 's'}: {'; '.join(groups)}{suffix}."


@tool(
    name="plan_sessions",
    description=(
        "Plan several sessions around Lethanial's week and propose them all as "
        "one question: tutoring lessons, study blocks, workouts, anything with a "
        "length and a number of times. Call this when he asks you to schedule or "
        "fit in several sessions. Turn what he says into the sessions list and "
        "let the code place them; never work out the times yourself. For each "
        "session give the event title, how many, how long, and when that person "
        "is available on school days if he said so. Anything he says is off limits "
        "that is not on his calendar, like a class or a career fair, goes in "
        "blocked, and so does any club event you know he should attend, such as a "
        "career fair or an info session with a company he wants to work for. "
        "Weekends and holidays default to mornings. It never books the same session twice in one day, avoids "
        "everything on his own calendars, and avoids club events whenever there is room. It adds "
        "them right away and reads the plan back to him word for word, and he can "
        "say undo. Say nothing before or after calling this."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "sessions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The event title, like 'Isaiah lesson'."},
                        "count": {"type": "integer", "description": "How many in the range."},
                        "minutes": {"type": "integer", "description": "Length of each, in minutes."},
                        "earliest": {"type": "string", "description": "Optional earliest start on school days, like '3:30pm'. Defaults to 3 PM."},
                        "latest_end": {"type": "string", "description": "Optional latest end, like '7pm'. Defaults to 8 PM."},
                    },
                    "required": ["name", "count", "minutes"],
                },
            },
            "blocked": {
                "type": "array",
                "description": "Times he said are off limits that are not on his calendar, like a class or a career fair.",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "Start, a day and a time, like 'monday 1pm'."},
                        "until": {"type": "string", "description": "End, like '6:30pm' for the same day."},
                    },
                    "required": ["from", "until"],
                },
            },
            "time_min": {"type": "string", "description": "Optional start of the range. Defaults to now."},
            "time_max": {"type": "string", "description": "Optional end of the range. Defaults to seven days later."},
        },
        "required": ["sessions"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def plan_sessions(sessions, blocked=None, time_min=None, time_max=None, now=None):
    now = now or datetime.datetime.now()
    start, end = resolve_window(time_min, time_max, now)
    start = max(start, now)
    if end is None:
        end = start + datetime.timedelta(days=7)
    if end <= start:
        raise WhenError("that whole range has already passed")

    specs = []
    for s in sessions:
        count, minutes = s.get("count"), s.get("minutes")
        if not isinstance(count, int) or count < 1 or not isinstance(minutes, int) or not 0 < minutes <= 600:
            raise WhenError(f"{s.get('name')!r} needs a count and a length in minutes")
        specs.append({"name": _title(s["name"]), "count": count, "minutes": minutes,
                      "earliest": _time_of(s.get("earliest"), _SCHOOL_DAY_START),
                      "latest_end": _time_of(s.get("latest_end"), _LATEST_END)})

    service = _service()
    calendars = _calendars(service)
    owned = [cid for cid, (_, selected, is_owned) in calendars.items() if selected and is_owned]
    holidays = [cid for cid in calendars if "#holiday@" in cid]
    followed = [cid for cid, (_, selected, is_owned) in calendars.items()
                if selected and not is_owned and cid not in holidays]
    queries = [{"calendarId": cid, "timeMin": _utc(start), "timeMax": _utc(end), "maxResults": 100,
                "singleEvents": True, "orderBy": "startTime"} for cid in owned + holidays + followed]

    busy, soft_busy, no_school = [], [], set()
    for cid, items, error in _fetch_events(queries):
        if error is not None:
            if cid in holidays or cid in followed:
                continue
            # A plan that cannot see one of his own calendars could land on top
            # of it, which is worse than no plan.
            raise RuntimeError(f"could not read {calendars[cid][0]}, so a plan could collide with it")
        for event in items:
            if cid in holidays:
                if "date" in event["start"]:
                    no_school.add(datetime.date.fromisoformat(event["start"]["date"]))
                continue
            if "dateTime" in event["start"]:
                event_start, event_end, _ = _event_bounds(event)
                if cid in followed:
                    soft_busy.append((event_start, event_end, event.get("summary", "a club event")))
                else:
                    busy.append((event_start, event_end))

    # Spoken limits that are on no calendar. On Sep 13 2026 he said "don't
    # schedule anything on Monday from 1 to 6:30, career fair", and a plan
    # built only from his calendar put three lessons in it. An end given as just
    # a time stays on the start's day, the same rule a moved event follows.
    for block in blocked or []:
        # A day named in a block includes today. Planned just after midnight on
        # a Monday, "monday 1pm" preferred the future and resolved to the Monday
        # after, so a career fair he had ruled out that very day was ignored and
        # three lessons went into it. Reading from a minute before today began
        # makes today's weekday mean today and tomorrow's mean tomorrow. A block
        # that says today, tomorrow or next is still read from now.
        base = now if _RELATIVE_TO_NOW.search(block["from"]) else (
            _start_of_day(now) - datetime.timedelta(minutes=1))
        block_start, has_time = parse_when(block["from"], base)
        if not has_time:
            block_start = _start_of_day(block_start)
        block_end = _resolve_new_start(block["until"], block_start, now)
        if block_end <= block_start:
            raise WhenError(f"the blocked time from {block['from']!r} to {block['until']!r} ends before it starts")
        busy.append((block_start, block_end))

    days = [start.date() + datetime.timedelta(days=i)
            for i in range((end.date() - start.date()).days + 1)]
    no_school |= {day for day in days if day.weekday() >= 5}
    placed, unplaced = plan_sessions_on(specs, busy, days, no_school, start, end, soft_busy)

    missing = "; ".join(f"{name}, {count} more" for name, count in unplaced)
    if not placed:
        return f"Nothing fits in that range with those rules. Tell him plainly: {missing}."

    # Added at once, like a single event, and the whole plan undoes together.
    added, failed = [], []
    for name, session_start, session_end in placed:
        try:
            _insert_event(name, session_start, session_end)
            added.append((name, session_start, session_end))
        except Exception:
            failed.append(f"{name} {_day_words(session_start, now)} at {_clock(session_start)}")

    # Everything is said by code, in one go: overlaps with club events and
    # anything that did not fit are part of what he hears, not left to Nova.
    notes = _overlap_notes(_club_overlaps(added, soft_busy), now)
    if unplaced:
        notes.append(f"Did not fit: {missing}")
    if failed:
        notes.append(f"Could not add: {'; '.join(failed)}")
    said = (". ".join(notes) + ". " if notes else "") + (_plan_sentence(added, now) if added else "")
    said = said.strip()
    pending_action.announce(said)
    return f"{said} This was read back to him word for word; he can say undo."


@tool(
    name="rename_calendar_events",
    description=(
        "Propose replacing a name or word in the title of every event on "
        "Lethanial's MILES calendar that contains it, like correcting Charlie to "
        "Charley on all of a student's lessons at once. This does not change "
        "them: it returns one short question to ask him. Call this whenever he "
        "asks to fix a name or a spelling, rather than renaming events one at a "
        "time. Say nothing before calling this. If he also wants the spelling "
        "kept from now on, call remember with the correct spelling too."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "find": {"type": "string", "description": "The word as it is now, like 'Charlie'."},
            "replace_with": {"type": "string", "description": "What it should be, like 'Charley'."},
            "time_min": {"type": "string", "description": "Optional start of the range. Defaults to now."},
            "time_max": {"type": "string", "description": "Optional end. Defaults to sixty days later."},
        },
        "required": ["find", "replace_with"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def rename_calendar_events(find, replace_with, time_min=None, time_max=None, now=None):
    now = now or datetime.datetime.now()
    start, end = resolve_window(time_min, time_max, now)
    start = max(start, now)
    if end is None:
        end = start + datetime.timedelta(days=60)
    service = _service()
    calendar_id = _miles_calendar_id(service)
    if calendar_id is None:
        raise EventLookupError("there is no MILES calendar yet, so there is nothing to rename")

    word = re.compile(rf"\b{re.escape(find)}\b", re.IGNORECASE)
    items = service.events().list(calendarId=calendar_id, timeMin=_utc(start), timeMax=_utc(end),
                                  singleEvents=True, orderBy="startTime", maxResults=250,
                                  ).execute().get("items", [])
    matches = [e for e in items if word.search(e.get("summary", ""))]
    if not matches:
        raise EventLookupError(f"no event on the MILES calendar has {find!r} in its title in that range")

    changes = [(e, word.sub(replace_with, e["summary"])) for e in matches]
    days = list(dict.fromkeys(_day_words(_event_bounds(e)[0], now) for e, _ in changes))
    when = days[0] if len(days) == 1 else ", ".join(days[:-1]) + " and " + days[-1]
    spelled = f", {_spelled_difference(find, replace_with)}" if _soundex(find) == _soundex(replace_with) else ""
    count = len(changes)
    question = (f"Rename {find} to {replace_with}{spelled}, on {count} "
                f"event{'' if count == 1 else 's'}: {when}?")

    def rename_all():
        done, failed = 0, []
        for event, new_title in changes:
            try:
                _service().events().patch(calendarId=calendar_id, eventId=event["id"],
                                          body={"summary": new_title}).execute()
                done += 1
            except Exception as exc:
                failed.append(f"{event.get('summary')} ({exc})")
        reply = f"Renamed {done} event{'' if done == 1 else 's'}."
        return reply + (f" Failed, not renamed: {'; '.join(failed)}." if failed else "")

    return _ask(pending_action.propose(question.rstrip("?"), rename_all))


@tool(
    name="undo_last_addition",
    description=(
        "Take back the most recent addition to Lethanial's calendar as a whole: "
        "the one event, or every session of the plan. Call this when he says undo "
        "that, take that off, or that was wrong, right after something was added. "
        "It works for thirty minutes; anything older, delete it by name instead. "
        "What was removed is read back to him word for word, so say nothing after."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def undo_last_addition():
    events = _recent_additions.get("events") or []
    if not events or time.monotonic() - _recent_additions.get("at", 0) > _UNDO_WINDOW_S:
        raise LookupError("nothing was added in the last thirty minutes to undo")
    removed, failed = [], []
    for calendar_id, event_id, title in events:
        try:
            _service().events().delete(calendarId=calendar_id, eventId=event_id).execute()
            removed.append(title)
        except Exception as exc:
            failed.append(f"{title} ({exc})")
    _recent_additions.clear()
    said = ""
    if removed:
        said = f"Removed {len(removed)} events." if len(removed) > 2 else f"Removed {_names(removed)}."
    if failed:
        said += f" Could not remove: {'; '.join(failed)}."
    said = said.strip()
    pending_action.announce(said)
    return said
