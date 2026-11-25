"""What each calendar event means for being busy.

Not every event on his calendars is time he has committed. His own calendar and
MILES are; club and event calendars he follows are things he could go to, and
a holiday is a day, not a block. Before this, all of them counted as busy, so
a club talk made a free evening look full and "conflicts" were mostly clubs
against each other. The roles live in config.CALENDAR_ROLES by calendar id.

Three more rules sit on top:
- An event he says he attends (the calendar_attending table, by calendar and
  series) becomes blocking, so a weekly team meeting he really goes to counts.
- On a break day a class does not meet, and a recurring club event is a stale
  repeat. Both are marked "likely cancelled" and are not busy, never a
  conflict, never suggested, even a club event he attends: the series is
  attended, but the repeat on a break is not. The break is a UF_BREAKS range or
  any all day event on the holidays calendar.
- Touching events are not a conflict, and all day events never are.

Pure functions over event dicts as calendar_tools.events_for_app returns them,
so all of it is tested without Google. This module imports nothing that talks
to Google or the database.
"""

import datetime

from config import CALENDAR_ROLES, CLASS_TITLES, UF_BREAKS, UF_HOLIDAYS

BLOCKING = "blocking"
CLUB = "club"
HOLIDAY = "holiday"


def role_of(calendar_id, miles_id=None):
    """blocking, club or holiday. MILES is blocking by its id, and any calendar
    nobody has classified is blocking too, so a new calendar counts as busy
    until it is added to CALENDAR_ROLES."""
    if miles_id is not None and calendar_id == miles_id:
        return BLOCKING
    return CALENDAR_ROLES.get(calendar_id, BLOCKING)


def local_naive(raw):
    """An ISO string, aware or not, as naive local time like every other time
    the review keeps. A bare date is local midnight."""
    moment = datetime.datetime.fromisoformat(raw)
    return moment.astimezone().replace(tzinfo=None) if moment.tzinfo else moment


def event_span(event):
    """(start, end) as naive local datetimes. An all day event ends on its
    exclusive end date, as Google gives it."""
    return local_naive(event["start"]), local_naive(event["end"])


def holiday_days(events):
    """{date: name} for each day covered by a public holiday UF observes. The
    holidays calendar also lists observances (St. Patrick's Day, Tax Day), which
    are class days, so an event counts only when its description says "Public
    holiday" and its title is in UF_HOLIDAYS. A class on Thanksgiving Day is
    then known to be off even though UF_BREAKS says nothing about that date."""
    days = {}
    for e in events:
        if e.get("role") != HOLIDAY or not e["all_day"]:
            continue
        if ("public holiday" not in (e.get("description") or "").casefold()
                or e["title"].strip() not in UF_HOLIDAYS):
            continue
        first, last = event_span(e)
        day = first.date()
        while day < max(last.date(), first.date() + datetime.timedelta(days=1)):
            days.setdefault(day, e["title"])
            day += datetime.timedelta(days=1)
    return days


def break_name(day, holidays=None):
    """The name of the break a date falls in, or None. UF_BREAKS first, then
    the holidays calendar's all day events."""
    if isinstance(day, datetime.datetime):
        day = day.date()
    for first, last, name in UF_BREAKS:
        if datetime.date.fromisoformat(first) <= day <= datetime.date.fromisoformat(last):
            return name
    return (holidays or {}).get(day)


def is_class(title):
    """Whether a title is one of his classes: the whole title, ignoring case,
    is in CLASS_TITLES."""
    wanted = " ".join(str(title).split()).casefold()
    return any(wanted == " ".join(c.split()).casefold() for c in CLASS_TITLES)


def annotate(events, attending=frozenset(), miles_id=None):
    """Each event with its role, whether it counts as busy, and the mark the
    week view shows.

    Needs calendar_id (and series_id for attendance) on each event, which
    events_for_app now gives. attending is a set of (calendar id, series id).
    Returns new dicts; the input is untouched. Adds:
      role: blocking, club or holiday, after attendance is applied
      busy: whether it blocks time
      mark: "", "(club)", "(holiday)", "(attending)" or
            "(likely cancelled, <break name>)"
      cancelled: true for a likely cancelled event
    """
    out = []
    for e in events:
        role = role_of(e.get("calendar_id"), miles_id)
        out.append({**e, "role": role})
    holidays = holiday_days(out)

    result = []
    for e in out:
        role = e["role"]
        mark = {CLUB: "(club)", HOLIDAY: "(holiday)"}.get(role, "")
        series = e.get("series_id") or e["id"]
        if role == CLUB and ((e.get("calendar_id"), series) in attending
                             or (e.get("calendar_id"), e["id"]) in attending):
            role, mark = BLOCKING, "(attending)"
        cancelled = False
        if not e["all_day"]:
            name = break_name(local_naive(e["start"]), holidays)
            if name:
                stale_repeat = e["role"] == CLUB and e.get("repeating")
                stale_class = role == BLOCKING and e["role"] != CLUB and is_class(e["title"])
                if stale_repeat or stale_class:
                    cancelled = True
                    mark = f"(likely cancelled, {name})"
        busy = (role == BLOCKING and not cancelled and not e.get("transparent")
                and not e.get("declined"))
        result.append({**e, "role": role, "busy": busy, "mark": mark, "cancelled": cancelled})
    return result


def find_attendable(events, title):
    """Club events whose title contains `title` (ignoring case), from raw or
    annotated events, for "I attend". Only events on club calendars qualify:
    his own events already count."""
    wanted = " ".join(title.split()).casefold()
    return [e for e in events
            if role_of(e.get("calendar_id")) == CLUB and wanted in e["title"].casefold()]


def counts(event):
    """Whether an annotated event blocks his time."""
    return bool(event.get("busy", event.get("role", BLOCKING) == BLOCKING))


def find_conflicts(events):
    """Pairs of timed events that overlap, as dicts with a, b, start and end
    (the overlap itself), in time order.

    Only events that count as busy are considered, so club events, holidays and
    likely cancelled classes never conflict, and all day events are ignored.
    Touching is not overlapping: one ending at 4:30 and another starting at
    4:30 is back to back."""
    timed = []
    for e in events:
        if e["all_day"] or not counts(e):
            continue
        start, end = event_span(e)
        timed.append((start, end, e))
    timed.sort(key=lambda t: (t[0], t[1], t[2]["title"]))
    found = []
    for i, (a_start, a_end, a) in enumerate(timed):
        for b_start, b_end, b in timed[i + 1:]:
            if b_start >= a_end:
                break
            if b_start < a_end and a_start < b_end:
                found.append({"a": a, "b": b, "start": max(a_start, b_start),
                              "end": min(a_end, b_end)})
    return found
