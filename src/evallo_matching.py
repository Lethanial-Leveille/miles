"""Which Evallo calendar event a request means, decided without a browser.

The browser code reads every event in the week into CalendarEvent rows and
hands them here. Everything that decides "is this the one" lives in this file,
so it can be tested on plain data, and so a change to Evallo's page can break
how events are read but never quietly change which one gets picked.

The rule is deliberately strict. One event must match on all three of: the
title starting with the student's first name, the title naming Lethanial as the
tutor, and the start time to the minute. Zero matches or several are both
reported and nothing is clicked: marking the wrong student's session is worse
than marking none.

Canceled sessions are left out before counting. Evallo keeps them on the
calendar, greyed out and struck through, with the very same title, so a lesson
that was canceled and rebooked at the same time would otherwise read as two
matches and block every rebooked session.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

# Evallo shows "(GMT-05:00) Eastern Time - New York"; confirmed Nov 23 2026.
EVALLO_TZ = ZoneInfo("America/New_York")
TUTOR_MARKER = "<> lethanial leveille"
STATUSES = ("Completed", "No Show", "Canceled")


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    start: datetime          # aware, Eastern
    canceled: bool = False   # struck through on the calendar
    end: datetime | None = None
    index: int = 0           # its position in its day column, to click it

    @property
    def minutes(self):
        return int((self.end - self.start).total_seconds() // 60) if self.end else None


class MatchError(Exception):
    """Zero or several events fit. The message names what was seen, so the
    failure record says why without opening the screenshot."""


def normalize(text):
    """Lowercase, single spaces, no accents. Evallo's titles carry stray double
    spaces and non breaking spaces, and speech to text writes "Joaquin" for
    "Joaqu\u00edn"; none of that should decide a match."""
    text = unicodedata.normalize("NFKD", text.replace("\u00a0", " "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().casefold()


def to_eastern(start):
    """A naive time is taken as Eastern, the zone Evallo shows; an aware one is
    converted. Comparisons are always made in one zone."""
    if start.tzinfo is None:
        return start.replace(tzinfo=EVALLO_TZ)
    return start.astimezone(EVALLO_TZ)


def starts_with_name(text, first_name):
    """text begins with first_name as a whole word. The name must end where a
    word ends: not "Taylor" inside "Mayan", and not "Mary" inside "Mary-Kate"."""
    name = normalize(first_name)
    return bool(name) and re.match(rf"{re.escape(name)}(?![a-z0-9'\-])",
                                   normalize(text)) is not None


def title_matches(title, first_name):
    """The title starts with the first name and names Lethanial as the tutor:
    "Taylor <> Lethanial Leveille | Test Prep | 1-1". A session of someone
    else's does not match at all."""
    return TUTOR_MARKER in normalize(title) and starts_with_name(title, first_name)


def student_on_list(full_names, first_name):
    """The names on his Evallo students list that this first name could be.
    Empty means the student is not mapped to him yet."""
    return [n for n in full_names if starts_with_name(n, first_name)]


_CARD_TIME = re.compile(r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)", re.I)


def parse_card(day, text, canceled=False, index=0):
    """One calendar card into a CalendarEvent, or None when it has no time.

    day is the column's data-date, "2026-11-24". text is the card's visible
    text, "Taylor <> Lethanial Leveille | Test Prep | 1-1 04:15 PM - 05:45 PM";
    the title is everything before the times."""
    m = _CARD_TIME.search(text)
    if not m:
        return None
    d = date.fromisoformat(day)

    def at(clock):
        t = datetime.strptime(re.sub(r"\s+", " ", clock.upper()), "%I:%M %p")
        return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=EVALLO_TZ)
    start, end = at(m.group(1)), at(m.group(2))
    title = re.sub(r"\s+", " ", text[:m.start()]).strip()
    return CalendarEvent(title=title, start=start, end=end if end > start else None,
                         canceled=canceled, index=index)


def panel_date(start):
    """How the panel writes a date: "24 Nov, 2026"."""
    s = to_eastern(start)
    return f"{s.day} {s:%b}, {s.year}"


def panel_time(start):
    """How the panel writes a time: "04:15 PM"."""
    return f"{to_eastern(start):%I:%M %p}"


def same_minute(a, b):
    a, b = to_eastern(a), to_eastern(b)
    return a.replace(second=0, microsecond=0) == b.replace(second=0, microsecond=0)


def find_event(events, first_name, start):
    """The one live event for this student at this time, or MatchError."""
    if not normalize(first_name):
        raise MatchError("No student name was given.")
    live = [e for e in events if not e.canceled]
    hits = [e for e in live
            if title_matches(e.title, first_name) and same_minute(e.start, start)]
    if len(hits) == 1:
        return hits[0]
    when = to_eastern(start).strftime("%a %b %-d %-I:%M %p")
    if not hits:
        # Name what was nearby, so the record shows whether the name or the
        # time was off.
        same_name = [e for e in live if title_matches(e.title, first_name)]
        seen = "; ".join(f"{e.title} at {to_eastern(e.start):%a %-I:%M %p}"
                         for e in same_name) or "none with that name this week"
        raise MatchError(f"No session for {first_name} at {when}. Seen: {seen}.")
    raise MatchError(f"{len(hits)} sessions match {first_name} at {when}: "
                     + "; ".join(e.title for e in hits) + ". Not guessing.")


def check_status(status):
    """The requested status exactly as Evallo spells it, or ValueError."""
    for s in STATUSES:
        if normalize(s) == normalize(status):
            return s
    raise ValueError(f"Status must be one of {', '.join(STATUSES)}; got {status!r}.")
