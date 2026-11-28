"""Offline interpretation of task scheduling. No tool registration, clock reads or writes."""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = ("january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december")

_WEEKDAY = r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?"
_MONTH = "|".join(f"{month}|{month[:3]}" for month in MONTHS)
_DAY = rf"(?:{_WEEKDAY}|today|tomorrow)"
_DATE_MARKER = rf"\b(?:{_DAY}|{_MONTH}|tonight|yesterday|week|weekend|month|year)\b|\d"
_START = re.compile(rf"\b(?:(?:starting|starts|beginning)(?: on)?|from(?= (?:(?:next|this|after|before) )?(?:{_DATE_MARKER})))\b", re.I)
_DAILY = re.compile(r"\b(?:as a daily task|every day|each day|daily)\b", re.I)
_OTHER_REPEAT = re.compile(
    rf"\b(?:(?:every|each) (?!day\b)\w+|alternate days|weekdays|weekends|weekly|fortnightly|"
    rf"biweekly|monthly|quarterly|annually|yearly|(?:{_WEEKDAY})s|"
    r"daily\s+(?:except|only)|(?:once|twice|three times) (?:a|per) (?:day|week|month)|"
    r"on the (?:first|second|third|fourth|last) of (?:each|the) (?:week|month)|"
    r"per (?:day|week|month))\b", re.I)
_WEEKDAY_LIST = re.compile(rf"\bon (?:{_WEEKDAY})\s*(?:,|and|or)\s*(?:{_WEEKDAY})\b", re.I)
_DEADLINE = re.compile(
    rf"\b(?:due(?! (?:to|diligence|process)\b)|deadline|no later than|by\s+(?:(?:{_DAY})\b|(?:this|next|"
    rf"the|end|tomorrow|tonight|then|noon|midnight|dawn|dusk|{_MONTH})\b|\d)|until|through)\b", re.I)
_CLOCK_TIME = re.compile(
    r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*[ap]\.?m\.?|noon|midnight|"
    r"at\s+(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s+o'clock)?|(?:at )?(?:half|quarter) (?:past|to) \w+|"
    r"at (?:lunchtime|bedtime|dawn|dusk|sunrise|sunset|daybreak|nightfall|night|dinner|breakfast|lunch))\b", re.I)
# Any count a person says, so "in four days" is as visible as "in 4 days". The
# quantity is any single word, not a list of number words, so "in thirteen days"
# is seen without anyone having to list thirteen.
_COUNT = r"\d+|\w+|a few|a couple of"
_TIMING_HINT = re.compile(
    rf"\b(?:{_DAY}|(?:next|this)\s+(?:week|month|year)|soon|later|tonight|yesterday|"
    rf"morning|afternoon|evening|weekend|by|before|after|from\s+now|"
    rf"(?:when|once|as soon as) (?:i|you|we|they)\b|"
    rf"(?:in|within)\s+(?:{_COUNT})\s+(?:days?|weeks?|months?|years?|hours?|minutes?)|"
    rf"(?:{_MONTH})\s+\d{{1,2}}|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH}))\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}\b", re.I)
_ADD = re.compile(r"^(?:please )?(?:add|create)\b\s+(?:(?:a |the )?task(?: to| for)?\s+)?", re.I)
_DAILY_INTRO = re.compile(r"^(?:a |the )?daily task (?:to |for )", re.I)
_EDIT = re.compile(r"^(?:please )?(?:make|change|edit|move|reschedule|update)\b", re.I)
_QUOTE = re.compile(r'"([^"]+)"|“([^”]+)”')
# A title such as Watch Friday Night Lights is ordinary task text, even unquoted.
_MEDIA = re.compile(
    rf"\b(?i:watch|read|listen to) +((?i:friday night lights)|(?i:{_WEEKDAY}) "
    r"(?!(?i:due|starting|starts|by|from|after|before|at)\b)[A-Z][a-z]+ "
    r"(?!(?i:due|starting|starts|by|from|after|before|at)\b)[A-Z][a-z]+)\b")
_YEARLESS = re.compile(
    rf"(?:(?P<month>{_MONTH}) (?P<day>\d{{1,2}})(?:st|nd|rd|th)?|"
    rf"(?P<reverse_day>\d{{1,2}})(?:st|nd|rd|th)? (?P<reverse_month>{_MONTH})|"
    r"(?P<month_number>\d{1,2})/(?P<day_number>\d{1,2}))")


@dataclass(frozen=True)
class SchedulingResult:
    outcome: Literal["supported", "clarify", "reject"]
    reason: str = ""
    recurrence: str | None = None
    start_date: str | None = None


# Words that say nothing about which task is meant. Short on purpose: "up",
# "out", "off" and "on" can flip a meaning ("turn on" and "turn off"), so they stay.
_FILLER = frozenset({"a", "an", "the", "to", "for", "my", "of", "our", "your", "some"})


def _stem(word):
    """A light stem so plural and tense endings do not matter. Both sides of a
    comparison go through it, so it only has to be consistent, not correct."""
    word = word.removesuffix("'s")
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "sl":
        word = word[:-1]
    return word.removesuffix("e") if len(word) > 3 else word


def content_words(text):
    """The words that carry a task's meaning, in order."""
    words = re.sub(r"[^a-z0-9' ]+", " ", (text or "").casefold().replace("\u2019", "'")).split()
    return [_stem(word) for word in words if word not in _FILLER]


def same_title(said, written):
    """True when `written` clearly means what was `said`: the same content words
    in the same order, ignoring case, punctuation, small filler words and plural
    or tense endings. Any added or dropped content word makes them differ."""
    words = content_words(said)
    return bool(words) and words == content_words(written)


def _normalize(text):
    return " ".join(text.split()).strip(" ,.;!?").casefold()


def _month_day(match, today):
    if match["month_number"]:
        month, day = int(match["month_number"]), int(match["day_number"])
    else:
        name = match["month"] or match["reverse_month"]
        month = next(i for i, item in enumerate(MONTHS, 1) if name in (item, item[:3]))
        day = int(match["day"] or match["reverse_day"])
    # Leap years are included in both directions, without coercing invalid dates.
    occurrences = []
    for year in range(max(1, today.year - 8), min(9999, today.year + 8) + 1):
        try:
            occurrences.append(date(year, month, day))
        except ValueError:
            continue
    if today in occurrences:
        return today
    if any(0 < (today - value).days <= 14 for value in occurrences):
        return SchedulingResult("clarify", "That month and day recently passed. Give the complete request with the intended date.")
    return next((value for value in occurrences if value > today), None)


def _resolve_first_day(phrase, today):
    if phrase == "today":
        return today
    if phrase == "tomorrow":
        return today + timedelta(days=1)
    weekday = re.fullmatch(rf"(?:(this|next) )?({_WEEKDAY})", phrase)
    if weekday:
        index = WEEKDAYS.index(next(item for item in WEEKDAYS if item.startswith(weekday[2])))
        advance = (index - today.weekday()) % 7
        qualifier = weekday[1]
        if (advance == 0 and qualifier != "next"
                or qualifier == "next" and 0 < advance < 7 - today.weekday()):
            return SchedulingResult("clarify", "That weekday is ambiguous. Give the full request with the exact first day.")
        return today + timedelta(days=advance or 7)
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", phrase):
        try:
            return date.fromisoformat(phrase)
        except ValueError:
            return None
    match = _YEARLESS.fullmatch(phrase)
    return _month_day(match, today) if match else None


def resolve_task_schedule(utterance, *, title, today, recurrence=None, starts=None):
    """Return a supported plan or an explicit no-write result from one utterance."""
    original = " ".join((utterance or "").split()).strip(" ,.;!?")
    if _EDIT.match(original):
        return SchedulingResult("reject", "Changing an existing start day or recurrence is unsupported.")
    command = _ADD.match(original)
    if not command or not title or not title.strip():
        return SchedulingResult("clarify", "Restate the full task request, including any timing.")

    # Mask quoted names and recognized media titles before examining clauses.
    body = original[command.end():]
    intro = _DAILY_INTRO.match(body)
    if intro:
        body = body[intro.end():]
    quoted = _QUOTE.sub(lambda m: " " * len(m[0]), body)
    if any(mark in quoted for mark in ('"', "“", "”")):
        return SchedulingResult("clarify", "Restate the complete task and its timing.")
    masked = _MEDIA.sub(
        lambda m: m[0][:-len(m[1])] + " " * len(m[1]), quoted)
    if _DEADLINE.search(masked):
        return SchedulingResult("reject", "Voice task creation does not currently support deadlines.")
    markers = list(_START.finditer(masked))
    before_start = masked[:markers[0].start()] if markers else masked
    if (_OTHER_REPEAT.search(masked) or _WEEKDAY_LIST.search(before_start)
            or recurrence not in (None, "daily")):
        return SchedulingResult("reject", "Only daily repetition is supported.")
    if _CLOCK_TIME.search(masked):
        return SchedulingResult("reject", "A start clock time is unsupported. Give only a first day.")

    daily = list(_DAILY.finditer(masked))
    if len(daily) + bool(intro) > 1 or len(markers) > 1:
        return SchedulingResult("clarify", "Restate the complete request with one recurrence and first day.")
    recurrence_span = daily[0] if daily else None
    first_span = markers[0] if markers else None
    is_daily = bool(intro or recurrence_span)
    title_end = min((span.start() for span in (recurrence_span, first_span) if span), default=len(body))
    first = None
    if first_span:
        if not is_daily:
            return SchedulingResult("reject", "A voice start date needs daily recurrence.")
        first_end = (recurrence_span.start()
                     if recurrence_span and recurrence_span.start() > first_span.start() else len(body))
        phrase = _normalize(body[first_span.end():first_end]).removesuffix(" please")
        if re.search(r"\b(?:or|and|after|before)\b", phrase):
            return SchedulingResult("clarify", "Give the complete request with one exact first day.")
        try:
            first = _resolve_first_day(phrase, today)
        except OverflowError:
            first = None
        if isinstance(first, SchedulingResult):
            return first
        if first is None or not 0 <= (first - today).days <= 365:
            return SchedulingResult("reject", "Use a real first day from today through the next 365 days.")
        supplied = _normalize(starts or "")
        full_clause = _normalize(body[first_span.start():first_end]).removesuffix(" please")
        if supplied not in {phrase, full_clause} and (supplied or phrase != "today"):
            return SchedulingResult("clarify", "Restate the full request with the first day you mean.")
    elif starts or recurrence and not is_daily:
        return SchedulingResult("clarify", "Restate the full scheduling request so I can verify it.")
    # Unknown tokens between scheduling clauses cannot be dropped into a default.
    if recurrence_span:
        end = (first_span.start() if first_span and first_span.start() > recurrence_span.start() else len(body))
        if _normalize(body[recurrence_span.end():end]) not in ("", "please"):
            return SchedulingResult("clarify", "Restate the full request and its timing.")
    if _TIMING_HINT.search(masked[:title_end]):
        return SchedulingResult("clarify", "Restate the request with the date's intended role.")
    original_title = _normalize(body[:title_end]).removesuffix(" please")
    if original_title.startswith(('"', "“")) and original_title.endswith(('"', "”")):
        original_title = _normalize(original_title[1:-1])
    if not same_title(original_title, title):
        return SchedulingResult("clarify", "I could not tell what the task is called. Say just the task, and then the timing if you have any.")
    return SchedulingResult("supported", recurrence="daily" if is_daily else None,
                            start_date=first.isoformat() if first and first != today else None)
