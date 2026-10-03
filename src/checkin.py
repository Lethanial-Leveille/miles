"""The evening check in: what is off about his day, said once, or nothing.

He asked on Oct 2 2026 for Nova to notice instead of asking. "Did you do X?"
every night is a survey he would tune out within a week, so everything here is
read from a service, and only what is off is said. On a good day that is
nothing at all.

This module is the rules alone. Every function takes the data and the time and
returns a Point or None, with no network and no database, so the tests can
replay a real week and a change to a rule shows exactly which nights it moves.
The plan and the measurements behind it are in BACKEND_TODO.md, under
"Accountability: she notices, she does not ask".

Each Point carries a key. A key that has been said is not said again, which is
how "the gym week is lost" is said once on Friday rather than every night
until Monday. The key, not a flag, carries the forgetting: a new week or a
recovered night produces a new key, so nothing has to be cleared.
"""

import datetime
import re
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import anthropic

import alerts
import calendar_tools
import hevy_tools
import oura_tools
from config import CHECKIN, CHECKIN_MODEL
from database import checkin_ran, checkin_said, record_checkin
from parsing import words_for_number

# summary is what Claude reads when the point is folded into his turn, and
# phrases in her own words. text is the plain sentence spoken if it is not.
Point = namedtuple("Point", "key summary text")

GYM_TARGET = 4

# A gap is said once, on its second day, so the walk back only has to find
# where it began. Two weeks covers any gap still worth naming by its weekday.
GAP_LOOKBACK_DAYS = 14

# His target is eight hours asleep, nine in bed (Oct 2 2026). Asleep is the
# one judged: Oura's time in bed runs about an hour over it, so judging both
# would say the same thing twice.
SLEEP_TARGET_H = 8.0
SLEEP_NIGHTS = 3

# Tomorrow's first commitment counts only inside the working day. Without the
# lower bound a 12am block counted as tomorrow's first class in the dry run.
DAY_STARTS, DAY_ENDS = datetime.time(6), datetime.time(18)

# Waking up and getting to a Zoom class: half an hour on top of the eight.
WAKE_TO_START = datetime.timedelta(minutes=30)

# The tomorrow line is said only when the latest bedtime for eight hours is
# closer than this. Further off, it is not yet anything he needs to hear.
BEDTIME_WARNING = datetime.timedelta(hours=2)


def day_lived(now):
    """The day he just lived. At 12:30am the new date has barely started, and
    a check in reading it would say he had not trained "today"."""
    return now.date() if now.hour >= 12 else now.date() - datetime.timedelta(days=1)


def _week_start(day):
    return day - datetime.timedelta(days=day.weekday())


def _clock(dt):
    """"10:40am", for Claude, which spells it for speech."""
    return dt.strftime("%-I:%M%p").lower()


def _clock_words(dt):
    """"ten forty", for the fallback line, which goes to speech as is."""
    hour = words_for_number(dt.hour % 12 or 12)
    if dt.minute == 0:
        return f"{hour} o'clock"
    if dt.minute < 10:
        return f"{hour} oh {words_for_number(dt.minute)}"
    return f"{hour} {words_for_number(dt.minute)}"


def _gap_start(workout_days, day):
    """The first day of the run of missed days ending on `day`, or None if he
    trained on `day` or the day before. One missed day is a break; his rule
    (Oct 2 2026) is that the second in a row is worth saying."""
    trained = set(workout_days)
    if day in trained or day - datetime.timedelta(days=1) in trained:
        return None
    start = day - datetime.timedelta(days=1)
    # Bounded, so a log that has gone quiet for months does not walk forever.
    for _ in range(GAP_LOOKBACK_DAYS):
        if start - datetime.timedelta(days=1) in trained:
            break
        start -= datetime.timedelta(days=1)
    return start


def gym_point(workout_days, day, said=frozenset()):
    """One gym line a night at most, from the week's pace first, then gaps.

    Pace: said only when he has run out of room, meaning the days left in the
    week (through Sunday) are no more than the sessions he still needs. Behind
    on a prorated pace would have spoken most nights of his last five months of
    Hevy, and a line heard every night stops meaning anything. A lost week is
    said once and then the gym is quiet until Monday; "you missed again" after
    "the week is done" is only nagging.

    Gaps: two missed days in a row, once per gap. Four sessions a week leaves
    three days off, so a single missed day is a break, never a miss. Once four
    are done the week is quiet, gaps included."""
    week = _week_start(day)
    done = sum(1 for d in workout_days if week <= d <= day)
    needed = GYM_TARGET - done
    days_left = 6 - day.weekday()
    if needed <= 0:
        return None

    done_words = words_for_number(done)
    if needed > days_left:
        key = f"gym_lost:{week}"
        if key in said:
            return None
        return Point(key,
                     f"Gym this week: {done_words} of {GYM_TARGET} sessions, and "
                     f"four is out of reach before Sunday. Say it once, plainly; "
                     f"the week starts over Monday.",
                     f"The gym week finishes at {done_words}. Monday starts over.")
    if needed == days_left and f"gym_tight:{week}" not in said:
        days_words = words_for_number(days_left)
        return Point(f"gym_tight:{week}",
                     f"Gym this week: {done_words} of {GYM_TARGET} sessions. He is "
                     f"behind: reaching four now means training every one of the "
                     f"{days_words} days left, with no day off.",
                     f"{done_words.capitalize()} gym sessions this week, so it's "
                     f"every day from here to make four.")

    gap = _gap_start(workout_days, day)
    if gap is None or f"gym_gap:{gap}" in said:
        return None
    missed = (day - gap).days + 1
    return Point(f"gym_gap:{gap}",
                 f"Gym: no workout for {words_for_number(missed)} days in a row, "
                 f"since {gap:%A}. One day off is a break; more in a row is "
                 f"worth a word. {done_words.capitalize()} of {GYM_TARGET} this week.",
                 f"That's {words_for_number(missed)} days in a row without the gym.")


def _hours(nights, end):
    """Hours asleep for the SLEEP_NIGHTS nights ending on `end`. None if any is
    missing, since a ring left on the charger is not a short night."""
    hours = []
    for back in range(SLEEP_NIGHTS):
        seconds = nights.get(end - datetime.timedelta(days=back))
        if seconds is None:
            return None
        hours.append(seconds / 3600)
    return hours


def _low(hours):
    """Under target on average and on most nights. The average alone let one
    bad night through: nine, nine and five average under eight, and that is
    the one bad night he said never to raise."""
    short = sum(1 for h in hours if h < SLEEP_TARGET_H)
    return sum(hours) / len(hours) < SLEEP_TARGET_H and short >= 2


def _duration_words(total_minutes):
    hours, minutes = divmod(total_minutes, 60)
    parts = []
    if hours:
        parts.append(f"{words_for_number(hours)} hour{'' if hours == 1 else 's'}")
    if minutes or not hours:
        parts.append(f"{words_for_number(minutes)} minute{'' if minutes == 1 else 's'}")
    return " ".join(parts)


def sleep_point(nights, day, said=frozenset()):
    """A trend, never one bad night: the last three nights under eight hours
    asleep, on average and on at least two of them.

    `nights` maps the day a night ended (Oura's "day") to seconds asleep in
    the main sleep. At the check in the latest is the night that ended this
    morning, filed under `day`.

    Said once per low stretch. The key is the last night his sleep was not
    low, so it changes only when he recovers and slips again; three short
    weeks in a row are one thing said, not twenty one."""
    hours = _hours(nights, day)
    if hours is None or not _low(hours):
        return None
    average = sum(hours) / len(hours)

    recovered = None
    for night in sorted(nights, reverse=True):
        if night >= day:
            continue
        earlier = _hours(nights, night)
        if earlier is not None and not _low(earlier):
            recovered = night
            break
    key = f"sleep_low:{recovered or min(nights)}"
    if key in said:
        return None

    spoken = _duration_words(round(average * 60))
    # The gap is given, because left to work it out Haiku said "two hours nine
    # minutes short" of a gap that was one hour nine (Oct 2 2026).
    short = _duration_words(round(SLEEP_TARGET_H * 60) - round(average * 60))
    return Point(key,
                 f"Sleep: {spoken} asleep a night, averaged over the last three "
                 f"nights. That is {short} short of his eight hour target. A "
                 f"trend, not one bad night.",
                 f"You've averaged {spoken} of sleep over the last three nights.")


def tomorrow_point(events, now, said=frozenset()):
    """Tomorrow's first commitment, only when it already limits tonight.

    `events` is [(start, title)] from the calendars he owns, as aware
    datetimes. Only a timed event between 6am and 6pm counts; all day events
    are birthdays and bills."""
    tomorrow = day_lived(now) + datetime.timedelta(days=1)
    candidates = [(start, title) for start, title in events
                  if start.date() == tomorrow and DAY_STARTS <= start.time() < DAY_ENDS]
    if not candidates:
        return None
    start, title = min(candidates)
    asleep_by = start - datetime.timedelta(hours=SLEEP_TARGET_H) - WAKE_TO_START
    if asleep_by - now >= BEDTIME_WARNING:
        return None

    key = f"tomorrow:{tomorrow}"
    if key in said:
        return None
    return Point(key,
                 f"Tomorrow: {title} at {_clock(start)}. For eight hours of "
                 f"sleep and half an hour to get up, he needs to be asleep by "
                 f"{_clock(asleep_by)}.",
                 f"{title} is at {_clock_words(start)} tomorrow, so asleep by "
                 f"{_clock_words(asleep_by)} for eight hours.")


def points(now, workout_days, nights, events, said=frozenset()):
    """Everything worth saying tonight, in the order it would be said. A source
    passed as None failed to load, and its rule says nothing tonight."""
    day = day_lived(now)
    found = (gym_point(workout_days, day, said) if workout_days is not None else None,
             sleep_point(nights, day, said) if nights is not None else None,
             tomorrow_point(events, now, said) if events is not None else None)
    return [p for p in found if p is not None]


# ── what the rules read ──
# Each source is fetched with the same helpers its tools use, and fails on its
# own: a source that is down is skipped and named in checkin_log, never read as
# an empty week, which would have her say he skipped the gym when Hevy was down.

# Twenty workouts is two weeks or more at his pace, enough to find any gap.
_WORKOUT_PAGES = 2

# The sleep rule walks back to the last night he was not low, so it needs more
# than the three nights it averages.
_SLEEP_DAYS = 14


def _workout_days():
    workouts = hevy_tools._all_pages("workouts", "workouts", _WORKOUT_PAGES)
    return [hevy_tools._local(w["start_time"]).date() for w in workouts]


def _main_sleeps(today):
    """{day: seconds asleep} for the longest main sleep filed under each day.
    Naps are left out; a nap does not make up a short night in his target."""
    periods = oura_tools._get("sleep",
                              start_date=(today - datetime.timedelta(days=_SLEEP_DAYS)).isoformat(),
                              end_date=(today + datetime.timedelta(days=1)).isoformat())
    nights = {}
    for p in periods:
        seconds = p.get("total_sleep_duration")
        if p.get("type") != "long_sleep" or not isinstance(seconds, (int, float)):
            continue
        day = datetime.date.fromisoformat(p["day"])
        nights[day] = max(seconds, nights.get(day, 0))
    return nights


def _tomorrow_events(now):
    """[(start, title)] for tomorrow's working day, on the calendars he owns.
    Club calendars he follows are things he could go to, not commitments."""
    service = calendar_tools._service()
    owned = [cid for cid, (_, _, is_owner) in calendar_tools._calendars(service).items()
             if is_owner]
    tomorrow = day_lived(now) + datetime.timedelta(days=1)
    start = datetime.datetime.combine(tomorrow, DAY_STARTS).astimezone()
    end = datetime.datetime.combine(tomorrow, DAY_ENDS).astimezone()
    queries = [{"calendarId": cid, "timeMin": start.isoformat(), "timeMax": end.isoformat(),
                "singleEvents": True} for cid in owned]
    events = []
    for _, items, error in calendar_tools._fetch_events(queries):
        if error is not None:
            raise error
        events += [(calendar_tools._event_start(e), e.get("summary", "Untitled"))
                   for e in items if "dateTime" in e["start"]]
    return events


def gather(now):
    """(workout_days, nights, events, skipped). A failed source comes back
    None and named in skipped, and points() drops its rule for the night."""
    skipped = []
    found = {}
    for name, fetch in (("hevy", _workout_days),
                        ("oura", lambda: _main_sleeps(day_lived(now))),
                        ("calendar", lambda: _tomorrow_events(now))):
        try:
            found[name] = fetch()
        except Exception as exc:
            print(f"Check in: {name} unavailable: {exc}", flush=True)
            found[name] = None
            skipped.append(name)
    return found["hevy"], found["oura"], found["calendar"], skipped


def run(now=None):
    """Tonight's points, once a night. [] if it already ran, or nothing is off.

    Recorded before it is returned: if speaking it fails, a second attempt
    tomorrow is a different night anyway, and saying it twice tonight is the
    failure he would notice."""
    now = now or datetime.datetime.now().astimezone()
    night = day_lived(now)
    if checkin_ran(night):
        return []

    workout_days, nights, events, skipped = gather(now)
    tonight = points(now, workout_days, nights, events, checkin_said())
    record_checkin(night, tonight, skipped)
    return tonight


# ── wording and delivery ──

# Nights run late: he was usually asleep between 1 and 4am in September 2026.
# Before 11:30pm the day is not over; after 4am it is the next one.
WINDOW_OPENS, WINDOW_CLOSES = datetime.time(23, 30), datetime.time(4)

_PROMPT = (
    "You are Nova, Lethanial's voice assistant. You just answered him, and now "
    "add a short evening check in. The facts below were already checked; say "
    "each one plainly, as it is. Do not soften it into good news, and do not "
    "comment on it (no \"worth noticing\", no \"keep it up\"). One or two "
    "spoken sentences, under thirty five words in all. Warm and direct, a "
    "little dry. Never ask whether he did something. Spell every number and "
    "time as words, with no hyphens: \"fifty one\", not \"fifty-one\". "
    "No dashes of any kind. Use his name at most once. Plain text only."
)

_client = anthropic.Anthropic(timeout=10.0, max_retries=1)

# One worker: a check in is one job a night, and a second can only be a
# duplicate.
_pool = ThreadPoolExecutor(max_workers=1)
_started = set()
_started_lock = threading.Lock()


def in_window(now):
    return now.time() >= WINDOW_OPENS or now.time() < WINDOW_CLOSES


# Past this a reply has wandered from two sentences; the written lines are
# safer than a ramble.
_MAX_WORDS = 45


def _checked(text):
    """Her wording, made safe to speak, or None to fall back.

    Dashes are swapped in code because the prompt alone did not hold them on
    Oct 2 2026. Digits mean a number she was told to spell came out raw, or
    one she computed herself: in the same samples she said "two hours nine
    minutes short" of a gap that was one hour nine."""
    text = re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    if re.search(r"\d", text) or len(text.split()) > _MAX_WORDS:
        return None
    return text


def phrase(found):
    """The points in her words, or the written lines if Claude is unreachable.
    The facts were decided in code; this only chooses the sentence."""
    try:
        reply = _client.messages.create(
            model=CHECKIN_MODEL, max_tokens=200, system=_PROMPT,
            messages=[{"role": "user",
                       "content": "\n".join(p.summary for p in found)}])
        text = _checked("".join(b.text for b in reply.content if b.type == "text").strip())
        if text:
            return text
        print("Check in: wording failed the check, using the written lines", flush=True)
    except Exception as exc:
        print(f"Check in: wording failed, using the written lines: {exc}", flush=True)
    return " ".join(p.text for p in found)


def _prepare(now):
    found = run(now)
    if not found:
        return None
    return phrase(found), " ".join(p.summary for p in found)


def start(now=None):
    """Begin tonight's check in in the background, or None if there is none to
    do. Called only after his voice is verified: these are his health and gym
    numbers, and a guest's turn must never start one."""
    now = now or datetime.datetime.now().astimezone()
    if not CHECKIN or not in_window(now):
        return None
    night = day_lived(now)
    with _started_lock:
        if night in _started or checkin_ran(night):
            return None
        _started.add(night)
    return _pool.submit(_prepare, now)


def _fire(future):
    try:
        result = future.result()
    except Exception as exc:
        print(f"Check in failed: {exc}", flush=True)
        return
    if result is not None:
        text, summary = result
        print(f"Check in: {text}", flush=True)
        alerts.fire("checkin", text, summary)


def deliver(future):
    """Queue the check in after her reply. Ready now, it is queued at once and
    the loop speaks it before the follow up window. Still fetching, it is
    queued when it finishes, and the alert queue delivers it at the next safe
    moment, the way a reminder that fires mid turn is."""
    if future is None:
        return
    if future.done():
        _fire(future)
    else:
        future.add_done_callback(_fire)
