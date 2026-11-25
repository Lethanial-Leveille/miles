"""When a scheduled session is opened, nudged and settled. Decisions only.

Every function takes the time and the facts and returns what to do. No clock is
read and no database is touched here, so the tests can walk a whole night
minute by minute, a daylight saving night and a reboot included, and the timer
in sessions.tick only applies the answer.

The nightly review for day D (its 4am day), per the contract:
  created at D 21:30,
  pushed when created, reminded at D 22:30 if never opened,
  no push at all from midnight on,
  settled at D+1 03:00: missed if he never wrote in it, otherwise closed.

Settling compares against for_day, never against "is it 3am now". A Pi that was
off at three, or a timer that missed its run, settles the session on the next
run, whenever that is.

All times are naive local times, as the rest of the system stores them. A naive
comparison is what makes a daylight saving night come out right: 03:00 on the
morning the clocks change is still the clock reading 03:00.
"""

import datetime

from database import task_day

PUSH = "push"
REMIND = "remind"
CLOSE = "close"
MISS = "miss"


def _day(for_day):
    return datetime.date.fromisoformat(for_day)


def _on(day, clock):
    return datetime.datetime.combine(day, clock)


def settles_at(schedule, for_day):
    """The moment a session for for_day is settled. A settle time earlier in
    the clock than the open time is on the following morning."""
    day = _day(for_day)
    if schedule.settles_at <= schedule.opens_at:
        day += datetime.timedelta(days=1)
    return _on(day, schedule.settles_at)


def push_cutoff(for_day):
    """Midnight at the end of for_day. From then on nothing is pushed: a phone
    buzzing at 12:40 about a review is worse than no review."""
    return _on(_day(for_day) + datetime.timedelta(days=1), datetime.time(0))


def should_create(now, schedule, already_exists):
    """Whether the timer should create today's session now.

    Only from opens_at until midnight of the same evening. After midnight the
    4am day is still yesterday's, but a session made then could never be pushed
    and would settle as missed within hours, so it is not made; he can still
    start one by hand. already_exists is whether that day has one, by hand or
    by an earlier run: once a night, never twice."""
    if already_exists:
        return False
    today = _day(task_day(now))
    if now.date() != today:
        return False
    return now >= _on(today, schedule.opens_at)


def decide(now, schedule, session, wrote_in_it):
    """What to do now with one open scheduled session, or None.

    session is its row (for_day, created_at, opened_at, push_sent_at,
    reminder_sent_at); wrote_in_it is whether he ever sent a message.

    Only a session the timer made gets a push. The timer makes it at or after
    opens_at; a session he started by hand before that was not announced and
    is not announced late. Either way a session he has already opened is
    never pushed: he is in it."""
    for_day = session["for_day"]
    if now >= settles_at(schedule, for_day):
        return CLOSE if wrote_in_it else MISS
    if now >= push_cutoff(for_day) or session["opened_at"] is not None:
        return None

    opens = _on(_day(for_day), schedule.opens_at)
    created = datetime.datetime.fromisoformat(session["created_at"])
    if session["push_sent_at"] is None:
        return PUSH if created >= opens and now >= opens else None
    if session["reminder_sent_at"] is None and now >= _on(_day(for_day), schedule.remind_at):
        return REMIND
    return None
