"""When the nightly review is made, pushed, reminded and settled.

The contract's rules, walked through a night: created at 21:30 once, reminded
at 22:30 only if never opened, nothing pushed after midnight, settled at 03:00
the next morning as missed or closed. Pure: every time is passed in."""

from datetime import datetime, time

import pytest

import session_schedule as sched
from session_types import Schedule

NIGHTLY = Schedule(opens_at=time(21, 30), remind_at=time(22, 30), settles_at=time(3, 0))


def _session(for_day="2026-11-24", created_at="2026-11-24T21:30:04.118201",
             opened_at=None, push_sent_at=None, reminder_sent_at=None):
    return {"for_day": for_day, "created_at": created_at, "opened_at": opened_at,
            "push_sent_at": push_sent_at, "reminder_sent_at": reminder_sent_at}


# ── creating ──

@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 11, 24, 21, 29), False),
    (datetime(2026, 11, 24, 21, 30), True),
    (datetime(2026, 11, 24, 23, 55), True),
    # After midnight the 4am day is still the 24th, but it is too late to open one.
    (datetime(2026, 11, 25, 0, 5), False),
    (datetime(2026, 11, 25, 3, 0), False),
    (datetime(2026, 11, 24, 12, 0), False),
])
def test_created_from_half_past_nine_until_midnight(now, expected):
    assert sched.should_create(now, NIGHTLY, already_exists=False) is expected


def test_created_once_a_night_never_twice():
    every_five_minutes = [datetime(2026, 11, 24, 21, 30 + m) for m in range(0, 30, 5)]
    made = []
    for now in every_five_minutes:
        if sched.should_create(now, NIGHTLY, already_exists=bool(made)):
            made.append(now)
    assert made == [datetime(2026, 11, 24, 21, 30)]


# ── pushing and reminding ──

def test_a_new_timer_session_is_pushed():
    assert sched.decide(datetime(2026, 11, 24, 21, 30, 5), NIGHTLY, _session(), False) == sched.PUSH


def test_pushed_once():
    s = _session(push_sent_at="2026-11-24T21:30:06")
    assert sched.decide(datetime(2026, 11, 24, 21, 35), NIGHTLY, s, False) is None


def test_reminded_at_half_past_ten_when_never_opened():
    s = _session(push_sent_at="2026-11-24T21:30:06")
    assert sched.decide(datetime(2026, 11, 24, 22, 29), NIGHTLY, s, False) is None
    assert sched.decide(datetime(2026, 11, 24, 22, 30), NIGHTLY, s, False) == sched.REMIND


def test_no_reminder_once_opened():
    s = _session(push_sent_at="2026-11-24T21:30:06", opened_at="2026-11-24T21:41:00")
    assert sched.decide(datetime(2026, 11, 24, 22, 30), NIGHTLY, s, False) is None


def test_reminded_once():
    s = _session(push_sent_at="2026-11-24T21:30:06", reminder_sent_at="2026-11-24T22:30:02")
    assert sched.decide(datetime(2026, 11, 24, 22, 35), NIGHTLY, s, False) is None


def test_nothing_is_pushed_after_midnight():
    """A Pi that was off from 21:00 to 00:10 must not buzz his phone at 00:10."""
    unpushed = _session()
    unreminded = _session(push_sent_at="2026-11-24T21:30:06")
    for now in (datetime(2026, 11, 25, 0, 0), datetime(2026, 11, 25, 0, 10),
                datetime(2026, 11, 25, 2, 55)):
        assert sched.decide(now, NIGHTLY, unpushed, False) is None
        assert sched.decide(now, NIGHTLY, unreminded, False) is None


def test_a_session_started_by_hand_before_half_past_nine_is_never_pushed():
    s = _session(created_at="2026-11-24T20:15:00")
    for now in (datetime(2026, 11, 24, 21, 30), datetime(2026, 11, 24, 22, 30)):
        assert sched.decide(now, NIGHTLY, s, False) is None


# ── settling ──

def test_settled_at_three_as_missed_when_he_never_wrote():
    s = _session(push_sent_at="2026-11-24T21:30:06", reminder_sent_at="2026-11-24T22:30:02")
    assert sched.decide(datetime(2026, 11, 25, 2, 59), NIGHTLY, s, False) is None
    assert sched.decide(datetime(2026, 11, 25, 3, 0), NIGHTLY, s, False) == sched.MISS


def test_opened_but_silent_is_still_missed():
    s = _session(opened_at="2026-11-24T21:40:00")
    assert sched.decide(datetime(2026, 11, 25, 3, 0), NIGHTLY, s, False) == sched.MISS


def test_settled_at_three_as_closed_when_he_wrote():
    s = _session(opened_at="2026-11-24T21:40:00")
    assert sched.decide(datetime(2026, 11, 25, 3, 0), NIGHTLY, s, True) == sched.CLOSE


def test_a_reboot_past_four_still_settles_yesterdays():
    """At 04:10 the 4am day is already the 25th; the session is judged by its
    own for_day, so it is settled, not left open forever."""
    s = _session(push_sent_at="2026-11-24T21:30:06")
    assert sched.decide(datetime(2026, 11, 25, 4, 10), NIGHTLY, s, False) == sched.MISS
    assert sched.decide(datetime(2026, 11, 27, 9, 0), NIGHTLY, s, True) == sched.CLOSE


def test_a_daylight_saving_night():
    """Clocks fall back at 2am on Nov 1 2026, so 1:30 happens twice. Neither
    1:30 settles Oct 31's review; 3:00 does, and the next night opens normally."""
    s = _session(for_day="2026-10-31", created_at="2026-10-31T21:30:01",
                 push_sent_at="2026-10-31T21:30:02")
    assert sched.decide(datetime(2026, 11, 1, 1, 30), NIGHTLY, s, False) is None
    assert sched.decide(datetime(2026, 11, 1, 1, 30, 30), NIGHTLY, s, False) is None
    assert sched.decide(datetime(2026, 11, 1, 3, 0), NIGHTLY, s, False) == sched.MISS
    assert sched.should_create(datetime(2026, 11, 1, 21, 30), NIGHTLY, already_exists=False)
    # And the spring night, when 2:00 to 3:00 never happens: 3:00 still settles.
    spring = _session(for_day="2026-03-07", created_at="2026-03-07T21:30:01")
    assert sched.decide(datetime(2026, 3, 8, 3, 0), NIGHTLY, spring, False) == sched.MISS


def test_settle_moment_is_the_next_morning():
    assert sched.settles_at(NIGHTLY, "2026-11-24") == datetime(2026, 11, 25, 3, 0)
    assert sched.push_cutoff("2026-11-24") == datetime(2026, 11, 25, 0, 0)


def test_a_late_first_push_is_not_followed_by_the_reminder():
    # The Pi came back at 22:45: the first push went then. It already is the
    # reminder, so the next tick must not buzz again five minutes later.
    s = _session(created_at="2026-11-24T22:45:00", push_sent_at="2026-11-24T22:45:02")
    assert sched.decide(datetime(2026, 11, 24, 22, 50), NIGHTLY, s, False) is None
    assert sched.decide(datetime(2026, 11, 24, 23, 30), NIGHTLY, s, False) is None
