"""Reminders survive a restart, which they did not before.

set_reminder used to spawn a threading.Thread that slept until the due time.
That made the thread the real state and the row a record of it, so the row
survived a restart and the mechanism did not. Nothing scanned the table at
boot, so a reminder set for tomorrow morning was silently dropped by any deploy
or crash, and Restart=always makes both routine.

The fix is not a boot rearm. It is that the table became the only state, so
there is nothing held in memory to lose and a restart is just the next poll.
These tests pin that property rather than the mechanism, because the mechanism
is what changed.
"""

from datetime import datetime, timedelta

import pytest

import actions
import alerts
import database


@pytest.fixture
def reminders(db, monkeypatch):
    """A throwaway database that BOTH modules point at.

    The conftest fixture redirects database.DB_PATH, but actions.py holds its
    own `from config import DB_PATH` for the insert, so redirecting one leaves
    set_reminder writing to the real ~/miles/data/miles.db."""
    monkeypatch.setattr(actions, "DB_PATH", database.DB_PATH)
    alerts.clear()
    yield database
    alerts.clear()


def _iso(**delta):
    return (datetime.now() + timedelta(**delta)).isoformat()


# ── the regression ──

def test_a_reminder_set_before_a_restart_still_fires(reminders):
    """The bug, stated as a test.

    Nothing here re arms anything. The reminder was written by one "process",
    the poll happens in another, and it fires because the row is the state."""
    actions.set_reminder("take the trash out", _iso(seconds=-30))

    assert actions.poll_reminders() == 1
    fired = alerts.take_for_speech()
    assert len(fired) == 1
    assert "take the trash out" in fired[0].text


def test_a_future_reminder_does_not_fire_yet(reminders):
    actions.set_reminder("call mom", _iso(hours=2))
    assert actions.poll_reminders() == 0
    assert alerts.pending_count() == 0


def test_a_reminder_with_no_due_time_never_fires(reminders):
    """The tool description explicitly supports saving without a time. Those
    are notes, not alarms, and a poller that fired them would announce every
    undated reminder at once on the next tick."""
    actions.set_reminder("look into the mmWave sensor")
    assert actions.poll_reminders() == 0


# ── exactly once ──

def test_a_reminder_fires_only_once(reminders):
    actions.set_reminder("stretch", _iso(seconds=-5))
    assert actions.poll_reminders() == 1
    assert actions.poll_reminders() == 0
    assert actions.poll_reminders() == 0


def test_completion_is_the_claim(reminders):
    """complete_reminder reporting whether it changed a row is what makes the
    UPDATE double as a lock. Two passes cannot both win the same reminder."""
    actions.set_reminder("drink water", _iso(seconds=-5))
    (reminder_id, _, _, _), = database.due_reminders(datetime.now().isoformat())

    assert database.complete_reminder(reminder_id) is True
    assert database.complete_reminder(reminder_id) is False


def test_identical_reminders_both_fire(reminders):
    """The old code completed on content AND due_at, so two reminders agreeing
    on both were closed by one firing and only one was ever spoken. Completion
    is by id now."""
    due = _iso(seconds=-5)
    actions.set_reminder("same thing", due)
    actions.set_reminder("same thing", due)

    assert actions.poll_reminders() == 2
    assert len(alerts.take_for_speech()) == 2


# ── lateness ──

def test_a_long_overdue_reminder_says_it_is_late(reminders):
    """Delivering a four hour old reminder as though it had just come due is a
    small lie that makes the clock look broken."""
    actions.set_reminder("the package", _iso(seconds=-actions.REMINDER_LATE_S - 60))
    actions.poll_reminders()
    fired, = alerts.take_for_speech()
    assert "while you were away" in fired.text


def test_a_barely_late_reminder_does_not(reminders):
    actions.set_reminder("the package", _iso(seconds=-5))
    actions.poll_reminders()
    fired, = alerts.take_for_speech()
    assert "while you were away" not in fired.text


# ── creation ──

def test_a_past_due_reminder_is_stored_and_announced_not_dropped(reminders):
    """It is a bug when this happens, almost always the model ignoring the
    clock guidance. alerts.py argues that silent non delivery is the worst
    available outcome, so it announces, late, and says so."""
    message = actions.set_reminder("already gone", _iso(hours=-3))
    assert "already passed" in message
    assert actions.poll_reminders() == 1


def test_an_unparseable_due_time_saves_nothing(reminders):
    """Refused where it can still be corrected, rather than stored and then
    skipped forever by a poller that cannot read it."""
    message = actions.set_reminder("something", "next tuesday-ish")
    assert "Could not read" in message
    assert database.active_reminder_count() == 0


# ── the poller thread ──

def test_the_poller_starts_once(monkeypatch):
    """Two pollers in one process is a bug whose only symptom is reminders
    announcing twice, so the guard is here rather than left to the caller."""
    monkeypatch.setattr(actions, "_poller_started", False)
    started = []
    monkeypatch.setattr(actions.threading, "Thread",
                        lambda **kw: type("T", (), {
                            "start": lambda self: started.append(1)})())

    assert actions.start_reminder_poller() is True
    assert actions.start_reminder_poller() is False
    assert len(started) == 1


def test_a_failing_poll_does_not_kill_the_thread(reminders, monkeypatch, capsys):
    """A poller that dies takes every future reminder with it, silently, which
    is the exact failure this change removes."""
    calls = []

    def boom(now=None):
        calls.append(1)
        raise RuntimeError("database is on fire")

    monkeypatch.setattr(actions, "poll_reminders", boom)
    monkeypatch.setattr(actions, "_poller_started", False)

    captured = {}

    def fake_thread(target=None, daemon=None):
        captured["loop"] = target
        return type("T", (), {"start": lambda self: None})()

    monkeypatch.setattr(actions.threading, "Thread", fake_thread)
    monkeypatch.setattr(actions.time, "sleep",
                        lambda s: (_ for _ in ()).throw(KeyboardInterrupt))

    actions.start_reminder_poller()
    with pytest.raises(KeyboardInterrupt):
        captured["loop"]()

    assert calls, "the poll was never attempted"
    assert "Reminder poll failed" in capsys.readouterr().out


# ── timers are rows too (Sep 16 2026) ──
# A timer was a thread sleeping in whichever process set it. Set from the app,
# it slept in miles-server, whose alert queue nothing drains, so it never went
# off; and a restart dropped every running timer.

def test_a_timer_is_a_row_that_outlives_the_process(reminders):
    actions.set_timer("10 minutes")
    (_, content, due_at, _, kind), = database.open_reminders()
    assert (content, kind) == ("10 minute timer", "timer")
    remaining = (datetime.fromisoformat(due_at) - datetime.now()).total_seconds()
    assert 590 < remaining <= 600


def test_a_timer_does_not_ring_early(reminders):
    actions.set_timer("10 minutes")
    assert actions.poll_reminders() == 0


def test_a_due_timer_rings_in_its_own_words(reminders):
    actions.set_timer("10 minutes")
    assert actions.poll_reminders(now=datetime.now() + timedelta(minutes=11)) == 1
    fired, = alerts.take_for_speech()
    assert fired.text == "[calmly] Lethanial, your ten minute timer is up."
    assert database.open_reminders() == []


def test_a_timer_set_long_ago_says_it_went_off_while_he_was_away(reminders):
    actions.set_timer("1 minutes")
    actions.poll_reminders(now=datetime.now() + timedelta(hours=3))
    fired, = alerts.take_for_speech()
    assert fired.text == "[calmly] Lethanial, your one minute timer went off while you were away."


def test_a_timer_can_be_cancelled_by_voice(reminders):
    """It could not be cancelled at all while it was a thread."""
    actions.set_timer("5 minutes")
    assert actions.cancel_reminder("timer").startswith("Removed 1")
    assert database.open_reminders() == []


def test_the_poller_is_quick_enough_for_a_timer():
    assert actions.REMINDER_POLL_S <= 5
