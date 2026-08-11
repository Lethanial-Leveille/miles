"""Deferred alert delivery.

The bug these exist for: on Aug 11 2026 a one minute timer fired 2 seconds into
a follow up window and spoke straight into the question being asked. speak_lock
kept Nova from talking over herself, but capture holds no lock, so a background
thread could not tell an open mic from an idle room.
"""

import time

import pytest

import alerts
import brain


@pytest.fixture(autouse=True)
def isolated(db, monkeypatch):
    """Throwaway database for the delivery log, empty queue either side."""
    monkeypatch.setattr(alerts, "log_alert", lambda **kw: db.log_alert(**kw))
    alerts.clear()
    yield
    alerts.clear()


def _fire(kind="timer", summary="the 1 minute timer just finished"):
    alerts.fire(kind=kind, text=f"Lethanial, {summary}.", summary=summary)


# ── queueing ──

def test_firing_speaks_nothing_and_only_queues():
    """The whole point. Producers never touch TTS."""
    _fire()
    assert alerts.pending_count() == 1


def test_fires_from_a_background_thread():
    import threading
    t = threading.Thread(target=_fire)
    t.start()
    t.join()
    assert alerts.pending_count() == 1


# ── folding ──

def test_fresh_alerts_fold_into_a_turn():
    _fire()
    folded = alerts.take_for_fold()
    assert len(folded) == 1
    assert alerts.pending_count() == 0


def test_stale_alerts_do_not_fold():
    """An alert that fired into silence must not sit waiting for a conversation
    that may never come. After the cap it falls through to standalone."""
    _fire()
    folded = alerts.take_for_fold(max_age=-1)
    assert folded == []
    assert alerts.pending_count() == 1, "must stay queued for standalone delivery"


def test_folding_takes_only_the_fresh_ones():
    old = alerts.Alert("timer", "t", "old one", time.monotonic() - 999, "x")
    with alerts._lock:
        alerts._pending.append(old)
    _fire()
    folded = alerts.take_for_fold()
    assert [a.summary for a in folded] == ["the 1 minute timer just finished"]
    assert alerts.pending_count() == 1


# ── standalone ──

def test_speech_drains_everything_regardless_of_age():
    _fire()
    _fire(kind="reminder", summary="a reminder came due")
    assert len(alerts.take_for_speech()) == 2
    assert alerts.pending_count() == 0


def test_an_alert_is_delivered_exactly_once():
    """Folded and standalone both remove from the queue, so no alert can be
    both worked into a response and then announced again."""
    _fire()
    alerts.take_for_fold()
    assert alerts.take_for_speech() == []


# ── the delivery log ──

def test_delivery_is_logged_with_mode_and_delay(db):
    _fire()
    alerts.take_for_fold()
    row = db_rows(db)[0]
    assert row["mode"] == "folded"
    assert row["kind"] == "timer"
    assert row["delay_ms"] >= 0


def test_standalone_delivery_logs_its_own_mode(db):
    _fire()
    alerts.take_for_speech()
    assert db_rows(db)[0]["mode"] == "spoken"


def test_an_undelivered_alert_is_not_logged(db):
    """Written at delivery, not at firing, so a queued alert never appears as
    though it landed."""
    _fire()
    assert db_rows(db) == []


def db_rows(db):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM alert_log ORDER BY id")]
    conn.close()
    return rows


# ── folding into the message array ──

def test_alerts_attach_to_the_final_user_turn():
    messages = [{"role": "user", "content": "how hot is the pi"}]
    out = brain._with_alerts(messages, [
        alerts.Alert("timer", "t", "the 1 minute timer just finished", 0, "x")
    ])
    assert out[-1]["content"].startswith("how hot is the pi")
    assert "[Alert: the 1 minute timer just finished]" in out[-1]["content"]


def test_alerts_land_after_the_cache_breakpoint_not_in_the_system_prompt():
    """Same placement as the clock and for the same reason. In the system
    prompt this would invalidate the cached prefix on every alert."""
    messages = [{"role": "user", "content": "hi"}]
    out = brain._with_alerts(messages, [alerts.Alert("timer", "t", "s", 0, "x")])
    assert len(out) == len(messages)
    assert out[-1]["role"] == "user"


def test_no_alerts_leaves_the_messages_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert brain._with_alerts(messages, []) is messages


def test_alerts_are_not_attached_to_an_assistant_turn():
    messages = [{"role": "assistant", "content": "done"}]
    assert brain._with_alerts(messages, [alerts.Alert("t", "t", "s", 0, "x")]) is messages


def test_several_alerts_all_attach():
    messages = [{"role": "user", "content": "hi"}]
    out = brain._with_alerts(messages, [
        alerts.Alert("timer", "t", "timer done", 0, "x"),
        alerts.Alert("reminder", "r", "reminder due", 0, "x"),
    ])
    assert "timer done" in out[-1]["content"]
    assert "reminder due" in out[-1]["content"]
