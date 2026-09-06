"""The health check's own judgment, which nothing else would catch.

A monitor that is wrong fails in the worst direction available: it either stays
quiet through a real outage, or it cries wolf until its alerts are ignored, and
the second is what the mic gain check had already been doing at every service
start for weeks. Both look like a working monitor from the outside.

So the two checks that make a judgment rather than wrapping a subprocess are
pinned here. The rest (systemctl, amixer, urlopen) are thin wrappers whose
behaviour belongs to the tools they call.
"""

import importlib.util
import itertools
import os
import sqlite3

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "healthcheck.py")

_spec = importlib.util.spec_from_file_location("healthcheck", _PATH)
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


_db_counter = itertools.count()


class _Config:
    """Stands in for src/config.py, so these tests never read the real DB."""
    def __init__(self, db_path):
        self.DB_PATH = db_path


def _timing_db(tmp_path, cache_values):
    """A timing_log holding just the column the cache check reads.

    Written newest last, since the check orders by id descending."""
    # A fresh file per call, so a test that builds several databases from one
    # tmp_path does not reopen the previous one.
    path = str(tmp_path / f"t{next(_db_counter)}.db")
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE timing_log (id INTEGER PRIMARY KEY, cache_read_tokens INT)")
    connection.executemany("INSERT INTO timing_log (cache_read_tokens) VALUES (?)",
                           [(v,) for v in cache_values])
    connection.commit()
    connection.close()
    return _Config(path)


# ── the prompt cache check ──

def test_cache_all_zero_warns(tmp_path):
    """Every recent turn reading zero is the documented signature of the
    cacheable prefix falling under the model minimum, which fails silently."""
    config = _timing_db(tmp_path, [0] * hc.CACHE_ZERO_RUN)
    result, = hc.check_cache(config)
    assert result.level == hc.WARN
    assert "cacheable prefix" in result.detail


def test_cache_scattered_zeroes_pass(tmp_path):
    """The false positive this check is most likely to produce.

    docs/SESSION_START.md is explicit that scattered zeroes are the five minute
    TTL expiring on turns spaced further apart than the window, and that a
    twenty to thirty percent miss rate is the expected steady state. A monitor
    that alarms on normal use gets muted, and then it is not a monitor.

    This exact pattern was read off the live database, seven consecutive zeroes
    inside a healthy window, which is why the run length is set high rather
    than at something that merely looks decisive."""
    live = [11472, 0, 11472, 0, 0, 0, 0, 0, 0, 0, 0, 11472]
    config = _timing_db(tmp_path, list(reversed(live)))
    result, = hc.check_cache(config)
    assert result.level == hc.OK


def test_cache_stays_quiet_without_enough_turns(tmp_path):
    """Two zeroes out of two turns is not evidence of anything.

    Declining to judge is the correct outcome for a thin sample. Reporting a
    failure here would make every fresh database look broken."""
    config = _timing_db(tmp_path, [0, 0])
    result, = hc.check_cache(config)
    assert result.level == hc.OK
    assert "not enough" in result.detail


def test_cache_never_fails_only_warns(tmp_path):
    """Severity is the whole point of this check being allowed to ship.

    Its false positive rate is not yet measured, so it is not permitted to send
    an alert. Promote it to FAIL only after watching it stay quiet through
    normal sparse use."""
    for values in ([0] * hc.CACHE_ZERO_RUN, [0, 0], [1] * hc.CACHE_ZERO_RUN):
        config = _timing_db(tmp_path, values)
        result, = hc.check_cache(config)
        assert result.level in (hc.OK, hc.WARN)


def test_cache_survives_a_missing_table(tmp_path):
    """A checker that raises reports nothing at all, which is worse than a
    checker that reports it could not look."""
    path = str(tmp_path / "empty.db")
    sqlite3.connect(path).close()
    result, = hc.check_cache(_Config(path))
    assert result.level == hc.WARN


# ── the restart delta ──

def test_first_run_records_a_baseline_without_alarming(monkeypatch):
    """NRestarts is cumulative, so its absolute value says nothing about now.

    A machine up for a month with three old restarts is healthy. Alarming on
    the raw number would fire on every fresh install of this check."""
    monkeypatch.setattr(hc, "_systemctl",
                        lambda *a: "active" if a[0] == "is-active" else "7")
    state = {}
    results = hc.check_units(state)
    assert not [r for r in results if r.level == hc.FAIL]
    assert state["restarts"] == {u: 7 for u in hc.UNITS}


def test_a_restart_since_the_last_check_fails(monkeypatch):
    """The failure is-active cannot see.

    A service crash looping under Restart=always reads as active at whatever
    instant it is sampled. CLAUDE.md describes exactly this outage: the room
    hears a chime, then silence, forever, with nothing reporting a problem."""
    monkeypatch.setattr(hc, "_systemctl",
                        lambda *a: "active" if a[0] == "is-active" else "9")
    state = {"restarts": {u: 7 for u in hc.UNITS}}
    failures = [r for r in hc.check_units(state) if r.level == hc.FAIL]
    assert len(failures) == len(hc.UNITS)
    assert "restarted 2 time(s)" in failures[0].detail


def test_an_inactive_unit_fails(monkeypatch):
    monkeypatch.setattr(hc, "_systemctl",
                        lambda *a: "failed" if a[0] == "is-active" else "0")
    failures = [r for r in hc.check_units({}) if r.level == hc.FAIL]
    assert len(failures) == len(hc.UNITS)
    assert "expected active" in failures[0].detail


def test_unreadable_restart_count_is_not_a_failure(monkeypatch):
    """systemctl show returning nothing means the question could not be asked,
    not that the answer was bad."""
    monkeypatch.setattr(hc, "_systemctl",
                        lambda *a: "active" if a[0] == "is-active" else "")
    assert not [r for r in hc.check_units({}) if r.level == hc.FAIL]


# ── notification ──

def test_notify_without_smtp_configured_does_not_raise(monkeypatch, capsys):
    """The journal is the floor and must work with nothing set up."""
    for var in ("MILES_SMTP_USER", "MILES_SMTP_PASSWORD", "MILES_ALERT_EMAIL"):
        monkeypatch.delenv(var, raising=False)
    hc.notify("subject", "body")
    assert "subject" in capsys.readouterr().out


def test_a_failed_send_never_masks_the_failure_it_carries(monkeypatch, capsys):
    """An alert that cannot be delivered still has to reach the journal.

    Swallowing the original failure because the mail server was down would lose
    the one thing worth keeping."""
    monkeypatch.setenv("MILES_SMTP_USER", "x@example.com")
    monkeypatch.setenv("MILES_SMTP_PASSWORD", "nope")
    monkeypatch.setenv("MILES_ALERT_EMAIL", "x@example.com")
    monkeypatch.setenv("MILES_SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("MILES_SMTP_PORT", "1")
    hc.notify("the real failure", "details")
    out = capsys.readouterr().out
    assert "the real failure" in out
    assert "Could not send alert email" in out
