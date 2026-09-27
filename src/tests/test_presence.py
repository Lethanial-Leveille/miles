import sqlite3

import ld2410
import presence
from ld2410 import Report
from presence import ABSENT, PRESENT, SILENT, PresenceTracker

SOMEONE = Report(ld2410.STILL, 0, 0, 162, 100)
NOBODY = Report(ld2410.NOBODY, 444, 0, 444, 13)


def _frame(report):
    data = (bytes([0x02, 0xAA, report.state]) + report.moving_cm.to_bytes(2, "little")
            + bytes([report.moving_energy]) + report.still_cm.to_bytes(2, "little")
            + bytes([report.still_energy]) + b"\x00\x00\x55\x00")
    return ld2410.HEADER + len(data).to_bytes(2, "little") + data + ld2410.FOOTER


class FakePort:
    """Hands out one queued chunk per read, then nothing, like a sensor that
    went quiet."""

    def __init__(self, *chunks):
        self._chunks = list(chunks)

    def read(self, size):
        return self._chunks.pop(0) if self._chunks else b""


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_starts_silent_until_a_frame_arrives():
    tracker = PresenceTracker(silent_s=3.0)
    assert tracker.state == SILENT
    assert tracker.tick(100.0) is None


def test_only_changes_are_reported():
    tracker = PresenceTracker(silent_s=3.0)
    assert tracker.feed(SOMEONE, 0.0) == PRESENT
    assert tracker.feed(SOMEONE, 0.1) is None
    assert tracker.feed(NOBODY, 0.2) == ABSENT
    assert tracker.feed(NOBODY, 0.3) is None


def test_silence_is_not_absence():
    tracker = PresenceTracker(silent_s=3.0)
    tracker.feed(SOMEONE, 0.0)
    assert tracker.tick(2.9) is None
    assert tracker.tick(3.0) == SILENT
    assert tracker.tick(10.0) is None


def test_a_frame_after_silence_restores_the_state():
    tracker = PresenceTracker(silent_s=3.0)
    tracker.feed(SOMEONE, 0.0)
    tracker.tick(5.0)
    assert tracker.feed(SOMEONE, 6.0) == PRESENT


def test_a_failed_port_is_silent_at_once():
    tracker = PresenceTracker(silent_s=3.0)
    tracker.feed(NOBODY, 0.0)
    assert tracker.lost() == SILENT
    assert tracker.lost() is None


def test_pump_logs_each_change_with_its_reading():
    logged = []
    tracker = PresenceTracker(silent_s=3.0)
    port = FakePort(_frame(NOBODY) + _frame(SOMEONE) + _frame(SOMEONE))
    presence._pump(port, b"", tracker, lambda *a: logged.append(a), clock=Clock())
    assert logged == [(ABSENT, NOBODY), (PRESENT, SOMEONE)]


def test_pump_carries_a_split_frame_into_the_next_read():
    logged = []
    frame = _frame(SOMEONE)
    tracker = PresenceTracker(silent_s=3.0)
    port = FakePort(frame[:10], frame[10:])
    clock = Clock()
    buf = presence._pump(port, b"", tracker, lambda *a: logged.append(a), clock=clock)
    assert logged == []
    presence._pump(port, buf, tracker, lambda *a: logged.append(a), clock=clock)
    assert logged == [(PRESENT, SOMEONE)]


def test_pump_notices_a_sensor_that_stops_talking():
    logged = []
    tracker = PresenceTracker(silent_s=3.0)
    port = FakePort(_frame(SOMEONE))
    clock = Clock()
    presence._pump(port, b"", tracker, lambda *a: logged.append(a), clock=clock)
    clock.now = 3.5
    presence._pump(port, b"", tracker, lambda *a: logged.append(a), clock=clock)
    assert logged == [(PRESENT, SOMEONE), (SILENT,)]


def test_current_is_silent_before_the_reader_starts(monkeypatch):
    monkeypatch.setattr(presence, "_tracker", None)
    assert presence.current() == SILENT


def test_log_presence_writes_the_reading(db):
    db.log_presence(PRESENT, SOMEONE)
    db.log_presence(SILENT)
    rows = sqlite3.connect(db.DB_PATH).execute(
        "SELECT state, moving_cm, moving_energy, still_cm, still_energy"
        " FROM presence_log ORDER BY id").fetchall()
    assert rows == [(PRESENT, 0, 0, 162, 100), (SILENT, None, None, None, None)]


def test_log_presence_never_raises(monkeypatch, tmp_path):
    import database
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "missing" / "no.db"))
    database.log_presence(PRESENT, SOMEONE)
