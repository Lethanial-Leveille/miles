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


# ── Per gate energy ──

def _engineering(state, moving_gates, still_gates):
    data = (bytes([0x01, 0xAA, state]) + bytes(6) + b"\x00\x00"
            + bytes([len(moving_gates) - 1, len(still_gates) - 1])
            + bytes(moving_gates) + bytes(still_gates) + b"\x00\x00\x55\x00")
    return ld2410.HEADER + len(data).to_bytes(2, "little") + data + ld2410.FOOTER


def _eng_report(state, moving_gates, still_gates):
    reports, _ = ld2410.parse(_engineering(state, moving_gates, still_gates))
    return reports[0]


def test_window_summarises_each_gate_once_it_has_run():
    window = presence.EnergyWindow(60)
    window.add(_eng_report(ld2410.STILL, [1, 5], [10, 2]), 0.0)
    window.add(_eng_report(ld2410.NOBODY, [3, 7], [4, 2]), 30.0)
    window.add(_eng_report(ld2410.STILL, [2, 6], [16, 2]), 45.0)
    assert window.take(59.0) is None
    assert window.take(60.0) == {
        "frames": 3, "present_frames": 2,
        "moving_med": [2, 6], "moving_max": [3, 7],
        "still_min": [4, 2], "still_med": [10, 2], "still_max": [16, 2],
    }
    # The next window starts with the next report, not at the old boundary.
    assert window.take(200.0) is None


def test_window_ignores_reports_without_gate_energy():
    window = presence.EnergyWindow(60)
    window.add(SOMEONE, 0.0)
    assert window.take(100.0) is None


def test_pump_writes_an_energy_row_per_window():
    energy = []
    window = presence.EnergyWindow(60)
    tracker = PresenceTracker(silent_s=3.0)
    clock = Clock()
    port = FakePort(_engineering(ld2410.STILL, [1, 2], [30, 4]),
                    _engineering(ld2410.STILL, [1, 2], [20, 4]))
    presence._pump(port, b"", tracker, lambda *a: None, window, energy.append, clock=clock)
    clock.now = 61.0
    presence._pump(port, b"", tracker, lambda *a: None, window, energy.append, clock=clock)
    assert [e["still_med"] for e in energy] == [[25, 4]]


class FakeRadar:
    """Acknowledges every command it is sent, the way the module does."""

    def __init__(self, refuse=()):
        self.sent = []
        self._out = b""
        self._refuse = refuse

    def write(self, frame):
        word = int.from_bytes(frame[6:8], "little")
        self.sent.append(word)
        status = b"\x01\x00" if word in self._refuse else b"\x00\x00"
        body = (word | 0x0100).to_bytes(2, "little") + status
        self._out += (ld2410.CMD_HEADER + len(body).to_bytes(2, "little")
                      + body + ld2410.CMD_FOOTER)

    def read(self, size):
        out, self._out = self._out, b""
        return out


def test_config_mode_always_leaves():
    radar = FakeRadar()
    try:
        with ld2410.config_mode(radar):
            ld2410.send(radar, ld2410.command(ld2410.ENGINEERING_ON), ld2410.ENGINEERING_ON)
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert radar.sent == [ld2410.ENABLE_CONFIG, ld2410.ENGINEERING_ON, ld2410.END_CONFIG]


def test_engineering_mode_failure_does_not_stop_the_reader(capsys):
    radar = FakeRadar(refuse=(ld2410.ENABLE_CONFIG,))
    presence._engineering_on(radar)
    assert "engineering mode unavailable" in capsys.readouterr().out


def test_log_presence_energy_writes_lists(db):
    db.log_presence_energy({"frames": 600, "present_frames": 590,
                            "moving_med": [1], "moving_max": [2], "still_min": [8],
                            "still_med": [12], "still_max": [34]})
    row = sqlite3.connect(db.DB_PATH).execute(
        "SELECT frames, present_frames, still_min, still_med FROM presence_energy").fetchone()
    assert row == (600, 590, "[8]", "[12]")
