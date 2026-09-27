"""Whether anyone is in the room, from the LD2410C radar, logged as it changes.

Logging only for now. Nothing Nova does reads this yet: the log comes first so
that how slow leaving should be is chosen from measured gaps, not guessed.
BACKEND_TODO.md, Presence detection, has the wiring, the tuning and the plan.

Three states, and silent is not absent. The sensor stops sending while the
HLKRadarTool app holds it, and a loose wire looks the same. A room nobody can
see into is logged as unknown, never as empty.

The port is read on its own thread because a read waits for bytes, and the
voice loop cannot wait anywhere but the microphone. The read has a timeout so
the thread wakes even when nothing arrives, which is how it notices silence.
"""

import threading
import time

import ld2410
from database import log_presence

PRESENT, ABSENT, SILENT = "present", "absent", "silent"

# How long to wait before reopening a port that failed or would not open. Long
# enough that an unplugged sensor costs a line in the journal now and then, not
# a loop.
REOPEN_S = 10.0

# Enough for several frames per read at 256000 baud, few enough that a read
# returns well inside its timeout.
_READ_BYTES = 64
_READ_TIMEOUT_S = 0.5


class PresenceTracker:
    """Turns reports and the passage of time into state changes.

    Returns the new state when it changes and None otherwise, so the caller
    writes one row per change. Starts silent: nothing has been heard yet."""

    def __init__(self, silent_s):
        self._silent_s = silent_s
        self._last_frame = None
        self.state = SILENT

    def feed(self, report, now):
        self._last_frame = now
        return self._change(PRESENT if ld2410.is_present(report) else ABSENT)

    def tick(self, now):
        """Silent once no frame has arrived for silent_s."""
        if self._last_frame is None or now - self._last_frame < self._silent_s:
            return None
        return self._change(SILENT)

    def lost(self):
        """The port itself failed. No need to wait out silent_s to know it."""
        return self._change(SILENT)

    def _change(self, state):
        if state == self.state:
            return None
        self.state = state
        return state


_tracker = None


def current():
    """The latest state, for whatever reads presence later. Silent until the
    reader has started and heard a frame."""
    return _tracker.state if _tracker is not None else SILENT


def start_reader(port, baud, silent_s, log=log_presence):
    """Read the radar on a daemon thread for the life of the process.

    Never raises into the caller and never exits: a failed port is logged as
    silent and reopened every REOPEN_S. Presence is optional, and the voice loop
    must not depend on a sensor being plugged in."""
    global _tracker
    tracker = PresenceTracker(silent_s)
    _tracker = tracker
    # Marks the restart in the log. Whatever the last row said before this
    # was true of a process that is no longer running.
    log(SILENT)

    def run():
        last_error = None
        while True:
            try:
                _read_until_error(port, baud, tracker, log)
            except Exception as exc:
                change = tracker.lost()
                if change:
                    log(change)
                # Once per distinct error, so an unplugged sensor is one line
                # in the journal and not one every ten seconds.
                if str(exc) != last_error:
                    last_error = str(exc)
                    print(f"Presence radar unavailable: {exc}", flush=True)
            time.sleep(REOPEN_S)

    threading.Thread(target=run, name="presence", daemon=True).start()


def _read_until_error(port, baud, tracker, log):
    import serial  # only the reader needs pyserial; the tracker is testable without it

    # exclusive: two readers on one serial port each get part of the stream and
    # decode neither. A second opener now fails loudly instead.
    with serial.Serial(port, baud, timeout=_READ_TIMEOUT_S, exclusive=True) as conn:
        print(f"Presence radar reading {port}.", flush=True)
        buf = b""
        while True:
            buf = _pump(conn, buf, tracker, log)


def _pump(conn, buf, tracker, log, clock=time.monotonic):
    """One read: decode what arrived, log any change, return the leftover bytes."""
    reports, buf = ld2410.parse(buf + conn.read(_READ_BYTES))
    now = clock()
    for report in reports:
        change = tracker.feed(report, now)
        if change:
            log(change, report)
    change = tracker.tick(now)
    if change:
        log(change)
    return buf
