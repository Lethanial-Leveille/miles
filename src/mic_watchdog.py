"""Ends the process when the microphone stops delivering audio, so systemd restarts it.

On Sep 26 2026 at 00:06:49 PortAudio restarted the capture stream after the usual
overrun during a reply, and the Seiren delivered 16 frames and then nothing. ALSA
still reported RUNNING, so no error was raised: the voice loop sat inside
stream.read for twenty minutes, deaf, with every service reporting active. See
INCIDENTS.md.

A read of one 80ms chunk returns in about 80ms. One still blocked after
MIC_STALL_S is not coming back, and a fresh process reopens the device cleanly,
which is what fixed it by hand. Only a read in progress is watched: nothing reads
the mic while Nova is talking, and that is not a stall.

Kept out of audio.py so it can be tested while miles-voice holds the microphone.
"""

import glob
import os
import threading
import time


class GuardedStream:
    """The PyAudio input stream, with every read timed. Everything else passes
    straight through, so no caller changes."""

    def __init__(self, stream, stall_s, on_stall=None):
        self._stream = stream
        self._stall_s = stall_s
        self._on_stall = on_stall or _exit_for_restart
        self._reading_since = None

    def read(self, *args, **kwargs):
        self._reading_since = time.monotonic()
        try:
            return self._stream.read(*args, **kwargs)
        finally:
            self._reading_since = None

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def stalled_for(self, now=None):
        """Seconds the current read has been blocked, or 0 when none is."""
        since = self._reading_since
        if since is None:
            return 0.0
        return (now if now is not None else time.monotonic()) - since

    def check(self, now=None):
        """Call on_stall once if a read has been blocked too long. True if it did."""
        blocked = self.stalled_for(now)
        if blocked < self._stall_s:
            return False
        self._on_stall(blocked)
        return True

    def start_watchdog(self, interval_s=1.0):
        def watch():
            while True:
                time.sleep(interval_s)
                if self.check():
                    return
        threading.Thread(target=watch, name="mic_watchdog", daemon=True).start()


def capture_status():
    """The ALSA state of the capture device this process holds, for the journal.
    A frozen hw_ptr under state RUNNING is what the Sep 26 stall looked like."""
    for path in glob.glob("/proc/asound/card*/pcm*c/sub*/status"):
        try:
            text = open(path).read()
        except OSError:
            continue
        if f"owner_pid   : {os.getpid()}" in text:
            return path + "\n" + text.strip()
    return "capture status not found"


def _exit_for_restart(blocked_s):
    print(f"MICROPHONE STALLED: a read has been blocked {blocked_s:.1f}s. "
          f"Exiting so systemd restarts the voice loop.\n{capture_status()}", flush=True)
    # _exit rather than sys.exit: the main thread is stuck inside the read and
    # would never see an exception raised here. A SQLite write cut short rolls
    # back under WAL; it does not corrupt.
    os._exit(1)
