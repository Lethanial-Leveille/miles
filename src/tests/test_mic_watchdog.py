"""A mic read that never returns ends the process instead of leaving her deaf."""

import threading
import time

import mic_watchdog


class _Stream:
    def __init__(self, block=None):
        self.block = block
        self.closed = False

    def read(self, n, exception_on_overflow=False):
        if self.block is not None:
            self.block.wait(2.0)
        return b"\x00" * n * 2

    def get_read_available(self):
        return 42


def test_a_read_that_returns_is_not_a_stall():
    stalls = []
    guarded = mic_watchdog.GuardedStream(_Stream(), 5.0, stalls.append)
    assert guarded.read(1280) == b"\x00" * 2560
    assert guarded.check(now=time.monotonic() + 60) is False
    assert stalls == []


def test_between_reads_is_not_a_stall():
    """Nothing reads the mic while she is talking, sometimes for a minute."""
    stalls = []
    guarded = mic_watchdog.GuardedStream(_Stream(), 5.0, stalls.append)
    assert guarded.stalled_for(now=time.monotonic() + 600) == 0.0
    assert guarded.check(now=time.monotonic() + 600) is False


def test_a_blocked_read_is_caught_after_the_limit():
    release = threading.Event()
    stalls = []
    guarded = mic_watchdog.GuardedStream(_Stream(block=release), 0.1, stalls.append)
    reader = threading.Thread(target=guarded.read, args=(1280,))
    reader.start()
    time.sleep(0.05)
    assert guarded.check() is False, "not yet"
    time.sleep(0.1)
    assert guarded.check() is True
    assert stalls and stalls[0] >= 0.1
    release.set()
    reader.join()


def test_the_watchdog_thread_calls_it_once():
    release = threading.Event()
    stalls = []
    guarded = mic_watchdog.GuardedStream(_Stream(block=release), 0.05, stalls.append)
    guarded.start_watchdog(interval_s=0.02)
    reader = threading.Thread(target=guarded.read, args=(1280,))
    reader.start()
    time.sleep(0.3)
    release.set()
    reader.join()
    assert len(stalls) == 1


def test_everything_else_passes_through():
    guarded = mic_watchdog.GuardedStream(_Stream(), 5.0, lambda s: None)
    assert guarded.get_read_available() == 42


def test_the_voice_loop_reads_through_the_guard():
    import ast, pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "audio.py").read_text()
    assert "stream = mic_watchdog.GuardedStream(stream, MIC_STALL_S)" in source
    assert "stream.start_watchdog()" in source
