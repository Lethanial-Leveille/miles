"""Whether he has started talking over her: the barge in speech decision alone.

Kept out of audio.py so it can be tested while miles-voice holds the microphone,
and so scripts can replay recordings through exactly the code that ships.

A frame is his when webrtcvad calls it speech and it is louder than the floor;
she stops once enough of those come in a row. Both halves matter. Level alone
stops her on a door closing; VAD alone stopped her on her own echo, which on
Sep 30 2026 was speech on half or more of channel 0's frames. With her on the
array's jack the canceller works, and on session 20261001T114604 her echo on
channel 1 never held a qualifying run past 180ms in 231s, while he cleared 300ms
29 of 29 times. Nothing here says whose speech it is: a guest or a TV stops her
too. That was chosen, Oct 1 2026; who it was is decided afterward, on the whole
interruption.

The watcher reads 80ms chunks for the wake model and webrtcvad takes 30ms
frames, which do not divide, so the remainder of each read is carried into the
next. Restarting the frame grid per read would change every frame boundary from
what was measured.
"""

import threading

import numpy as np

FRAME = 480        # 30ms at 16kHz, VAD_FRAME everywhere else in the pipeline


def frame_dbfs(frame):
    """RMS level of int16 samples, in dBFS. RMS, not peak: on the replay, peak at
    the same floor let one of her runs reach 540ms."""
    x = frame.astype(np.float64) / 32768.0
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


class SpeechOnset:
    """Fed the watcher's audio in any size of read; fires once.

    is_speech takes 30ms of int16 PCM bytes, as webrtcvad.Vad.is_speech does
    once given the rate. onset is the sample, counted from the first byte fed,
    where the run that fired began, so the audio from there on can be kept."""

    def __init__(self, floor_dbfs, need_ms, is_speech, frame_ms=30):
        self.floor = floor_dbfs
        self.need = -(-need_ms // frame_ms)     # whole frames, rounded up
        self.is_speech = is_speech
        self.carry = b""
        self.frames = 0
        self.run = 0
        self.fired = False
        self.onset = None

    def feed(self, pcm):
        """Returns True once the run is long enough, and on every read after."""
        if self.fired:
            return True
        data = self.carry + pcm
        usable = len(data) - len(data) % (FRAME * 2)
        self.carry = data[usable:]
        for i in range(0, usable, FRAME * 2):
            chunk = data[i:i + FRAME * 2]
            # VAD on every frame, never only the loud ones. webrtcvad adapts
            # its noise estimate to what it is shown; fed only frames over the
            # floor, it learned her loudest moments as the background and called
            # him not speech, missing 2 of 29 on the Oct 1 2026 replay.
            speech = self.is_speech(chunk)
            loud = frame_dbfs(np.frombuffer(chunk, dtype=np.int16)) > self.floor
            self.run = self.run + 1 if loud and speech else 0
            self.frames += 1
            if self.run >= self.need:
                self.fired = True
                self.onset = (self.frames - self.run) * FRAME
                return True
        return False


def seed_from(kept, onset, preroll_samples):
    """What the next recording starts with: his audio from a preroll before his
    onset. The same preroll every recording keeps, so a soft first consonant
    is not cut off where the run of loud frames began."""
    return kept[max(0, onset - preroll_samples) * 2:]


class Carrier:
    """Keeps recording him after a speech stop, until the voice loop claims it.

    Between her stopping and the follow up listener opening, the turn still has
    to finish, and on Oct 1 2026 what he said in that gap was flushed as if it
    were her echo: "Are you?" reached Claude as all that was left of his
    interruption. This reads on, so nothing is dropped and the buffer never
    overflows. Its own thread rather than the watcher's, because the player
    waits on the watcher, and every interruption would wait with it.

    One reader at a time on the mic stream: claim stops the reading before
    returning, and whoever claims is the next reader. Stops by itself at
    max_seconds in case nothing claims it."""

    def __init__(self, seed, read, max_seconds, rate=16000):
        self._chunks = [seed]
        self._size = len(seed)
        self._read = read
        self._max_bytes = int(max_seconds * rate) * 2
        self._release = threading.Event()
        self._done = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while not self._release.is_set() and self._size < self._max_bytes:
                data = self._read()
                self._chunks.append(data)
                self._size += len(data)
        except Exception as e:
            print(f"  barge in: carrying his speech failed: {e!r}", flush=True)
        finally:
            self._done.set()

    def claim(self, timeout=1.0):
        """Everything kept, once the reading has stopped. A read is 80ms, so the
        timeout is only reached if the mic itself has stalled, which the mic
        watchdog ends the process for."""
        self._release.set()
        if not self._done.wait(timeout):
            print("  barge in: the carrier did not stop; the mic may be stalled", flush=True)
        return b"".join(self._chunks)
