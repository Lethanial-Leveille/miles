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


# Below this a frame is silence, for comparing envelopes. Her own digital
# silence sits near minus 240 dBFS and the room near minus 50; unclamped, the
# moment she starts would be the only thing a correlation ever saw.
ENVELOPE_FLOOR = -60.0
# Her stretches quieter than this throughout say nothing about whether the mic
# is hearing her, so they never count as a match.
HER_SILENT_DBFS = -50.0


def envelope(samples, rate, frame_s=0.03):
    """Level per frame_s of int16 samples, in dBFS, clamped at the floor. Works
    at any rate, so her 22050Hz audio and the 16kHz mic line up frame for frame."""
    n = int(round(rate * frame_s))
    count = len(samples) // n
    if count == 0:
        return np.zeros(0)
    x = samples[:count * n].astype(np.float64).reshape(count, n) / 32768.0
    db = 20 * np.log10(np.sqrt(np.mean(x ** 2, axis=1)) + 1e-12)
    return np.maximum(db, ENVELOPE_FLOOR)


def echo_score(mic_env, her_env, min_lag, max_lag):
    """How closely the mic's recent envelope follows hers, at the best delay.

    mic_env is the last W frames heard. her_env is what she played over the
    same stretch reaching max_lag frames earlier, so it is W + max_lag long and
    ends at the same moment. Her echo is a delayed, quieter copy of her, so it
    rises and falls with her at some delay; his voice over her follows his own
    syllables. Only delays from min_lag to max_lag frames are tried: every
    extra delay is another chance for two unrelated voices to line up by luck,
    and over a 1.5s search his real speech over her scored as high as 0.91.

    Returns the best Pearson correlation, or -1.0 when no delay can be judged."""
    w = len(mic_env)
    best = -1.0
    if w < 2 or np.std(mic_env) == 0:
        return best
    for lag in range(min_lag, max_lag + 1):
        start = len(her_env) - w - lag
        if start < 0:
            break
        hers = her_env[start:start + w]
        if hers.max() < HER_SILENT_DBFS or np.std(hers) == 0:
            continue
        best = max(best, float(np.corrcoef(mic_env, hers)[0, 1]))
    return best


# aplay's ALSA buffer, as tts._open_aplay opens it: 8192 frames of 16 bit mono.
# aplay starts the device only once that is full, or once its input closes on a
# sentence shorter than that.
APLAY_START_BYTES = 8192 * 2


class HerAudio:
    """What she sent to aplay, placed on the clock the room heard it on.

    Written by play() on the main thread, read by the watcher; hence the lock.
    Writes arrive at whatever pace ElevenLabs streams, so the write times say
    little about when she was heard. What does: aplay starts the device when
    its buffer fills or its input closes, and from then plays in real time. An
    underrun breaks that, and the comparison then quietly fails to match,
    which leaves her stoppable as before rather than deaf to him."""

    def __init__(self, rate=22050, clock=None):
        import time
        self.rate = rate
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._pcm = bytearray()
        self.playing_from = None

    def wrote(self, chunk):
        with self._lock:
            self._pcm += chunk
            if self.playing_from is None and len(self._pcm) >= APLAY_START_BYTES:
                self.playing_from = self._clock()

    def closed(self):
        with self._lock:
            if self.playing_from is None and self._pcm:
                self.playing_from = self._clock()

    def envelope_until(self, t_end, frames, frame_s=0.03):
        """Her envelope over the frames frames ending at t_end, on the
        monotonic clock; floor wherever she was not playing."""
        out = np.full(frames, ENVELOPE_FLOOR)
        with self._lock:
            if self.playing_from is None:
                return out
            pcm = np.frombuffer(bytes(self._pcm), dtype=np.int16)
            start = self.playing_from
        # Her frame index at the first output frame; negative before she began.
        # Nearest, not floor: her grid and the mic's never line up exactly, and
        # floor on a float a hair under a whole frame shifted her by one.
        first = int(round((t_end - frames * frame_s - start) / frame_s))
        hers = envelope(pcm, self.rate, frame_s)
        for i in range(frames):
            j = first + i
            if 0 <= j < len(hers):
                out[i] = hers[j]
        return out


class SpeechOnset:
    """Fed the watcher's audio in any size of read; fires once.

    is_speech takes 30ms of int16 PCM bytes, as webrtcvad.Vad.is_speech does
    once given the rate. onset is the sample, counted from the first byte fed,
    where the run that fired began, so the audio from there on can be kept.

    is_echo, when given, is asked each frame a run is long enough: it takes the
    clamped levels heard so far and returns True when they are her own echo,
    and she is not stopped for that frame. Asked again on every frame after,
    so him talking over her echo still fires once the match breaks. Oct 4 and
    5 2026 she stopped on her own first syllables, leaking through the
    canceller after hours of quiet, at minus 30 dBFS: above any floor that
    still hears him."""

    def __init__(self, floor_dbfs, need_ms, is_speech, frame_ms=30,
                 is_echo=None):
        self.floor = floor_dbfs
        self.need = -(-need_ms // frame_ms)     # whole frames, rounded up
        self.is_speech = is_speech
        self.is_echo = is_echo
        self.levels = []
        self.echo_frames = 0
        self.run_had_echo = False
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
            level = frame_dbfs(np.frombuffer(chunk, dtype=np.int16))
            self.levels.append(max(level, ENVELOPE_FLOOR))
            self.run = self.run + 1 if level > self.floor and speech else 0
            if self.run == 0:
                self.run_had_echo = False
            self.frames += 1
            if self.run >= self.need and self.is_echo is not None \
                    and self.is_echo(self.levels):
                self.echo_frames += 1
                self.run_had_echo = True
                continue
            if self.run >= self.need:
                self.fired = True
                # After her echo, only the frames that fired are his: kept
                # from the run's start, the next recording opened on her words.
                length = self.need if self.run_had_echo else self.run
                self.onset = (self.frames - length) * FRAME
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


# Her pace, words per second of audio: the median over 73 replies in
# data/spoken on Oct 2 2026 (the middle 80 percent ran 2.7 to 3.9). Used only
# when a cut sentence's audio had not all arrived, so its length is unknown.
HER_WORDS_PER_S = 3.2


def _split_heard(text, heard_s, audio_s=None):
    """text as words, and how many of them played in heard_s seconds.

    One place for the cut point, so what she is recorded as having said and
    what she is recorded as having left can never overlap or leave a gap."""
    words = text.split()
    if audio_s:
        count = round(len(words) * heard_s / audio_s)
    else:
        count = round(heard_s * HER_WORDS_PER_S)
    return words, max(0, min(len(words), count))


def words_heard(text, heard_s, audio_s=None):
    """The words of text that played in heard_s seconds, cut at a word.

    By the share of its audio that played when the whole length is known, which
    follows her pace in that sentence; by her usual pace when it is not. A few
    words either way: exact timings would mean ElevenLabs' timestamped
    endpoint, a bigger change for little gain."""
    words, count = _split_heard(text, heard_s, audio_s)
    return " ".join(words[:count])


def words_unheard(text, heard_s, audio_s=None):
    """The words of text that had not played yet: the other side of the same cut
    words_heard makes, so the two always join back into text."""
    words, count = _split_heard(text, heard_s, audio_s)
    return " ".join(words[count:])
