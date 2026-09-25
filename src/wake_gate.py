"""Reject a wake that arrived too quiet to have been him.

The wake score cannot separate a real wake from the neighbours. Measured on 49
labelled hits, real and false wakes overlap from 0.40 to 0.94, so raising
WAKE_THRESHOLD to 0.8 still admits a false wake while discarding half the real
ones. There is no threshold to find.

Level does separate them, because speech arriving through a wall is quiet and
openWakeWord never looks at loudness at all. Both distributions are in
docs/AUDIO_PIPELINE.md.

Kept in its own module, away from audio.py, for one reason: audio.py takes the
microphone lock at import, so anything living there can only be tested with
miles-voice stopped. This is the guard that can make her deaf, and it is not
allowed to be the one that goes untested on an ordinary run.
"""

import numpy as np

from config import WAKE_LEVEL_GATE, WAKE_LEVEL_FLOOR


def window_peak_dbfs(frames):
    """Peak level of the raw wake window, in dBFS, or None if there is nothing.

    Peak rather than RMS because peak is what separates the two populations: on
    38 real and 22 false wakes, no real wake is lost above -42 dBFS, while RMS
    at its best line loses 12 of 38. A wake phrase carries a loud consonant even
    when its average is low, and averaging is exactly what discards that.

    Read the result as a floor, not as a loudness. Two far field clips measured
    an identical -40.23 dBFS on different utterances, which means the peak had
    found the window's noise floor rather than the voice."""
    if not frames:
        return None
    samples = np.frombuffer(b"".join(frames), dtype=np.int16)
    if not samples.size:
        return None
    peak = float(np.abs(samples.astype(np.float64) / 32768.0).max())
    return float(20 * np.log10(peak)) if peak > 0 else None


def too_quiet(level, window_len, capacity):
    """Whether a detection should be dropped for arriving too quiet to be him.

    **Fails open on every uncertain case.** No level, a window not yet full, or
    the gate switched off all wake her. A partial window is the first 2.5
    seconds after a restart, and judging a real wake from a short buffer trades
    a visible failure for an invisible one.

    Exclusive at the floor, like WAKE_THRESHOLD: a clip sitting exactly on the
    line is not evidence against itself."""
    if not WAKE_LEVEL_GATE or level is None:
        return False
    if window_len < capacity:
        return False
    return level < WAKE_LEVEL_FLOOR
