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

from config import (WAKE_LEVEL_GATE, WAKE_LEVEL_FLOOR, BARGE_IN_WAKE_FLOOR,
                    WAKE_RULE, WAKE_RULE_SCORE_BELOW, WAKE_RULE_PEAK_ABOVE)


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


def barge_in_too_quiet(level):
    """Whether a wake heard over her should be ignored for arriving too quiet.

    Its own floor, far above WAKE_LEVEL_FLOOR: over her, the model scored 0.43
    on near silence, and a "hey nova" meant to stop her is said over a speaker,
    so it is loud. Fails open on no level. A window shorter than the full one
    is still judged: it is real audio from this sentence, not the empty buffer
    after a restart that too_quiet guards against."""
    if level is None:
        return False
    return level < BARGE_IN_WAKE_FLOOR


def low_score_needs_louder(score, level, window_len, capacity):
    """Whether a low scoring wake should be dropped for not being loud.

    The same rule scripts/wake_rule.py measured: under WAKE_RULE_SCORE_BELOW,
    the window's peak must be louder than WAKE_RULE_PEAK_ABOVE, and exactly at
    it is not louder. Fails open on no level and on a window not yet full, like
    too_quiet; the script rejected a missing level, but no labelled clip had
    one, and a guard that can deafen her does not guess."""
    if not WAKE_RULE or level is None or window_len < capacity:
        return False
    return score < WAKE_RULE_SCORE_BELOW and level <= WAKE_RULE_PEAK_ABOVE
