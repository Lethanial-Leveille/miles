"""Choosing Whisper's audio window, and cutting long recordings to fit it.

Pure functions, no microphone and no whisper binary, so they can be tested
without touching the live audio device the voice service holds.
"""

import numpy as np

from config import WHISPER_AUDIO_CTX, WHISPER_AUDIO_CTX_LONG

# The longest clip the fast window has evidence behind. It was validated in
# August on recordings of 3, 6, 10 and 14 seconds, and never above that.
# Checked Sep 13 2026 on his five longest archived recordings, all 18 seconds:
# the full window changed three of the five transcripts, and in one it kept
# trailing words the fast window dropped, "I'll put it" against "I'll put it on
# the floor". The fast window's own twenty second limit is not a safe edge, so
# the line sits just above what was actually validated.
_FAST_WINDOW_SECONDS = 15.0


def audio_ctx_for(seconds):
    """The fast window for normal commands, Whisper's full window past it.

    The fast window was validated in August and saves most of a second per
    turn, but it silently drops any audio past twenty seconds, and it was never
    validated past fourteen. That is why MAX_RECORD had to sit at 18, which cut
    him off mid explanation on Sep 13 2026. Now only a recording longer than
    fifteen seconds pays for the full window: about 0.6s more on an 18 second
    clip, measured that day."""
    return WHISPER_AUDIO_CTX if seconds <= _FAST_WINDOW_SECONDS else WHISPER_AUDIO_CTX_LONG


def segment_bounds(samples, rate, max_seconds, search_seconds=4.0, window_ms=200):
    """Sample ranges covering the whole recording, each at most max_seconds.

    Each cut goes at the quietest short window in the last search_seconds
    before the limit, so it lands in a breath between words rather than
    through one. A cut through a word garbles it in both pieces."""
    max_len = int(max_seconds * rate)
    search = int(search_seconds * rate)
    window = max(2, int(window_ms / 1000 * rate))
    bounds, start, total = [], 0, len(samples)

    while total - start > max_len:
        lo = max(start + 1, start + max_len - search)
        hi = start + max_len
        cut, quietest = hi, None
        for w_start in range(lo, hi - window + 1, window // 2):
            level = float(np.mean(np.abs(samples[w_start:w_start + window].astype(np.int32))))
            if quietest is None or level < quietest:
                cut, quietest = w_start + window // 2, level
        bounds.append((start, cut))
        start = cut

    bounds.append((start, total))
    return bounds
