"""Keep what Nova said, as she said it, so a reply that sounded off can be
replayed and pointed at instead of described.

On Sep 28 2026 he heard her as "coherent but a little off", with pauses where
there should not be any, and could not say more than that. Nothing kept her
output, so there was nothing to point at.

Each reply is one wav and one txt with the same stem. The wav is the pieces in
the order they played, with the silence measured before each later piece put
back in, so a stall between requests is heard as silence and a pause v3 made is
heard as her pausing. The txt holds what ElevenLabs was actually sent, after the
ellipses and pronunciation swaps, because that is what the voice read; history
keeps what Nova wrote, which is a different thing.

Never raises. This exists for diagnosis, and losing a copy of a reply is not a
reason to disturb the turn that produced it.
"""

import os
import wave
from datetime import datetime

from config import ARCHIVE_SPOKEN, SPOKEN_DIR, SPOKEN_MAX_FILES
from wake_capture import clips_to_evict

# pcm_22050 is raw S16_LE mono. play() opens aplay with the same values; the
# app's mp3 path never reaches here.
_RATE, _WIDTH = 22050, 2


def _silence(ms):
    frames = int(_RATE * max(ms or 0, 0) / 1000.0)
    return b"\x00" * (frames * _WIDTH)


def save_reply(pieces, directory=SPOKEN_DIR, max_files=SPOKEN_MAX_FILES):
    """pieces is a list of (text_sent, pcm, gap_ms), in the order they played.
    gap_ms is the silence heard before that piece and is ignored for the first,
    whose wait is time to first audio, not a pause inside the reply.

    Returns the wav path, or None when off, empty, or failed."""
    if not ARCHIVE_SPOKEN or not any(pcm for _, pcm, _ in pieces):
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        stem = os.path.join(directory, datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3])

        with wave.open(stem + ".wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(_WIDTH)
            w.setframerate(_RATE)
            for i, (_, pcm, gap_ms) in enumerate(pieces):
                if i:
                    w.writeframes(_silence(gap_ms))
                w.writeframes(pcm)

        lines = []
        for i, (text, _, gap_ms) in enumerate(pieces):
            if i:
                lines.append(f"[{gap_ms or 0:.0f}ms before this piece]")
            lines.append(text)
        with open(stem + ".txt", "w") as f:
            f.write("\n".join(lines) + "\n")

        for stale in clips_to_evict(directory, max_files):
            for path in (stale, stale[:-len(".wav")] + ".txt"):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        return stem + ".wav"
    except Exception as exc:
        print(f"Could not archive what Nova said: {exc}", flush=True)
        return None
