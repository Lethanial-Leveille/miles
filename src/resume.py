"""What she had left to say when he cut her off, held so a "go on" can finish it.

brain.py puts the remainder here when a reply was interrupted and clears it when
the conversation moves on; nothing here decides whether to say it. That belongs
to whoever takes it: a recording that came back empty, or him asking her to
continue.

State is per process, like pending_action. A reply cut in the room is resumed in
the room, and the app has no reply of this kind to finish.
"""

import threading
import time

# Long enough to answer someone at the door and come back to it, short enough
# that a "go on" minutes later is not read as finishing a thought nobody holds.
RESUME_WINDOW_S = 120

_lock = threading.Lock()
_left = None    # (text, when it was kept)


def keep(text, now=None):
    """Hold what she had left, replacing anything held before.

    Nothing left clears it rather than keeping an empty string, so take() never
    has to tell a blank remainder from no remainder."""
    global _left
    now = time.monotonic() if now is None else now
    with _lock:
        _left = (text.strip(), now) if text and text.strip() else None


def take(now=None):
    """What she had left, once, or None when there is nothing or it has gone stale.

    Cleared either way, so one "go on" can never say it twice and a stale
    remainder cannot come back on a later try."""
    global _left
    now = time.monotonic() if now is None else now
    with _lock:
        held, _left = _left, None
    if held is None or now - held[1] > RESUME_WINDOW_S:
        return None
    return held[0]


def clear():
    """The conversation moved on: a "go on" now would not mean the old reply."""
    global _left
    with _lock:
        _left = None
