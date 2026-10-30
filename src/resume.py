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

# The most words a resume ever backs up, and so the most of what she said that
# is worth keeping. The reply's earlier words are never needed again.
SAID_KEPT_WORDS = 10

# How far back to look for a place to pick up, by how long ago she was cut. A
# few seconds and the thought is still in the room; a minute and he has lost it.
REWIND_BY_GAP_S = ((10, 4), (30, 6), (60, 8))

# Said in turn, so a second resume does not open exactly like the first.
LEADS = ("Anyway,", "So,", "As I was saying,")
_CLAUSE_WORDS = {"and", "but", "so"}

_lock = threading.Lock()
_left = None    # (text, last words she said, when kept)
_lead_next = 0


def keep(text, said="", now=None):
    """Hold what she had left, replacing anything held before.

    said is what she did say of this reply, kept to its last few words so a
    resume can back up to a clause instead of starting on the cut word.

    Nothing left clears it rather than keeping an empty string, so take() never
    has to tell a blank remainder from no remainder."""
    global _left
    now = time.monotonic() if now is None else now
    words = said.split()
    with _lock:
        _left = ((text.strip(), words[-SAID_KEPT_WORDS:], now)
                 if text and text.strip() else None)


def take(now=None):
    """What she had left, once, or None when there is nothing or it has gone stale.

    Cleared either way, so one "go on" can never say it twice and a stale
    remainder cannot come back on a later try."""
    global _left
    now = time.monotonic() if now is None else now
    with _lock:
        held, _left = _left, None
    if held is None or now - held[2] > RESUME_WINDOW_S:
        return None
    return held[0]


def _rewind_words(gap_s):
    for limit_s, words in REWIND_BY_GAP_S:
        if gap_s <= limit_s:
            return words
    return SAID_KEPT_WORDS


def _pick_up_at(said, back):
    """Where in said to start again: the latest clause boundary within back words
    of the end, or back words from the end when there is none.

    A boundary is the first word of a sentence, the word after a comma, or one of
    and, but, so. The end itself counts when she was cut right after a comma or a
    full stop, which needs no backing up. Running out of words counts as the start
    of a sentence: said is her last words, so its edge is as far back as we can go."""
    floor = max(0, len(said) - back)
    for i in range(len(said), floor - 1, -1):
        if i == 0:
            return 0
        if said[i - 1][-1] in ",.!?;:":
            return i
        if i < len(said) and said[i].strip(",.!?;:").lower() in _CLAUSE_WORDS:
            return i
    return floor


def _next_lead(first_word):
    """The next lead in that does not repeat the word it is about to introduce."""
    global _lead_next
    for _ in LEADS:
        lead = LEADS[_lead_next % len(LEADS)]
        _lead_next += 1
        if lead.rstrip(",").lower() != first_word.strip(",.!?;:").lower():
            return lead
    return ""


def speech(now=None):
    """What to say to pick the reply back up, once, or None as take() would.

    Backs up to a clause she had already said, so he hears the thought start and
    not a fragment, and opens with a short lead in. It only chooses where to start
    and never reorders or rewrites her words, and nothing here calls a model."""
    global _left
    now = time.monotonic() if now is None else now
    with _lock:
        held, _left = _left, None
        if held is None or now - held[2] > RESUME_WINDOW_S:
            return None
        text, said, kept_at = held
        if not said:
            # Cut before her first word: nothing was said to pick back up.
            return text
        start = _pick_up_at(said, _rewind_words(now - kept_at))
        replay = said[start:]
        first = replay[0] if replay else text.split()[0]
        return " ".join([_next_lead(first), *replay, text]).strip()


def clear():
    """The conversation moved on: a "go on" now would not mean the old reply."""
    global _left
    with _lock:
        _left = None
