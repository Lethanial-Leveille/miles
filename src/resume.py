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
from collections import namedtuple

# A resume has two wordings. spoken is what he hears, lead in included. written
# is what history keeps: the same words with no lead in and her own capitals,
# because history is what she told him and the lead in is only a spoken cue.
# lead_words is how many words of spoken the lead in took, and replayed is how
# many words at the end of the cut reply's row this restarts, so finishing it
# can replace that partial sentence instead of repeating it.
Plan = namedtuple("Plan", "spoken written lead_words replayed")

# Long enough to answer someone at the door and come back to it, short enough
# that a "go on" minutes later is not read as finishing a thought nobody holds.
RESUME_WINDOW_S = 120

# A resume restarts the sentence she was cut in, the way a person picks a thought
# back up; a fragment opening on "but" read wrong and synthesized with the wrong
# intonation (Oct 30 2026, in the room). Past this many words of it already said,
# replaying the whole sentence for its last word is the bigger annoyance, so she
# backs up to its last usable comma instead.
LONG_SENTENCE_WORDS = 20

# Under this, the interruption was a breath and she simply carries on; a lead in
# after three seconds sounded scripted. From here on she signals she is coming
# back to something. His call, Oct 30 2026.
LEAD_IN_AFTER_S = 10

# Said in turn, never the same twice in a row. "As I was saying," was dropped
# as the most scripted of the three.
LEADS = ("Anyway,", "So,")

# A clause opening on one of these is not a place to start: it hangs off the
# clause before it, which is what made "Anyway, but" wrong.
_CLAUSE_WORDS = {"and", "but", "so"}

_lock = threading.Lock()
_left = None    # (text, the cut sentence's words so far, when kept, id of the saved reply)
_lead_next = 0


def keep(text, said="", message_id=None, now=None):
    """Hold what she had left, replacing anything held before.

    said is what she did say of this reply, kept from the start of the sentence
    she was cut in, so a resume can restart that sentence. Empty when she was
    cut cleanly between sentences, and then nothing is replayed.

    message_id is the history row saved for the cut reply, so finishing it can
    amend that row instead of leaving it saying she was cut off.

    Nothing left clears it rather than keeping an empty string, so take() never
    has to tell a blank remainder from no remainder."""
    global _left
    now = time.monotonic() if now is None else now
    words = said.split()
    # None, not empty: cut before her first word is a reply he never began to
    # hear, and gets no lead in; cut between sentences is a reply he did.
    sentence = _cut_sentence(words) if words else None
    with _lock:
        _left = ((text.strip(), sentence, now, message_id)
                 if text and text.strip() else None)


def _cut_sentence(words):
    """The words after the last full stop, question mark or exclamation."""
    for i in range(len(words), 0, -1):
        if words[i - 1][-1] in ".!?":
            return words[i:]
    return words


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


def _bare(word):
    return word.strip(",.!?;:").lower()


def _pick_up_at(sentence):
    """Where in the cut sentence to start again: its first word, or for a long one
    the latest clause after a comma that does not open on and, but or so.

    A clause right at the cut is skipped: there is nothing of it to replay."""
    if len(sentence) <= LONG_SENTENCE_WORDS:
        return 0
    for i in range(len(sentence) - 1, 0, -1):
        if sentence[i - 1].endswith(",") and _bare(sentence[i]) not in _CLAUSE_WORDS:
            return i
    return 0


def _next_lead(avoid):
    """The next lead in whose word is not in avoid: the word it introduces, and
    the lead the last resume opened with, so two in a row never match."""
    global _lead_next
    for _ in LEADS:
        lead = LEADS[_lead_next % len(LEADS)]
        _lead_next += 1
        if _bare(lead) not in avoid:
            return lead
    return ""


def _after_lead(word):
    """A sentence's first word as it reads after a lead in: lower case, but "I"
    stays "I". Only the written history sees the case; the voice does not."""
    if word == "I" or word.startswith("I'"):
        return word
    return word[:1].lower() + word[1:]


def speech(now=None):
    """What to say to pick the reply back up, once, or None as take() would."""
    planned = plan(now)
    return planned.spoken if planned else None


def plan(now=None):
    """How to pick the reply back up, once, as a Plan, or None as take() would.

    Restarts the sentence she was cut in, so he hears the thought start and not a
    fragment, and opens with a lead in only after a real gap. It chooses where to
    start and the lead, and otherwise never rewrites her words; nothing here
    calls a model."""
    global _left
    now = time.monotonic() if now is None else now
    with _lock:
        held, _left = _left, None
        if held is None or now - held[2] > RESUME_WINDOW_S:
            return None
        text, sentence, kept_at, _ = held
        if sentence is None:
            return Plan(text, text, 0, 0)
        replay = sentence[_pick_up_at(sentence):]
        words = [*replay, *text.split()]
        if now - kept_at < LEAD_IN_AFTER_S:
            return Plan(" ".join(words), " ".join(words), 0, len(replay))
        # A sentence that opened on a lead in, cut again: that lead is dropped,
        # never stacked, and the new one must differ from it.
        avoid = set()
        if replay and replay[0] in LEADS:
            avoid.add(_bare(words.pop(0)))
        avoid.add(_bare(words[0]))
        written = " ".join(words)
        lead = _next_lead(avoid)
        if not lead:
            return Plan(written, written, 0, len(replay))
        spoken = " ".join([lead, _after_lead(words[0]), *words[1:]])
        return Plan(spoken, written, len(lead.split()), len(replay))


def attach(message_id):
    """Point what is held at a different history row, if anything is held.

    For a go on: the remainder of a cut resume is held before the go on saves its
    reply, so it starts out pointing at the reply before that one."""
    global _left
    with _lock:
        if _left is not None:
            _left = (*_left[:3], message_id)


def held(now=None):
    """Whether speech() would say something right now, without taking it.

    For a gate that has to decide whether a "go on" means anything before the
    turn is committed to the local path."""
    now = time.monotonic() if now is None else now
    with _lock:
        return _left is not None and now - _left[2] <= RESUME_WINDOW_S


def message_id(now=None):
    """The history row of the held reply, without taking it, or None.

    Read before speech() because speech() lets go of everything, the id with it."""
    now = time.monotonic() if now is None else now
    with _lock:
        if _left is None or now - _left[2] > RESUME_WINDOW_S:
            return None
        return _left[3]


def clear():
    """The conversation moved on: a "go on" now would not mean the old reply."""
    global _left
    with _lock:
        _left = None
