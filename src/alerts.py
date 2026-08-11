"""Deferred spoken alerts from timers and reminders.

Timer and reminder threads used to call tts.speak directly. speak_lock kept
Nova from talking over herself, but nothing kept her from talking over
Lethanial: capture holds no lock, so an open mic looks exactly like an idle
room to a background thread. On Aug 11 2026 a one minute timer fired 2 seconds
into a follow up window and spoke straight into the question being asked.

So alerts no longer speak. They queue here, and the voice loop decides when.
Because that loop is single threaded through capture, any point it chooses is
by definition not mid recording. That is the whole mechanism; there is no mic
state flag to keep in sync.

Two ways out of the queue:

  fold      An alert younger than FOLD_MAX_AGE_S is attached to the next turn
            so Nova says it in her own words alongside whatever she was
            already answering, which is what a person would do.

  speak     Anything else is announced on its own at the next safe moment.

The split exists because folding alone is not safe. If Lethanial never speaks
again, a folded alert is never delivered, and a timer that silently does not go
off is worse than one that interrupts. The age cap bounds that: after fifteen
seconds it stops waiting for a conversation that may not come.
"""

import threading
import time
from collections import namedtuple
from datetime import datetime

from database import log_alert

Alert = namedtuple("Alert", "kind text summary fired_at fired_wall")

# Younger than this folds into a turn; older gets announced standalone.
# Fifteen seconds is roughly two exchanges, so an alert that fires mid
# conversation rides along with the next thing Nova says, while one that fires
# into silence does not wait around for a conversation that is not coming.
FOLD_MAX_AGE_S = 15

_lock = threading.Lock()
_pending = []


def fire(kind, text, summary):
    """Queue an alert. Called from timer and reminder threads, never speaks.

    `text` is the standalone announcement. `summary` is the compressed form
    handed to the model when the alert is folded into a turn, where Nova
    supplies her own phrasing and the stock sentence would fight it.
    """
    with _lock:
        _pending.append(Alert(kind, text, summary,
                              time.monotonic(), datetime.now().isoformat()))


def _take(predicate, mode):
    now = time.monotonic()
    with _lock:
        taken = [a for a in _pending if predicate(now, a)]
        for alert in taken:
            _pending.remove(alert)

    for alert in taken:
        log_alert(kind=alert.kind, content=alert.summary,
                  fired_at=alert.fired_wall, mode=mode,
                  delay_ms=(now - alert.fired_at) * 1000.0)
    return taken


def take_for_fold(max_age=FOLD_MAX_AGE_S):
    """Alerts fresh enough to be worked into the response Nova is about to give."""
    return _take(lambda now, a: now - a.fired_at <= max_age, "folded")


def take_for_speech():
    """Everything still queued, to be announced on its own.

    Called only from the voice loop at points where the mic is not open."""
    return _take(lambda now, a: True, "spoken")


def pending_count():
    with _lock:
        return len(_pending)


def clear():
    """Drop everything without delivering. Tests only."""
    with _lock:
        _pending.clear()
