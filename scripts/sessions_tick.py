#!/usr/bin/env python3
"""Open, push, remind and settle scheduled sessions (the nightly review).

Run by miles-sessions.timer every 5 minutes; the rules are in
src/session_schedule.py and the work in sessions.tick. Safe to run by hand: a
session is made once a night, each push is sent once, and a settled session
is never settled again.

    python3 scripts/sessions_tick.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from database import init_db  # noqa: E402
import nightly_review  # noqa: E402,F401  registers the nightly_review type
import sessions  # noqa: E402

init_db()
for session_id, action in sessions.tick():
    print(f"session {session_id}: {action}", flush=True)
