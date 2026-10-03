#!/usr/bin/env python3
"""Raise the weights his last workouts earned, in his Hevy routines.

Run by miles-hevy.timer every 15 minutes; the work and the reasons are in
src/hevy_progress.py. Safe to run by hand: a workout is only ever judged once.

    python3 scripts/hevy_progress.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import hevy_progress  # noqa: E402
from database import init_db  # noqa: E402

init_db()
for kind, routine, exercise, from_lb, to_lb in hevy_progress.run():
    print(f"{routine}: {exercise} {kind} {from_lb:g} to {to_lb:g} pounds", flush=True)
