"""Sessions he taught that could not be logged in Evallo, kept to log later.

Two reasons land a session here: the student is not mapped to him in Evallo
yet (Prep Academy adds students after the first lesson sometimes), or the
student is mapped but no session exists at that time. Nothing here is ever
created in Evallo automatically; this is a list for him, and booking sessions
is a later phase.

Stored as data/evallo_unlogged.json, a short list read and written whole. It
holds student first names and lesson times, so it lives under data/, which
git ignores.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evallo_matching import to_eastern

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "evallo_unlogged.json"

NOT_MAPPED = "not_mapped"
NO_SESSION = "no_session"


def load(path=PATH):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return []


def add(student, start, minutes, reason, path=PATH, now=None):
    """Add one session, once. The same student at the same time is kept once
    however many times it is tried; the reason is updated, since a student
    who gets mapped later moves from not_mapped to no_session."""
    start = to_eastern(start)
    entry = {
        "student": student.strip(),
        "date": start.date().isoformat(),
        "time": start.strftime("%H:%M"),
        "length_minutes": minutes,
        "reason": reason,
        "recorded_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }
    rows = load(path)
    key = (entry["student"].casefold(), entry["date"], entry["time"])
    rows = [r for r in rows if (r["student"].casefold(), r["date"], r["time"]) != key]
    rows.append(entry)
    rows.sort(key=lambda r: (r["date"], r["time"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    os.replace(tmp, path)   # never a half written list
    return entry


def remove(student, start, path=PATH):
    """Drop a session once it has been logged. Returns whether one was there."""
    start = to_eastern(start)
    key = (student.strip().casefold(), start.date().isoformat(), start.strftime("%H:%M"))
    rows = load(path)
    kept = [r for r in rows if (r["student"].casefold(), r["date"], r["time"]) != key]
    if len(kept) == len(rows):
        return False
    path.write_text(json.dumps(kept, indent=2))
    return True
