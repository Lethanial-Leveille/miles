#!/usr/bin/env python3
"""What a resume would say for every real cut she has had: nothing runs live.

Asked Oct 30 2026, as rung 3 of docs/TESTING.md, after a resume in the room said
"Anyway, but most granola is heavy on sugar and oil". Every reply he cut off is
in conversation_history ending in the cut off mark, holding what she said; the
timing_log row written beside it holds the whole reply, and the difference is
what she had left. This rebuilds each cut and runs resume.plan() on it, so a
change to resume.py can be read against his real interruptions before anyone
hears it. The database is opened read only.

The gap before a real resume was never logged, so each cut is shown both ways:
a quick go on (no lead in) and one after a real gap (with one).

    python3 scripts/replay_resume.py                    # the code in this checkout
    python3 scripts/replay_resume.py --compare df6ce14  # and resume.py at a commit
"""
import argparse
import importlib.util
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import resume
from config import DB_PATH

CUT_OFF_MARK = "[You were cut off here.]"
QUICK_S, LATE_S = 3.0, 40.0
# The timing row is written within a second of the history row; anything
# further off belongs to a different turn.
MATCH_WITHIN_S = 5.0
SHOWN_CHARS = 170


def _resume_at(ref):
    """resume.py as it was at a commit, loaded as its own module."""
    source = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:src/resume.py"],
                            capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "resume_at_ref.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("resume_at_ref", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuts(db):
    """Each cut reply as (row id, when, what she said, what she had left), or a
    reason it could not be rebuilt in place of the last two."""
    cut_rows = db.execute(
        "SELECT id, created_at, content FROM conversation_history "
        "WHERE role = 'assistant' AND content LIKE ? ORDER BY id",
        (f"%{CUT_OFF_MARK}",)).fetchall()
    timings = [(datetime.fromisoformat(at), response) for at, response in db.execute(
        "SELECT created_at, response FROM timing_log WHERE response IS NOT NULL")]
    for row_id, at, content in cut_rows:
        when = datetime.fromisoformat(at)
        said = content[:-len(CUT_OFF_MARK)].rstrip()
        near = [(abs((t - when).total_seconds()), r) for t, r in timings]
        near = [(d, r) for d, r in near if d <= MATCH_WITHIN_S]
        if not near:
            yield row_id, when, said, None, "no timing row beside it"
            continue
        reply = min(near)[1]
        # The voice strips bracket cues before speaking, so history never has them.
        for candidate in (reply, re.sub(r"\[[^\]]*\]\s*", "", reply)):
            if candidate.startswith(said):
                # A go on's reply is logged as what she said, mark and all.
                left = candidate[len(said):].replace(CUT_OFF_MARK, "").strip()
                break
        else:
            yield row_id, when, said, None, "what she said is not the start of the logged reply"
            continue
        if not left:
            yield row_id, when, said, None, "nothing was left: a go on that was cut, logged as said"
            continue
        yield row_id, when, said, left, None


def _plan(module, said, left, gap):
    module.clear()
    module.keep(left, said=said, now=0.0)
    if hasattr(module, "plan"):
        return module.plan(now=gap)
    spoken = module.speech(now=gap)
    return None if spoken is None else (spoken, None, None, None)


def _short(text):
    return text if len(text) <= SHOWN_CHARS else text[:SHOWN_CHARS].rstrip() + " …"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", metavar="REF",
                        help="also show what resume.py at this commit would say")
    args = parser.parse_args()
    older = _resume_at(args.compare) if args.compare else None

    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    shown = skipped = 0
    for row_id, when, said, left, why in _cuts(db):
        print(f"\n#{row_id}  {when:%b %d %H:%M}")
        if why:
            skipped += 1
            print(f"  skipped: {why}")
            continue
        shown += 1
        tail = " ".join(said.split()[-10:])
        print(f"  cut:      …{tail} ‖ {_short(left)}")
        if older:
            print(f"  before:   {_short(_plan(older, said, left, QUICK_S)[0])}")
            print(f"  before+:  {_short(_plan(older, said, left, LATE_S)[0])}")
        quick = _plan(resume, said, left, QUICK_S)
        late = _plan(resume, said, left, LATE_S)
        print(f"  now:      {_short(quick.spoken)}")
        print(f"  now+:     {_short(late.spoken)}")
        kept = said.split()[:len(said.split()) - quick.replayed]
        print(f"  history:  …{' '.join(kept[-6:])} ‖ {_short(quick.written)}")
    print(f"\n{shown} cuts replayed, {skipped} skipped. "
          f"now and before at a {QUICK_S:.0f}s gap, now+ and before+ at {LATE_S:.0f}s. "
          "‖ marks the cut, and in history where the restarted words join the row.")


if __name__ == "__main__":
    main()
