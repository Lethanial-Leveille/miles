#!/usr/bin/env python3
"""Improve the voiceprint from real use instead of a second enrollment session.

Staged enrollment is twelve samples of eight seconds across four conditions:
ten minutes of reading prompts and moving around a room. You do it once, and
the profile stays as good as that one afternoon made it. Ordinary conversation
produces the same audio for free, in the conditions that actually occur.

Nothing here happens automatically. Samples accumulate during use; applying
them is a decision you make after seeing what it would change. That is the same
reasoning as the memory review queue: automating writes into a store that
cannot be corrected is worse than capturing by hand, and this store has already
been poisoned once. A single enrollment sample at 33 percent voiced audio
dragged the centroid to 0.63 average pairwise similarity and caused months of
rejections nobody could explain.

    python3 scripts/voiceprint.py status
    python3 scripts/voiceprint.py preview
    python3 scripts/voiceprint.py apply
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import (VOICEPRINT_PATH, VOICEPRINT_SAMPLE_CAP,
                    VOICEPRINT_LEARN_MIN_SIMILARITY, VOICEPRINT_LEARN_MIN_SECONDS)
from database import get_voiceprint_samples


def _vectors(rows):
    return np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])


def cmd_status():
    rows = get_voiceprint_samples()
    print(f"kept samples : {len(rows)} of {VOICEPRINT_SAMPLE_CAP}")
    print(f"kept when    : similarity >= {VOICEPRINT_LEARN_MIN_SIMILARITY} "
          f"and at least {VOICEPRINT_LEARN_MIN_SECONDS}s of voiced audio")
    if not rows:
        print("\nNothing yet. Talk to Nova; anything well clear of the accept bar is kept.")
        return
    mics = {r["mic"] or "unknown" for r in rows}
    print(f"microphones  : {', '.join(sorted(mics))}")
    if len(mics) > 1:
        print("\n  WARNING: samples span more than one microphone. A voiceprint does")
        print("  not transfer across capsules, so mixing them builds a centroid that")
        print("  represents neither. Clear the old ones before applying.")
    sims = [r["similarity"] for r in rows]
    print(f"similarity   : min {min(sims):.3f}  median {sorted(sims)[len(sims)//2]:.3f}  max {max(sims):.3f}")


def cmd_preview():
    rows = get_voiceprint_samples()
    if len(rows) < 5:
        print(f"Only {len(rows)} samples. Not worth recomputing yet.")
        return
    if not os.path.exists(VOICEPRINT_PATH):
        print(f"No voiceprint at {VOICEPRINT_PATH}. Enroll first.")
        return

    current = np.load(VOICEPRINT_PATH)
    vectors = _vectors(rows)

    # Blend rather than replace. These samples are real usage and the enrolled
    # centroid deliberately spans conditions real usage does not reach, so
    # throwing it away would narrow the profile to wherever he usually stands.
    learned = vectors.mean(axis=0)
    learned /= np.linalg.norm(learned)
    blended = current / np.linalg.norm(current) + learned
    blended /= np.linalg.norm(blended)

    def score(centroid):
        c = centroid / np.linalg.norm(centroid)
        return vectors @ c

    now, after = score(current), score(blended)
    print(f"samples          : {len(rows)}")
    print(f"norm of current  : {np.linalg.norm(current):.3f}  (a poisoned centroid drifts below 1.0)")
    print(f"agreement now    : median {np.median(now):.3f}  min {now.min():.3f}")
    print(f"agreement after  : median {np.median(after):.3f}  min {after.min():.3f}")
    print(f"\nmovement         : cosine {float(np.dot(current / np.linalg.norm(current), blended)):.4f}")
    print("\nA cosine near 1.0 means the update barely moves the profile, which is")
    print("the healthy case. A large move means these samples disagree with the")
    print("enrolled voice, and that is a reason to look at them rather than apply.")


def cmd_apply():
    rows = get_voiceprint_samples()
    if len(rows) < 10:
        print(f"Only {len(rows)} samples. Refusing: too few to average safely.")
        return
    mics = {r["mic"] or "unknown" for r in rows}
    if len(mics) > 1:
        print(f"Refusing: samples span {len(mics)} microphones ({', '.join(sorted(mics))}).")
        print("A centroid built across capsules represents neither of them.")
        return
    if not os.path.exists(VOICEPRINT_PATH):
        print(f"No voiceprint at {VOICEPRINT_PATH}. Enroll first.")
        return

    backup = f"{VOICEPRINT_PATH}.{datetime.now():%Y%m%dT%H%M%S}.bak"
    shutil.copy2(VOICEPRINT_PATH, backup)

    current = np.load(VOICEPRINT_PATH)
    learned = _vectors(rows).mean(axis=0)
    learned /= np.linalg.norm(learned)
    blended = current / np.linalg.norm(current) + learned
    blended /= np.linalg.norm(blended)
    np.save(VOICEPRINT_PATH, blended)

    print(f"applied {len(rows)} samples")
    print(f"backup at {backup}")
    print("Restart miles-voice, then watch verification_log. If scores fall, "
          "restore the backup.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("status", "preview", "apply"):
        sub.add_parser(name)
    args = p.parse_args()
    {"status": cmd_status, "preview": cmd_preview, "apply": cmd_apply}[args.cmd]()


if __name__ == "__main__":
    main()
