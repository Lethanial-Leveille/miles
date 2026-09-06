#!/usr/bin/env python3
"""Label who is actually speaking in the evaluation clips, by ear.

The clips in data/speaker_eval came out of one window in which more than one
person spoke to Nova. Transcripts cannot separate them: "my internship has
ended" is plainly Lethanial and "say chicken butt" plainly is not, but most
lines are neither, and a similarity score cannot label them either since
whether the score is right is the entire question being asked.

Nothing may be measured against these until they are labelled by listening.
An unverified label would corrupt an equal error rate exactly the way an
unverified sample corrupted the voiceprint centroid in April, and it would be
harder to notice: a wrong label does not crash, it just quietly reports that
the encoder is better or worse than it is.

    python3 scripts/label_speakers.py            # label anything still blank
    python3 scripts/label_speakers.py --all      # revisit everything
    python3 scripts/label_speakers.py --status   # counts, no playback

Labels: m = me (Lethanial), o = other person, u = unclear or both, s = skip.
Anything left unclear stays out of the evaluation rather than being guessed.
"""

import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import tts                                                  # noqa: E402

EVAL_DIR = os.path.expanduser("~/miles/data/speaker_eval")
MANIFEST = os.path.join(EVAL_DIR, "manifest.csv")

CHOICES = {"m": "me", "o": "other", "u": "unclear"}


def load():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def save(rows):
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def status(rows):
    counts = {}
    for row in rows:
        counts[row["label"] or "unlabelled"] = counts.get(row["label"] or "unlabelled", 0) + 1
    print("\n".join(f"  {k:12} {v}" for k, v in sorted(counts.items())))
    me = counts.get("me", 0)
    other = counts.get("other", 0)
    print(f"\n  usable for an EER: {me} me, {other} other")
    if other < 5:
        print("  not enough impostor clips yet to trust a false acceptance rate")


def play(path):
    subprocess.run(["aplay", "-D", tts.SPEAKER_DEVICE, path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="revisit labelled rows too")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    rows = load()
    if args.status:
        status(rows)
        return

    todo = [r for r in rows if args.all or not r["label"]]
    print(f"{len(todo)} to label. m = me, o = other, u = unclear, s = skip, q = quit\n")

    for i, row in enumerate(todo, 1):
        path = os.path.join(EVAL_DIR, row["file"])
        if not os.path.exists(path):
            continue
        print(f"[{i}/{len(todo)}] {row['created_at'][11:19]}  "
              f"sim={float(row['similarity']):.3f}  {row['duration_s']}s")
        print(f"    {row['transcript']!r}")
        if row["label"]:
            print(f"    currently: {row['label']}")

        while True:
            play(path)
            answer = input("    m/o/u/s/q (enter to replay): ").strip().lower()
            if answer == "":
                continue
            if answer == "q":
                save(rows)
                print("saved")
                return
            if answer == "s":
                break
            if answer in CHOICES:
                row["label"] = CHOICES[answer]
                break
            print("    m = me, o = other, u = unclear, s = skip, q = quit")
        print()

    save(rows)
    print("saved\n")
    status(rows)


if __name__ == "__main__":
    main()
