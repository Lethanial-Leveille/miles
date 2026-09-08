#!/usr/bin/env python3
"""Label what the wake word model actually heard, by ear.

There are two questions and they need opposite fixes, which is why a score on
its own answers neither:

    wake_misses   she did not answer. Was that a real "hey nova" she missed, or
                  the room, correctly ignored?
    wake_hits     she answered. Was that a real "hey nova", or the television?

hey_nova.onnx is dated Apr 8 2026. It predates the mic gain tuning of Aug 10
and it predates this apartment. CLAUDE.md records the old room separating
cleanly, successful wakes from 0.520 up and failures topping out at 0.365, with
an empty band between. That is not what the new room does: observed wakes run
0.43, 0.46, 0.50, 0.64 and 0.75, and the low one was genuine while the high one
was not. Overlapping in both directions means no threshold fixes it, and the
model is the thing to look at.

This builds the set that makes that measurable. Without labels, "the model is
weak" and "those were never attempts" are indistinguishable, and there is
nothing to evaluate a retrain against.

Labelling by ear, and only by ear. A transcript cannot do it: the trigger audio
is the phrase itself, not the command after it, and the score is the thing
being judged so it cannot also be the judge. label_speakers.py makes the same
argument, and for the same reason: a wrong label does not crash, it quietly
reports the model as better or worse than it is.

    python3 scripts/label_wake.py hits            # label wakes that fired
    python3 scripts/label_wake.py misses          # label wakes that did not
    python3 scripts/label_wake.py hits --min 0.3  # only the loudest cases
    python3 scripts/label_wake.py --status        # counts, no playback

Labels: y = yes, I said the wake phrase
        n = no, I did not
        u = unclear, or someone else said it
        r = replay      s = skip      q = quit

Anything left unclear stays out of the evaluation rather than being guessed.
"""

import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from config import (WAKE_MISS_DIR, WAKE_HIT_DIR, WAKE_THRESHOLD,  # noqa: E402
                    SPEAKER_NAME_HINT)

SETS = {"hits": WAKE_HIT_DIR, "misses": WAKE_MISS_DIR}
LABELS = {"y": "wake_phrase", "n": "not_wake_phrase", "u": "unclear"}
FIELDS = ["file", "score", "label"]


def _manifest_path(directory):
    return os.path.join(directory, "labels.csv")


def _load(directory):
    """Existing labels, so a session can be stopped and resumed.

    Keyed by filename rather than by index, because the capture directories are
    pruned as they fill and an index would silently point at a different clip
    after any eviction."""
    path = _manifest_path(directory)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def _save(directory, rows):
    path = _manifest_path(directory)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])


def _clips(directory, minimum):
    """Filenames, worst first.

    The score leads the filename precisely so this sort is meaningful: for
    hits, the highest scoring false positive is the most damaging, and for
    misses the highest scoring failure is the closest call."""
    if not os.path.isdir(directory):
        return []
    names = [f for f in os.listdir(directory) if f.endswith(".wav")]
    scored = []
    for name in names:
        try:
            scored.append((float(name.split("_")[0]), name))
        except ValueError:
            continue
    scored.sort(reverse=True)
    return [(s, n) for s, n in scored if s >= minimum]


def _speaker_device():
    """Resolved by name, since ALSA card numbers shift between boots.

    Duplicated from tts.py rather than imported, because importing tts builds
    an ElevenLabs client at module scope and labelling should not need an API
    key. enroll.py already duplicates this for the same reason; if a fourth
    caller appears, move it somewhere shared instead of copying again."""
    try:
        with open("/proc/asound/cards") as f:
            for block in f.read().split("\n"):
                if SPEAKER_NAME_HINT in block:
                    return f"plughw:{block.strip().split()[0]},0"
    except OSError:
        pass
    return "default"


def cmd_status():
    for name, directory in SETS.items():
        clips = _clips(directory, 0.0)
        rows = _load(directory)
        done = sum(1 for _, n in clips if n in rows)
        print(f"{name:8} {len(clips):>4} clips, {done:>4} labelled")
        if not clips:
            continue
        counts = {}
        for _, n in clips:
            if n in rows:
                counts[rows[n]["label"]] = counts.get(rows[n]["label"], 0) + 1
        for label, count in sorted(counts.items()):
            print(f"           {label:<18} {count}")
        bands = {"<0.20": 0, "0.20-0.40": 0, ">=0.40": 0}
        for score, _ in clips:
            key = "<0.20" if score < 0.2 else ("0.20-0.40" if score < 0.4 else ">=0.40")
            bands[key] += 1
        print(f"           by score: {bands}")


def cmd_label(which, minimum, relabel):
    directory = SETS[which]
    clips = _clips(directory, minimum)
    if not clips:
        print(f"No clips in {directory} at or above {minimum}.")
        return

    rows = _load(directory)
    todo = [(s, n) for s, n in clips if relabel or n not in rows]
    if not todo:
        print(f"All {len(clips)} clips already labelled. Use --relabel to revisit.")
        return

    device = _speaker_device()
    question = ("She ANSWERED on this. Did you say the wake phrase?"
                if which == "hits" else
                "She did NOT answer. Did you say the wake phrase?")

    print(f"{len(todo)} to label, loudest first. Threshold is {WAKE_THRESHOLD}.")
    print("  y = yes, I said it    n = no    u = unclear    r = replay")
    print("  s = skip              q = quit\n")

    for index, (score, name) in enumerate(todo, 1):
        path = os.path.join(directory, name)
        print("=" * 68)
        print(f"  [{index}/{len(todo)}]  score {score:.3f}   {name}")
        print(f"  {question}")

        while True:
            subprocess.run(["aplay", "-D", device, path], capture_output=True)
            answer = input("  > ").strip().lower()
            if answer == "r":
                continue
            break

        if answer == "q":
            break
        if answer == "s" or answer not in LABELS:
            print("  skipped\n")
            continue

        rows[name] = {"file": name, "score": f"{score:.3f}",
                      "label": LABELS[answer]}
        _save(directory, rows)
        print(f"  -> {LABELS[answer]}\n")

    print(f"\nLabels in {_manifest_path(directory)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="?", choices=sorted(SETS),
                        help="which capture set to label")
    parser.add_argument("--min", type=float, default=0.0,
                        help="only clips scoring at or above this")
    parser.add_argument("--relabel", action="store_true",
                        help="revisit clips already labelled")
    parser.add_argument("--status", action="store_true",
                        help="counts only, no playback")
    args = parser.parse_args()

    if args.status or not args.which:
        cmd_status()
        return
    cmd_label(args.which, args.min, args.relabel)


if __name__ == "__main__":
    main()
