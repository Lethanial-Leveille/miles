#!/usr/bin/env python3
"""What a stricter wake rule would have cost, from his labels: nothing runs live.

Asked Oct 2 2026, after his roommates heard her chime on false wakes. On 26
array wakes he labelled (18 his, 8 not), no threshold separated them: 0.55
cut 6 false wakes and lost 3 of his 18. His 4 low scoring wakes were all loud
and 6 of the 7 false ones beside them were not, so the candidate is a rule on
both: a score under SCORE_BELOW also needs a peak louder than PEAK_ABOVE. Four
of his wakes, all from his seat, are no basis for a guard that can deafen her,
and one of his real wakes scored 0.656 at minus 27 dBFS, so this reports the
rule against every labelled wake before anything acts on it.

The level is the wake window's peak, read from the clip in data/wake_hits; on
all 26 it matched the level the voice loop logged to the tenth of a dB, which
is why no live logging was added.

    python3 scripts/wake_rule.py                       # the array era, labelled
    python3 scripts/wake_rule.py --score 0.6 --peak -9
"""
import argparse
import csv
import os
import re
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import WAKE_HIT_DIR
from wake_gate import window_peak_dbfs

ARRAY_SINCE = "20261001T111200"   # the 48kHz firmware went live


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", type=float, default=0.6, help="SCORE_BELOW")
    parser.add_argument("--peak", type=float, default=-9.0, help="PEAK_ABOVE, dBFS")
    parser.add_argument("--after", default=ARRAY_SINCE)
    args = parser.parse_args()

    rows = []
    with open(os.path.join(WAKE_HIT_DIR, "labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            stamp = re.search(r"_(\d{8}T\d{6})", row["file"])
            path = os.path.join(WAKE_HIT_DIR, row["file"])
            if not stamp or stamp.group(1) < args.after or not os.path.exists(path):
                continue
            with wave.open(path) as w:
                frames = [w.readframes(w.getnframes())]
            peak = window_peak_dbfs(frames)
            rejected = (float(row["score"]) < args.score
                        and (peak is None or peak <= args.peak))
            rows.append((float(row["score"]), peak, row["label"] == "wake_phrase", rejected))

    his = [r for r in rows if r[2]]
    false = [r for r in rows if not r[2]]
    print(f"{len(rows)} labelled wakes since {args.after}: {len(his)} his, {len(false)} not")
    print(f"rule: score under {args.score} needs a peak above {args.peak} dBFS")
    print(f"  his wakes it would have lost: {sum(r[3] for r in his)} of {len(his)}")
    print(f"  false wakes it would have cut: {sum(r[3] for r in false)} of {len(false)}")
    for score, peak, mine, rejected in sorted(rows):
        if rejected and mine:
            print(f"  LOST: his wake at {score:.3f}, peak {peak:.1f} dBFS")


if __name__ == "__main__":
    main()
