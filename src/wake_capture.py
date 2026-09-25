"""What to throw away when a wake capture directory is full.

Both capture directories are ring buffers: they fill to a cap and the oldest
file is evicted. Until Sep 25 2026 that was all they did, and it quietly ate the
only part of the buffer with any value. `wake_misses` held 29 clips labelled by
ear on Sep 13; eleven days later 14 of them were gone, including the closest
calls at 0.392, 0.371 and 0.351. The verdicts survived in labels.csv, and a
verdict with no audio cannot train or evaluate anything, which is the entire
purpose the capture exists for.

So labelling is preferred over recency: an unlabelled clip goes first, and a
labelled one is only touched when nothing unlabelled is left.

**That last clause is not optional.** A plain "never delete a labelled clip"
rule fails slowly and invisibly. Once the buffer is entirely labelled it would
stop recording anything new, while the directory sits at its cap looking
perfectly healthy. A preference with a fallback keeps capture alive; a
prohibition does not.

Kept out of audio.py because that module takes the microphone lock at import, so
anything living there can only be tested with miles-voice stopped. Code that
deletes a person's hand labelled data does not get to be the code the suite
skips.
"""

import csv
import os

# Written by scripts/label_wake.py, read here. Nothing in src/ imports scripts/,
# and this does not change that: the manifest is a data file that happens to sit
# beside the clips it describes.
LABEL_FILE = "labels.csv"


def labelled_clips(directory):
    """Filenames in this directory that carry a verdict.

    An empty verdict is not a verdict: `label_wake.py` writes a row only once a
    key is pressed, but a hand edited or half written manifest should not
    protect a clip nobody has actually judged.

    Never raises. A missing manifest is the normal case for a directory nobody
    has labelled yet, and a malformed one must not take the wake loop down with
    it, so both read as nothing labelled."""
    try:
        with open(os.path.join(directory, LABEL_FILE), newline="") as f:
            return {row["file"] for row in csv.DictReader(f)
                    if row.get("file") and row.get("label")}
    except (OSError, csv.Error, KeyError, UnicodeDecodeError):
        return set()


def clips_to_evict(directory, max_files):
    """Paths to delete so the directory holds at most `max_files` clips.

    Unlabelled before labelled, oldest before newest within each group. The
    manifest itself is never a candidate: it is not a clip, and it is the record
    that outlives them."""
    try:
        names = [f for f in os.listdir(directory) if f.endswith(".wav")]
    except OSError:
        return []

    surplus = len(names) - max_files
    if surplus <= 0:
        return []

    labelled = labelled_clips(directory)

    def rank(name):
        try:
            age = os.path.getmtime(os.path.join(directory, name))
        except OSError:
            # Vanished between listing and now. Sorting it first means it is
            # named for eviction, and removing something already gone is
            # harmless; the alternative is protecting a file that is not there.
            age = 0.0
        return (name in labelled, age)

    return [os.path.join(directory, name)
            for name in sorted(names, key=rank)[:surplus]]
