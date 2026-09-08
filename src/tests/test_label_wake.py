"""Labelling must survive the directories being pruned underneath it.

Both capture directories are ring buffers: they fill to a cap and the oldest
files are evicted. A labelling tool that keyed on position rather than filename
would, after any eviction, silently attach an old verdict to a different clip.
That does not crash and it does not look wrong, which is the same failure
label_speakers.py warns about one level up.
"""

import csv
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "label_wake.py")

_spec = importlib.util.spec_from_file_location("label_wake", _PATH)
lw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lw)


def _clip_dir(tmp_path, scores):
    for i, score in enumerate(scores):
        (tmp_path / f"{score:.3f}_2026090{i}T120000_000.wav").write_bytes(b"RIFF")
    return str(tmp_path)


def test_clips_come_back_worst_first(tmp_path):
    """The score leads the filename so this sort means something. For hits the
    highest scoring false positive is the most damaging one to see; for misses
    the highest scoring failure is the closest call."""
    directory = _clip_dir(tmp_path, [0.11, 0.87, 0.42])
    assert [s for s, _ in lw._clips(directory, 0.0)] == [0.87, 0.42, 0.11]


def test_the_minimum_filters_out_the_noise_floor(tmp_path):
    """371 of 400 real miss clips score under 0.20 and are empty room. Labelling
    all of them is how a labelling set ends up abandoned half finished."""
    directory = _clip_dir(tmp_path, [0.05, 0.18, 0.31, 0.55])
    assert [s for s, _ in lw._clips(directory, 0.20)] == [0.55, 0.31]


def test_labels_are_keyed_by_filename_not_position(tmp_path):
    """The property that makes this safe against pruning.

    A verdict recorded against index 3 points at a different clip the moment
    one file is evicted. A verdict recorded against a filename either finds its
    clip or does not."""
    directory = _clip_dir(tmp_path, [0.90, 0.50, 0.10])
    names = [n for _, n in lw._clips(directory, 0.0)]

    lw._save(directory, {names[1]: {"file": names[1], "score": "0.500",
                                    "label": "not_wake_phrase"}})
    os.remove(os.path.join(directory, names[0]))

    reloaded = lw._load(directory)
    assert names[1] in reloaded
    assert reloaded[names[1]]["label"] == "not_wake_phrase"


def test_labelling_is_resumable(tmp_path):
    directory = _clip_dir(tmp_path, [0.90, 0.50])
    names = [n for _, n in lw._clips(directory, 0.0)]

    lw._save(directory, {names[0]: {"file": names[0], "score": "0.900",
                                    "label": "wake_phrase"}})
    rows = lw._load(directory)
    rows[names[1]] = {"file": names[1], "score": "0.500", "label": "unclear"}
    lw._save(directory, rows)

    assert len(lw._load(directory)) == 2


def test_an_absent_directory_is_not_an_error(tmp_path):
    """wake_hits does not exist until the first wake fires after this ships."""
    assert lw._clips(str(tmp_path / "nope"), 0.0) == []


def test_files_without_a_score_prefix_are_skipped(tmp_path):
    """labels.csv lives in the same directory and is not a clip."""
    directory = _clip_dir(tmp_path, [0.42])
    (tmp_path / "notes.wav").write_bytes(b"RIFF")
    assert len(lw._clips(directory, 0.0)) == 1


def test_both_capture_sets_are_reachable():
    """Two questions needing opposite fixes. A tool covering only one of them
    would leave the false wake case with no audio behind it at all, which is
    the hole this was written to close."""
    assert set(lw.SETS) == {"hits", "misses"}


def test_the_three_verdicts_are_distinct():
    """unclear must not collapse into either answer. A guessed label is worse
    than a missing one, because it corrupts the evaluation silently."""
    assert len(set(lw.LABELS.values())) == 3
