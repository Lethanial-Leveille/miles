"""Pruning a wake capture directory must not eat the labelling.

`wake_misses` held 29 clips labelled by ear on Sep 13 2026. Eleven days later 14
were gone, evicted by age, including the closest calls at 0.392 and 0.371. The
verdicts were still in labels.csv and the audio was not, which makes the pair
useless for the retrain the capture exists to enable.

The failure was silent in both directions: nothing logged a deletion, and the
directory sat at its cap looking exactly as healthy as it had the day before.
"""

import csv
import os

import wake_capture as wc


def _clip(tmp_path, name, mtime):
    p = tmp_path / name
    p.write_bytes(b"RIFF")
    os.utime(p, (mtime, mtime))
    return p


def _labels(tmp_path, *names):
    with open(tmp_path / wc.LABEL_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "score", "label"])
        w.writeheader()
        for n in names:
            w.writerow({"file": n, "score": "0.500", "label": "wake_phrase"})


def test_nothing_is_evicted_under_the_cap(tmp_path):
    _clip(tmp_path, "0.500_a.wav", 1000)
    assert wc.clips_to_evict(str(tmp_path), 300) == []


def test_the_unlabelled_clip_goes_first_even_when_it_is_newer(tmp_path):
    """The whole point. Age alone would have taken the labelled one."""
    _clip(tmp_path, "0.500_old_labelled.wav", 1000)
    _clip(tmp_path, "0.500_new_unlabelled.wav", 2000)
    _labels(tmp_path, "0.500_old_labelled.wav")

    evict = wc.clips_to_evict(str(tmp_path), 1)
    assert [os.path.basename(p) for p in evict] == ["0.500_new_unlabelled.wav"]


def test_oldest_first_within_the_unlabelled(tmp_path):
    for i, t in enumerate((3000, 1000, 2000)):
        _clip(tmp_path, f"0.50{i}_c.wav", t)
    evict = wc.clips_to_evict(str(tmp_path), 1)
    assert [os.path.basename(p) for p in evict] == ["0.501_c.wav", "0.502_c.wav"]


def test_capture_still_prunes_when_everything_is_labelled(tmp_path):
    """A preference, not a prohibition. "Never delete a labelled clip" would
    stop the directory recording anything new once it filled, while looking
    perfectly healthy at its cap."""
    names = ["0.500_a.wav", "0.500_b.wav"]
    _clip(tmp_path, names[0], 1000)
    _clip(tmp_path, names[1], 2000)
    _labels(tmp_path, *names)

    evict = wc.clips_to_evict(str(tmp_path), 1)
    assert [os.path.basename(p) for p in evict] == ["0.500_a.wav"]


def test_the_manifest_is_never_a_candidate(tmp_path):
    """It is not a clip, and it is the record that outlives them."""
    _clip(tmp_path, "0.500_a.wav", 1000)
    _labels(tmp_path, "0.500_a.wav")
    assert wc.clips_to_evict(str(tmp_path), 0) == [
        str(tmp_path / "0.500_a.wav")]


def test_a_missing_manifest_reads_as_nothing_labelled(tmp_path):
    """The normal case for a directory nobody has labelled yet."""
    assert wc.labelled_clips(str(tmp_path)) == set()


def test_a_malformed_manifest_does_not_take_the_wake_loop_down(tmp_path):
    (tmp_path / wc.LABEL_FILE).write_bytes(b"\x00\xff not csv at all")
    assert wc.labelled_clips(str(tmp_path)) == set()


def test_a_row_with_no_verdict_does_not_protect_a_clip(tmp_path):
    """label_wake.py writes a row per keypress, but a half written or hand
    edited manifest must not shield a clip nobody has actually judged."""
    with open(tmp_path / wc.LABEL_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "score", "label"])
        w.writeheader()
        w.writerow({"file": "0.500_a.wav", "score": "0.500", "label": ""})
    assert wc.labelled_clips(str(tmp_path)) == set()


def test_a_directory_that_is_not_there_evicts_nothing(tmp_path):
    assert wc.clips_to_evict(str(tmp_path / "nope"), 10) == []
