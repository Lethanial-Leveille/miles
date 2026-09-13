"""Labelling must survive the directories being pruned underneath it.

Both capture directories are ring buffers: they fill to a cap and the oldest
files are evicted. A labelling tool that keyed on position rather than filename
would, after any eviction, silently attach an old verdict to a different clip.
That does not crash and it does not look wrong, which is the same failure
label_speakers.py warns about one level up.
"""

import csv
import importlib.util
import math
import os
import wave

import numpy as np

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


# ── playback normalization ──
# The tool was unusable on its first run for a reason nothing here would have
# caught: the clips are 30 to 38 dB below normal speech, so aplay played them
# perfectly and the room heard nothing. Judging a clip you cannot hear is not
# labelling, it is guessing.


def _wav(tmp_path, name, peak, seconds=0.5, rate=16000):
    """A tone at a chosen peak amplitude, so gain is checkable in dB."""
    path = str(tmp_path / name)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * peak * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


def test_a_quiet_clip_is_boosted_to_the_target(tmp_path):
    """A real near miss peaks near -40 dBFS against -10 for close speech."""
    path = _wav(tmp_path, "quiet.wav", 10 ** (-30 / 20.0))
    out, gain_db, peak_db = lw._normalized_copy(path)

    assert peak_db == pytest.approx(-30, abs=0.5)
    assert gain_db == pytest.approx(27, abs=0.5)
    assert lw._levels(out)[0] == pytest.approx(lw.PLAYBACK_TARGET_DBFS, abs=0.5)
    os.unlink(out)


def test_the_boost_is_capped(tmp_path):
    """An empty room sits near -70 dBFS. Without a ceiling it would arrive as a
    wall of amplified noise floor, which is worse than silence because it is
    loud and still carries nothing."""
    path = _wav(tmp_path, "silent.wav", 10 ** (-70 / 20.0))
    out, gain_db, _ = lw._normalized_copy(path)

    assert gain_db == pytest.approx(lw.PLAYBACK_MAX_GAIN_DB)
    os.unlink(out)


def test_a_loud_clip_is_played_untouched(tmp_path):
    """No temp file, no copy, and the caller must not try to delete the
    original. The return value is the original path itself."""
    path = _wav(tmp_path, "loud.wav", 0.9)
    out, gain_db, _ = lw._normalized_copy(path)

    assert out == path
    assert gain_db == 0.0


def test_the_original_clip_is_never_modified(tmp_path):
    """The capture directories are ring buffers holding the only record of what
    the model actually heard. Boosting is for the ear, and writing the boost
    back would destroy the evidence the score was computed from."""
    path = _wav(tmp_path, "quiet.wav", 10 ** (-30 / 20.0))
    before = open(path, 'rb').read()

    out, _, _ = lw._normalized_copy(path)
    assert open(path, 'rb').read() == before
    os.unlink(out)


def test_boosting_does_not_clip_into_distortion(tmp_path):
    path = _wav(tmp_path, "quiet.wav", 10 ** (-30 / 20.0))
    out, _, _ = lw._normalized_copy(path)

    with wave.open(out, 'rb') as wf:
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert np.abs(samples).max() < 32767
    os.unlink(out)


def test_an_empty_clip_does_not_raise(tmp_path):
    """A truncated write leaves a header and no frames. A diagnostic tool that
    crashes on its own corrupt input is worse than one that skips it."""
    path = str(tmp_path / "empty.wav")
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
    out, gain_db, peak_db = lw._normalized_copy(path)
    assert out == path and gain_db == 0.0 and peak_db is None


def test_a_failed_play_is_reported_not_swallowed(tmp_path, capsys, monkeypatch):
    """The defect that made the first run undiagnosable.

    capture_output hid aplay's stderr, so a genuine device failure looked
    exactly like a clip too quiet to hear. Those need opposite responses."""
    class _Fail:
        returncode = 1
        stderr = "aplay: main:831: audio open error: No such file or directory"

    monkeypatch.setattr(lw.subprocess, "run", lambda *a, **k: _Fail())
    assert lw._play("x.wav", "plughw:9,9") is False
    assert "audio open error" in capsys.readouterr().out
