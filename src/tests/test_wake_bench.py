"""The bench's counting must not flatter or punish an engine by accident.

Every number wake_bench.py reports comes out of two small functions: the
refractory window that turns frames into fires, and the cue matching that turns
fires into catches and false wakes. A bug in either skews every engine the same
way, so the comparison still looks plausible while the numbers are wrong.
"""

import importlib.util
import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "wake_bench.py")

_spec = importlib.util.spec_from_file_location("wake_bench", _PATH)
wb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wb)


class _Scripted:
    """Fires on the frame indexes it is given, so the test controls the timing."""

    frame_samples = 1600   # 0.1s at 16kHz, so frame i ends at (i + 1) / 10 seconds

    def __init__(self, firing):
        self.firing = set(firing)
        self.i = 0
        self.resets = 0

    def reset(self):
        self.i = 0
        self.resets += 1

    def fired(self, frame):
        hit = self.i in self.firing
        self.i += 1
        return hit


def _silence(seconds):
    return np.zeros(int(seconds * wb.RATE), dtype=np.int16)


def test_one_phrase_scoring_over_several_frames_counts_once():
    # A real "hey nova" holds the score up for a few frames in a row.
    engine = _Scripted([10, 11, 12, 13])
    assert wb.fire_times(engine, _silence(5)) == [1.1]


def test_a_second_wake_after_the_refractory_window_counts():
    # 1.1s, then 3.2s: past the two second window, a separate false wake.
    engine = _Scripted([10, 31])
    assert wb.fire_times(engine, _silence(5)) == [1.1, 3.2]


def test_a_fire_inside_the_window_is_absorbed():
    engine = _Scripted([10, 25])
    assert wb.fire_times(engine, _silence(5)) == [1.1]


def test_each_clip_starts_from_a_reset_engine():
    # Without the reset, state from the last clip could fire on this one.
    engine = _Scripted([])
    wb.fire_times(engine, _silence(1))
    wb.fire_times(engine, _silence(1))
    assert engine.resets == 2


def test_a_fire_shortly_after_the_cue_is_a_catch():
    assert wb.match_cues([5.8], [5.0]) == (1, 0)


def test_a_fire_far_from_any_cue_is_false():
    assert wb.match_cues([5.8, 20.0], [5.0]) == (1, 1)


def test_one_fire_cannot_catch_two_cues():
    # Two cues close together and one fire between them: one catch, one miss.
    assert wb.match_cues([6.0], [5.0, 6.5]) == (1, 0)


def test_a_missed_cue_is_not_false():
    assert wb.match_cues([], [5.0, 12.0]) == (0, 0)
