"""The level gate, which is the one guard here that can make her deaf.

The score cannot separate a real wake from the neighbours: on 49 labelled hits
the two overlap from 0.40 to 0.94, so no threshold exists. Level does separate
them, because speech through a wall arrives quiet and the model never looks at
loudness.

Every test that matters here is about failing open. A false wake costs a turn
and is visible in the journal; a wake silently dropped costs trust and is
visible nowhere, which is why the two are not symmetric.
"""

import numpy as np

import pytest


def _frames(value, count=32):
    return [np.full(1280, value, dtype=np.int16).tobytes() for _ in range(count)]


@pytest.fixture
def aud():
    """wake_gate, not audio: audio.py takes the mic lock at import, and this
    guard must not be one the suite skips while miles-voice is running."""
    import wake_gate
    return wake_gate


def test_peak_is_measured_in_dbfs(aud):
    assert aud.window_peak_dbfs(_frames(32767)) == pytest.approx(0.0, abs=0.01)
    assert aud.window_peak_dbfs(_frames(8192)) == pytest.approx(-12.04, abs=0.01)


def test_silence_has_no_level_rather_than_a_level_of_zero(aud):
    """None is not -inf and not 0. A clip with nothing in it cannot be judged,
    and the gate reads None as a reason to wake rather than a reason not to."""
    assert aud.window_peak_dbfs(_frames(0)) is None
    assert aud.window_peak_dbfs([]) is None


def test_a_quiet_window_is_rejected(aud):
    assert aud.too_quiet(-48.0, 32, 32) is True


def test_a_loud_window_is_kept(aud):
    assert aud.too_quiet(-20.0, 32, 32) is False


def test_the_floor_itself_wakes_her(aud):
    """Exclusive, like WAKE_THRESHOLD. A clip exactly on the line is not
    evidence against it."""
    assert aud.too_quiet(aud.WAKE_LEVEL_FLOOR, 32, 32) is False


def test_a_partial_window_fails_open(aud):
    """The first 2.5 seconds after a restart. Judging a real wake from a short
    buffer would trade a visible failure for an invisible one."""
    assert aud.too_quiet(-99.0, 5, 32) is False


def test_no_level_fails_open(aud):
    assert aud.too_quiet(None, 32, 32) is False


def test_the_gate_can_be_switched_off(aud, monkeypatch):
    """One flag restores the old behaviour, because a guard that can deafen her
    needs an off switch that does not require reasoning about a threshold."""
    monkeypatch.setattr(aud, "WAKE_LEVEL_GATE", False)
    assert aud.too_quiet(-99.0, 32, 32) is False


def test_the_two_far_field_wakes_that_set_the_floor_still_wake_her(aud):
    """Sep 25 2026: two genuine attempts from the next room measured -40.23
    dBFS. They are the reason the floor is -42 and not -40, and a change that
    silently starts rejecting them is the regression this file exists to
    catch."""
    assert aud.too_quiet(-40.23, 32, 32) is False


def test_the_quietest_labelled_false_wake_is_still_cut(aud):
    """-52.36 dBFS, the quietest of the 22 labelled false wakes."""
    assert aud.too_quiet(-52.36, 32, 32) is True


# ── a wake heard over her ──

def test_the_oct_5_false_barge_in_is_ignored(aud):
    """0.43 on near silence, 14.9s into a reply; its window peaked at -35.1."""
    assert aud.barge_in_too_quiet(-35.1) is True


def test_his_quietest_hey_nova_over_her_still_stops_her(aud):
    """Lowest of his 37 over her, wake_eval 20261001T114604: -22.5."""
    assert aud.barge_in_too_quiet(-22.5) is False


def test_a_barge_in_wake_with_no_level_still_stops_her(aud):
    assert aud.barge_in_too_quiet(None) is False


def test_the_watcher_judges_level_before_it_stops_her():
    import ast
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "audio.py").read_text()
    watcher = ast.unparse(next(n for n in ast.walk(ast.parse(source))
                               if isinstance(n, ast.FunctionDef)
                               and n.name == "watch_for_interrupt"))
    assert watcher.index("barge_in_too_quiet(") < watcher.index("heard = 'wake'")
    assert "wakegated_" in watcher, "an ignored wake must keep its audio"
