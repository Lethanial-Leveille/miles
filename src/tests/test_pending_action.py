"""Confirmation is enforced by turn order and time, not by the prompt."""

import pytest

import pending_action as pa


@pytest.fixture(autouse=True)
def fresh():
    pa._pending, pa._turn = None, 0
    yield
    pa._pending, pa._turn = None, 0


def _stage(runs, now=0.0):
    pa.propose("'gym' on Monday", lambda: runs.append("ran") or "Created 'gym'.", now=now)


def test_confirming_on_the_same_turn_is_refused_and_the_proposal_survives():
    runs = []
    pa.begin_turn()
    _stage(runs)
    assert pa.resolve(True, now=1.0).startswith("Refused")
    assert runs == []
    pa.begin_turn()
    assert pa.resolve(True, now=5.0) == "Created 'gym'."
    assert runs == ["ran"]


def test_an_answer_cannot_be_used_twice():
    runs = []
    pa.begin_turn(); _stage(runs)
    pa.begin_turn(); pa.resolve(True, now=5.0)
    assert pa.resolve(True, now=6.0).startswith("Nothing is waiting")
    assert runs == ["ran"]


def test_declining_runs_nothing():
    runs = []
    pa.begin_turn(); _stage(runs)
    pa.begin_turn()
    assert pa.resolve(False, now=5.0).startswith("Cancelled")
    assert runs == []


def test_only_the_very_next_turn_can_confirm():
    """Something unrelated in between drops the proposal."""
    runs = []
    pa.begin_turn(); _stage(runs)
    pa.begin_turn(); pa.begin_turn()
    assert "expired" in pa.resolve(True, now=5.0)
    assert runs == []


def test_a_slow_answer_expires():
    runs = []
    pa.begin_turn(); _stage(runs, now=0.0)
    pa.begin_turn()
    assert "expired" in pa.resolve(True, now=pa.CONFIRM_WINDOW_S + 1)
    assert runs == []


def test_a_new_proposal_replaces_the_old_one():
    first, second = [], []
    pa.begin_turn(); _stage(first)
    pa.begin_turn(); pa.propose("'gym' at eleven", lambda: second.append("ran"), now=1.0)
    pa.begin_turn(); pa.resolve(True, now=2.0)
    assert first == [] and second == ["ran"]
