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


def test_several_proposals_on_one_turn_are_confirmed_together():
    """Sep 13 2026: seven lessons staged in one turn replaced each other, Nova
    asked about the first, and his yes created the seventh."""
    runs = []
    pa.begin_turn()
    pa.propose("Add Isaiah on Tuesday at 3:30 PM", lambda: runs.append("isaiah") or "Added Isaiah.", now=0.0)
    question = pa.propose("Add Andrew on Thursday at 4 PM", lambda: runs.append("andrew") or "Added Andrew.", now=0.0)
    assert question == "Add Isaiah on Tuesday at 3:30 PM, and add Andrew on Thursday at 4 PM?"
    pa.begin_turn()
    assert pa.resolve(True, now=1.0) == "Added Isaiah. Added Andrew."
    assert runs == ["isaiah", "andrew"]


def test_declining_a_batch_runs_none_of_it():
    runs = []
    pa.begin_turn()
    for name in ("A", "B", "C"):
        pa.propose(f"Add {name}", lambda n=name: runs.append(n), now=0.0)
    pa.begin_turn()
    assert pa.resolve(False, now=1.0) == "Cancelled. Nothing was done: Add A, add B, and add C."
    assert runs == []


def test_one_failure_in_a_batch_is_reported_beside_what_worked():
    def broken():
        raise RuntimeError("Google said no")
    pa.begin_turn()
    pa.propose("Add Isaiah", lambda: "Added Isaiah.", now=0.0)
    pa.propose("Add Andrew", broken, now=0.0)
    pa.begin_turn()
    reply = pa.resolve(True, now=1.0)
    assert reply.startswith("Added Isaiah.")
    assert "Failed, not done: Add Andrew (Google said no)." in reply
