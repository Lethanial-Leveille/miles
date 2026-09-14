"""The executor enforces the tier it is handed, before any tool runs."""

from types import SimpleNamespace

import pytest

import brain
from tools import registry


@pytest.fixture
def ran(monkeypatch):
    """Records which tools actually executed, and runs none of them for real."""
    names = []
    monkeypatch.setattr(brain, "log_tool_call", lambda *a, **k: None)
    monkeypatch.setattr(registry, "call", lambda name, args: names.append(name) or "ran")
    return names


def _block(name, **inputs):
    return SimpleNamespace(name=name, id="toolu_test", input=inputs)


def test_a_guest_is_refused_before_the_tool_runs(ran):
    block = _block("create_calendar_event", summary="x",
                   start_time="tomorrow at 2pm", duration_minutes=30)
    [result] = brain._run_tools([block], "model", "genin")
    assert result["is_error"]
    assert result["output"].startswith("Refused")
    assert ran == []


def test_the_turn_tier_wins_over_the_database(ran, monkeypatch):
    """The database says hokage, the turn says jonin. The gate has to agree with
    the prompt, which was built for jonin."""
    monkeypatch.setattr(brain, "effective_tier", lambda *a, **k: "hokage")
    [result] = brain._run_tools([_block("get_oura_sleep")], "model", "jonin")
    assert result["is_error"]
    assert ran == []


def test_control_still_works_for_a_guest(ran):
    [result] = brain._run_tools([_block("dismiss")], "model", "genin")
    assert not result["is_error"]
    assert ran == ["dismiss"]


def test_hokage_runs_private_reads(ran):
    [result] = brain._run_tools([_block("get_oura_sleep")], "model", "hokage")
    assert not result["is_error"]
    assert ran == ["get_oura_sleep"]



def test_a_failed_tool_always_goes_back_to_nova():
    """Even one declared fire and forget. Otherwise the fallback says "Done."
    over something that did not happen."""
    quiet_failure = {"block": _block("cancel_reminder"), "output": "nothing matched", "is_error": True}
    quiet_success = {"block": _block("cancel_reminder"), "output": "Removed 1", "is_error": False}
    assert brain._needs_second_call([quiet_failure])
    assert not brain._needs_second_call([quiet_success])
    assert brain._needs_second_call([{"block": _block("get_weather"), "output": "{}", "is_error": False}])



def test_a_claimed_save_without_the_call_is_caught():
    assert brain._claims_a_save_without_calling("I've got that down. Twelve credits flat.", set())
    assert not brain._claims_a_save_without_calling("I've got that down.", {"remember"})
    assert not brain._claims_a_save_without_calling("Twelve credits is exactly full time.", set())



def test_a_proposal_is_spoken_by_code_not_rephrased():
    """Sep 14 2026: staged for the 21st, spoken as the 14th, created on the 21st.
    Guards the wiring, since a whole turn cannot run here without speaking."""
    import inspect
    source = inspect.getsource(brain.ask_nova_async)
    assert "staged = pending_action.words_for_turn()" in source
    assert "needs_second_call = _needs_second_call(results) and staged is None" in source
    assert "late_question = pending_action.words_for_turn()" in source



def test_an_unstaged_change_question_is_caught():
    assert brain._asks_an_unstaged_change("Rename Charley lesson on Wednesday at 4 PM to Charley lesson?", None)
    assert not brain._asks_an_unstaged_change("Rename Charlie to Charley?", "Rename Charlie to Charley?")
    assert not brain._asks_an_unstaged_change("How did you sleep?", None)
