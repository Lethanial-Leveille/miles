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
