"""set_session_status as a MILES tool: a dry run runs, a real run waits for him."""

import pytest

import evallo_tools
import pending_action
from tools import Permission, permits, registry


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake(student, start, status, real, minutes=None):
        seen.append((student, start.isoformat(), status, real, minutes))
        return {"ok": True, "outcome": "done" if real else "dry_run_ok",
                "message": "real" if real else "dry"}
    monkeypatch.setattr(evallo_tools, "run_script", fake)
    monkeypatch.setattr(pending_action, "_pending", None)
    pending_action.begin_turn()
    return seen


def test_registered_as_external_write_for_hokage_only():
    spec = registry.get("set_session_status")
    assert spec.permission is Permission.EXTERNAL_WRITE
    assert permits(spec, "hokage")
    assert not permits(spec, "chunin")


def test_dry_run_runs_at_once_without_asking(calls):
    said = evallo_tools.set_session_status("Taylor", "2026-11-23T16:15", "Completed", True)
    assert said == "dry"
    assert calls == [("Taylor", "2026-11-23T16:15:00-05:00", "Completed", False, None)]


def test_real_run_only_asks(calls):
    said = evallo_tools.set_session_status("Taylor", "2026-11-23T16:15", "Completed", False, 90)
    assert calls == []
    assert "Mark Taylor's session on Monday November 23 at 4:15 PM Completed in Evallo?" in said


def test_real_run_cannot_be_confirmed_on_the_same_turn(calls):
    evallo_tools.set_session_status("Taylor", "2026-11-23T16:15", "Completed", False)
    assert "Refused" in pending_action.resolve(True)
    assert calls == []


def test_real_run_happens_on_his_yes_next_turn(calls):
    evallo_tools.set_session_status("Taylor", "2026-11-23T16:15", "No Show", False, 60)
    pending_action.begin_turn()
    assert pending_action.resolve(True) == "real"
    assert calls == [("Taylor", "2026-11-23T16:15:00-05:00", "No Show", True, 60)]


def test_his_no_runs_nothing(calls):
    evallo_tools.set_session_status("Taylor", "2026-11-23T16:15", "Completed", False)
    pending_action.begin_turn()
    assert pending_action.resolve(False).startswith("Cancelled")
    assert calls == []
