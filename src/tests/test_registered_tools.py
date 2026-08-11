"""The production tool set, as registered.

These assert on tools.registry rather than a throwaway one, because the point
is what Nova actually has. If a tool stops being registered, the capability
block silently loses a line and the prompt stops claiming it, with no error
anywhere. This is the test that notices.
"""

import pytest

import actions        # noqa: F401  registers the action tools
import system_state   # noqa: F401  registers get_system_state
from tools import Permission, registry

EXPECTED = {
    "get_weather":      (Permission.READ,    True),
    "get_system_state": (Permission.READ,    True),
    "set_timer":        (Permission.WRITE,   False),
    "set_reminder":     (Permission.WRITE,   False),
    "cancel_reminder":  (Permission.WRITE,   False),
    "dismiss":          (Permission.CONTROL, False),
}


def test_every_expected_tool_is_registered():
    assert set(registry.names()) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_permission_and_round_trip_flags(name):
    permission, returns = EXPECTED[name]
    spec = registry.get(name)
    assert spec.permission is permission
    assert spec.returns_to_model is returns


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_tool_says_when_to_call_it(name):
    """Under description is the common failure mode. A description that only
    says what a tool does, without saying when to reach for it, measurably
    lowers how often the model calls it."""
    d = registry.get(name).description.lower()
    assert "call this" in d or "call it" in d


def test_only_read_tools_make_a_second_call():
    """returns_to_model replaced a hardcoded whitelist of one action type. The
    invariant it encodes: work that produces data to talk about costs a second
    round trip, work that just happens does not."""
    for name in registry.names():
        spec = registry.get(name)
        if spec.returns_to_model:
            assert spec.permission is Permission.READ, name


def test_schemas_are_sorted_and_api_shaped():
    schemas = registry.api_schemas()
    assert [s["name"] for s in schemas] == sorted(EXPECTED)
    for s in schemas:
        assert set(s) == {"name", "description", "input_schema"}


def test_capability_block_lists_every_tool():
    prose = registry.capability_prose()
    assert prose.count("\n- ") == len(EXPECTED)
    assert "only tools you have" in prose


# ── dismiss ──

def test_dismiss_executes_nothing():
    """It is a tool for the syntax and the audit trail, not for the work. The
    state transition happens in brain.py, which reads the call and exits the
    follow up loop."""
    assert registry.call("dismiss", {}) == "dismissed"


def test_dismiss_takes_no_arguments():
    assert registry.get("dismiss").input_schema["properties"] == {}


# ── timers ──

def test_timer_schema_is_structured_not_a_free_string():
    """A structured amount and unit means the word number fallback inside
    set_timer can never be reached from the tool path."""
    schema = registry.get("set_timer").input_schema
    assert schema["properties"]["unit"]["enum"] == ["seconds", "minutes", "hours"]
    assert sorted(schema["required"]) == ["amount", "unit"]


def test_timer_tool_starts_a_timer(monkeypatch):
    seen = {}
    monkeypatch.setattr(actions, "set_timer", lambda d: seen.setdefault("duration", d))
    registry.call("set_timer", {"amount": 10, "unit": "minutes"})
    assert seen["duration"] == "10 minutes"


# ── reminders ──

def test_reminder_due_is_optional():
    schema = registry.get("set_reminder").input_schema
    assert schema["required"] == ["content"]
    assert "due" in schema["properties"]


def test_reminder_description_points_at_the_supplied_clock():
    """Regression guard for 5ad97de from the tool side. If the description ever
    stops saying where the date comes from, Nova starts inventing one again."""
    d = registry.get("set_reminder").description.lower()
    assert "clock" in d and "past" in d


def test_reminder_tool_passes_content_and_due(monkeypatch):
    seen = {}
    monkeypatch.setattr(actions, "set_reminder",
                        lambda c, d=None: seen.update(content=c, due=d))
    registry.call("set_reminder", {"content": "push code", "due": "2026-08-11T21:00:00"})
    assert seen == {"content": "push code", "due": "2026-08-11T21:00:00"}


def test_cancel_reminder_tool_passes_content(monkeypatch):
    seen = {}
    monkeypatch.setattr(actions, "cancel_reminder", lambda c: seen.setdefault("content", c))
    registry.call("cancel_reminder", {"content": "push code"})
    assert seen["content"] == "push code"
