"""The production tool set, as registered.

These assert on tools.registry rather than a throwaway one, because the point
is what Nova actually has. If a tool stops being registered, the capability
block silently loses a line and the prompt stops claiming it, with no error
anywhere. This is the test that notices.
"""

import pytest

import actions        # noqa: F401  registers the action tools
import memory_tool    # noqa: F401  registers remember
import system_state   # noqa: F401  registers get_system_state
import tier_tool      # noqa: F401  registers lower_access
import calendar_tools # noqa: F401  registers the calendar tools
import oura_tools     # noqa: F401  registers the Oura tools
import pending_action # noqa: F401  registers confirm_pending_action
from tools import Permission, permits, registry

EXPECTED = {
    "get_weather":      (Permission.READ,    True),
    "get_system_state": (Permission.READ,    True),
    "set_timer":        (Permission.WRITE,   False),
    "set_reminder":     (Permission.WRITE,   False),
    "cancel_reminder":  (Permission.WRITE,   False),
    "remember":         (Permission.WRITE,   True),
    "dismiss":          (Permission.CONTROL, False),
    "ignore":           (Permission.CONTROL, False),
    "lower_access":     (Permission.WRITE,   True),
    "get_upcoming_events":     (Permission.READ,           True),
    "check_calendar_freebusy": (Permission.READ,           True),
    "create_calendar_event":   (Permission.EXTERNAL_WRITE, True),
    "confirm_pending_action":  (Permission.EXTERNAL_WRITE, True),
    "update_calendar_event":   (Permission.EXTERNAL_WRITE, True),
    "list_pending_memories":   (Permission.READ,           True),
    "find_schedule_conflicts": (Permission.READ,           True),
    "plan_sessions":           (Permission.EXTERNAL_WRITE, True),
    "rename_calendar_events":  (Permission.EXTERNAL_WRITE, True),
    "undo_last_change":        (Permission.EXTERNAL_WRITE, True),
    "review_pending_memory":   (Permission.WRITE,          True),
    "delete_calendar_event":   (Permission.EXTERNAL_WRITE, True),
    "get_oura_readiness":      (Permission.READ,           True),
    "get_oura_sleep":          (Permission.READ,           True),
    "get_oura_heartrate":      (Permission.READ,           True),
    "get_oura_activity":       (Permission.READ,           True),
}

# READ in kind, private in content, or writes beyond this Pi. None of these
# may reach anyone below hokage, whatever their category default says.
HOKAGE_ONLY = {
    "lower_access", "get_upcoming_events", "check_calendar_freebusy",
    "create_calendar_event", "confirm_pending_action", "get_oura_readiness",
    "get_oura_sleep", "get_oura_heartrate", "get_oura_activity",
    "update_calendar_event", "delete_calendar_event",
    "list_pending_memories", "review_pending_memory",
    "find_schedule_conflicts", "plan_sessions", "rename_calendar_events",
    "undo_last_change",
}

# WRITE tools that legitimately cost a second round trip, with the reason.
# The default remains that a write just happens and is not spoken about; this
# is the list of cases where that default is wrong, kept short on purpose.
ROUND_TRIP_WRITES = {
    # Nothing to report, but without the round trip Nova often said nothing at
    # all, and "my last day is September 25" was answered "Done." (Sep 15 2026).
    "remember",
    # It can refuse: an escalation dressed as a demotion, a name that matches
    # nobody, someone already at the floor. A security control that fails
    # silently is worse than one that costs a second, and Nova cannot announce
    # the outcome before the call because she does not know it yet.
    "lower_access",
    # Changes made at once, read back in code with the time the code resolved.
    # The follow up is where Nova learns what happened, or why it could not.
    "create_calendar_event",
    "update_calendar_event",
    "delete_calendar_event",
    # The write itself can fail at Google, and a confirmation nobody hears
    # sounds exactly like one that did not happen.
    "confirm_pending_action",
    # He has to hear whether it was kept or discarded, and an id that was
    # not waiting for review is refused out loud rather than silently.
    "review_pending_memory",
    # A proposal of a whole plan; the question it returns is the point.
    "plan_sessions",
    # A proposal to rename many events; one question for all of them.
    "rename_calendar_events",
    # What was undone is read back word for word.
    "undo_last_change",
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
        if spec.returns_to_model and name not in ROUND_TRIP_WRITES:
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


# ── pluralization ──

@pytest.mark.parametrize("amount,unit,expected", [
    (1, "minutes", "minute"),
    (1, "hours", "hour"),
    (1, "seconds", "second"),
    (2, "minutes", "minutes"),
    (10, "minutes", "minutes"),
])
def test_units_singularize_at_one(amount, unit, expected):
    """The tool enum only offers plural units, so every one minute timer used
    to announce "your 1 minutes timer is up"."""
    assert actions._plural(amount, unit) == expected


def test_one_minute_timer_reads_correctly(db, monkeypatch):
    # A timer is a row now, so it has to be written somewhere throwaway.
    monkeypatch.setattr(actions, "DB_PATH", db.DB_PATH)
    assert actions.set_timer("1 minutes") == "Timer set for 1 minute (60 seconds)."
    assert actions.set_timer("5 minutes") == "Timer set for 5 minutes (300 seconds)."


@pytest.mark.parametrize("name", sorted(HOKAGE_ONLY))
def test_private_tools_are_hokage_only(name):
    spec = registry.get(name)
    assert not permits(spec, "jonin")
    assert permits(spec, "hokage")



def test_cancelling_a_reminder_that_does_not_exist_is_a_failure(monkeypatch):
    """Sep 13 2026: "cancel all of the tutoring sessions" matched no reminders
    three times over, and Nova said "Done."."""
    monkeypatch.setattr(actions, "cancel_reminder",
                        lambda c: f"No active reminders found matching '{c}'.")
    with pytest.raises(LookupError):
        registry.call("cancel_reminder", {"content": "Isaiah lesson"})
