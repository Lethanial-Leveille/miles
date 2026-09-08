"""A tool that only changes Nova's conversational state must not speak.

Observed live on Sep 6 2026. A false wake at 0.50 transcribed as "over here",
the model correctly called ignore, voice_main logged "Not addressed to Nova,
staying quiet", and Nova said "Done." into the room anyway. Every layer behaved
correctly and the word was spoken before any of them could matter.

The cause is that ignore and dismiss share returns_to_model=False with the
timer and reminder tools, so they landed in the branch carrying the fallback
that exists to stop a timer being set in silence. NOT_ADDRESSED_TO_YOU tells
Nova to say nothing at all when speech was not aimed at her, so correct
behaviour produced an empty spoken_parts and the fallback then filled it. The
better she behaved, the more certain she was to speak.

Permission.CONTROL is the distinction that separates them, and its docstring
already said so: it changes Nova's own conversational state rather than doing
work. Work needs confirming. A state transition does not.
"""

import pytest

import actions        # noqa: F401  registers ignore, dismiss, timers, reminders
from tools import Permission, registry


def test_ignore_and_dismiss_are_control():
    """The fix keys on permission, so it is only correct while these two carry
    CONTROL. A tool re registered as WRITE would start speaking again."""
    assert registry.get("ignore").permission is Permission.CONTROL
    assert registry.get("dismiss").permission is Permission.CONTROL


def test_the_tools_that_need_a_confirmation_are_not_control():
    """The other half. These must keep the fallback, because a timer set in
    total silence reads as the tool having failed when it worked."""
    for name in ("set_timer", "set_reminder", "cancel_reminder"):
        assert registry.get(name).permission is not Permission.CONTROL, name


def test_every_fire_and_forget_tool_is_classified():
    """A tool added with returns_to_model=False and no thought about permission
    inherits whichever branch it happens to fall into, silently."""
    for name in registry.names():
        tool = registry.get(name)
        if not tool.returns_to_model:
            assert tool.permission in set(Permission), name


# ── the branch itself ──

class _Block:
    def __init__(self, name):
        self.name = name


def _only_control(names):
    """The predicate as brain.py computes it, over a set of called tools."""
    results = [{"block": _Block(n)} for n in names]
    return all(
        registry.get(r["block"].name).permission is Permission.CONTROL
        for r in results
    )


def test_a_lone_ignore_suppresses_the_fallback():
    assert _only_control(["ignore"]) is True


def test_a_lone_dismiss_suppresses_the_fallback():
    """A dismissal where the model said nothing is silence, not "Done.".

    The farewell it should carry comes from the model's own text alongside the
    call, or from the phrase bank on the local path."""
    assert _only_control(["dismiss"]) is True


def test_a_timer_still_gets_its_confirmation():
    assert _only_control(["set_timer"]) is False


def test_a_mixed_turn_still_confirms():
    """A turn that both sets a timer and says goodbye did real work, so it owes
    an acknowledgement. all() over the called tools is what gets this right;
    any() would have silenced the timer."""
    assert _only_control(["set_timer", "dismiss"]) is False


def test_a_reminder_alongside_an_ignore_still_confirms():
    assert _only_control(["set_reminder", "ignore"]) is False
