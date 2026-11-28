"""Refusals and questions are plain sentences, and true to what voice supports.
Offline, fake names."""

from datetime import date

import pytest

import task_intent
from task_intent import Intent
from task_scheduling import resolve_task_schedule
from voice_action_safety import NO_ACTION, NOT_ASKED, ONE_CHANGE, REFUSED

TODAY = date(2026, 11, 25)
JARGON = ("restate", "intended role", "includes a time or a date", "match the timing", "validate",
          "recurrence", "unsupported", "eligib", "clearance", "confirmation to apply")


def _said(verdict):
    assert verdict.kind in ("refuse", "ask")
    assert verdict.text.startswith("Nothing was changed.")
    lowered = verdict.text.casefold()
    assert not [word for word in JARGON if word in lowered], verdict.text
    return verdict.text


def test_unsupported_timing_asks_when_and_offers_only_what_voice_can_do(db):
    text = _said(task_intent.validate(Intent("task_add", "call the bank", "in four days"),
                                      "add call the bank in four days", TODAY))
    assert text == ("Nothing was changed. I can't set that timing by voice. When do you want that? "
                    "I can add it for today, or as a daily task starting today or on a day you name.")
    # The same question when the timing words were left in the title.
    other = _said(task_intent.validate(Intent("task_add", "call the bank in four days"),
                                       "add call the bank in four days", TODAY))
    assert other == text.replace("set that timing", "set 'in four days'")


@pytest.mark.parametrize("said, target, timing", [
    ("add call the bank at dawn", "call the bank", "at dawn"),
    ("add review next week", "review", ""),
    ("add reading every Tuesday", "reading", "every Tuesday"),
    ("add reading every day by Friday", "reading", "every day"),
    ("add reading every day at 5 pm", "reading", "every day"),
    ("add reading every day starting nonsense", "reading", "every day starting nonsense"),
    ("add reading every day starting Monday and Tuesday", "reading", "every day starting Monday and Tuesday"),
    ("add reading starting Monday", "reading", "starting Monday"),
    ("add reading every day starting 2020-01-01", "reading", "every day starting 2020-01-01"),
])
def test_every_scheduling_refusal_is_plain(db, said, target, timing):
    _said(task_intent.validate(Intent("task_add", target, timing), said, TODAY))


@pytest.mark.parametrize("utterance, title", [
    ("make reading every day", "reading"), ("add", "reading"), ('add "reading every day', "reading"),
    ("add reading every day every day", "reading"), ("add call the bank at noon", "call the bank"),
    ("add reading due Friday", "reading"), ("add reading by Friday", "reading"),
])
def test_scheduler_reasons_are_plain(utterance, title):
    result = resolve_task_schedule(utterance, title=title, today=TODAY)
    assert result.outcome != "supported"
    assert not [word for word in JARGON if word in result.reason.casefold()], result.reason


def test_a_title_that_may_hold_a_time_says_what_to_do(db):
    text = _said(task_intent.validate(Intent("task_add", "reading every day"), "add reading", TODAY))
    assert "task name" in text and "Say just the task" in text


def test_the_fixed_refusals_are_plain():
    for text in (REFUSED, NO_ACTION, NOT_ASKED, ONE_CHANGE):
        assert text.startswith("Nothing was changed.")
        assert not [word for word in JARGON if word in text.casefold()]
