from datetime import date, datetime

import pytest

from database import task_day
from task_scheduling import resolve_task_schedule

TODAY = date(2026, 11, 25)


@pytest.mark.parametrize(("text", "title", "starts", "first_day", "recurrence"), [
    ("Add reading", "reading", None, None, None),
    ("Add due diligence", "due diligence", None, None, None),
    ("Add a daily task to read starting Monday", "read", "Monday", "2026-11-30", "daily"),
    ("Add pickup from school daily starting Monday", "pickup from school", "Monday", "2026-11-30", "daily"),
    ("Add reading starting Monday every day", "reading", "Monday", "2026-11-30", "daily"),
    ("Add watch Friday Night Lights", "watch Friday Night Lights", None, None, None),
    ('Add "Deadline Friday"', "Deadline Friday", None, None, None),
    ("Add reading daily", "reading", None, None, "daily"),
    ("Add reading each day starting today", "reading", None, None, "daily"),
    ("Add reading as a daily task starts on Monday", "reading", "Monday", "2026-11-30", "daily"),
    ("Add reading every day from Monday", "reading", "from Monday", "2026-11-30", "daily"),
    ("Add reading daily starting tomorrow", "reading", "tomorrow", "2026-11-26", "daily"),
    ("Add reading daily starting next Monday", "reading", "next Monday", "2026-11-30", "daily"),
    ("Add reading daily starting this Mon", "reading", "this Mon", "2026-11-30", "daily"),
    ("Add reading daily starting Dec 1st", "reading", "Dec 1st", "2026-12-01", "daily"),
    ("Add reading daily starting 1 December", "reading", "1 December", "2026-12-01", "daily"),
    ("Add reading daily starting 1/3", "reading", "1/3", "2027-01-03", "daily"),
    ("Add reading daily starting 2027-11-25", "reading", "2027-11-25", "2027-11-25", "daily"),
])
def test_supported_phrases_store_only_the_verified_day(text, title, starts, first_day, recurrence):
    result = resolve_task_schedule(text, title=title, starts=starts, today=TODAY)
    assert (result.outcome, result.start_date, result.recurrence) == ("supported", first_day, recurrence)


@pytest.mark.parametrize(("text", "starts", "outcome"), [
    ("Add reading every day starting next week", None, "reject"),
    ("Add reading on Mondays and Wednesdays", None, "reject"),
    ("Add reading on Monday and Wednesday", None, "reject"),
    ("Add reading every other day", None, "reject"),
    ("Add reading twice a week", None, "reject"),
    ("Add reading weekdays only", None, "reject"),
    ("Add reading monthly", None, "reject"),
    ("Add reading daily starting Monday at five", "Monday", "reject"),
    ("Add reading daily starting February 30", "February 30", "reject"),
    ("Add reading daily starting 2027-11-26", "2027-11-26", "reject"),
    ("Add reading daily starting 2026-11-24", "2026-11-24", "reject"),
    ("Add reading daily starting February 29", "February 29", "reject"),
    ("Add reading daily starting Monday until Friday", "Monday", "reject"),
    ("Add reading daily due Friday", "Friday", "reject"),
    ("Add reading by Friday", "Friday", "reject"),
    ("Add reading daily starting Monday, due Friday", "Monday", "reject"),
    ("Add reading daily starting Monday, no later than Friday", "Monday", "reject"),
    ("Add reading daily starting Monday by end of week", "Monday", "reject"),
    ("Add reading starting Monday", "Monday", "reject"),
    ("Make reading start Monday", "Monday", "reject"),
    ("Add reading every day starting after Monday", "Monday", "clarify"),
    ("Add reading daily starting Monday or Tuesday", "Monday", "clarify"),
    ("Add reading daily starting on Monday and Tuesday", "Monday", "clarify"),
    ("Add reading daily starting next Friday", "next Friday", "clarify"),
    ("Add reading daily starting Wednesday", "Wednesday", "clarify"),
    ("Add reading daily starting November 20", "November 20", "clarify"),
    ("Add reading Friday", "Friday", "clarify"),
    ("Add reading in two weeks", None, "clarify"),
    ("Add reading daily from Monday", None, "clarify"),
    ("Add reading daily starting next Monday", "Monday", "clarify"),
    ("Add reading daily starting Monday", "2026-11-30", "clarify"),
    ("Tuesday", "Tuesday", "clarify"),
    ("Yes", None, "clarify"),
    (None, None, "clarify"),
])
def test_unsupported_or_ambiguous_phrases_never_return_a_default_plan(text, starts, outcome):
    result = resolve_task_schedule(text, title="reading", recurrence="daily", starts=starts, today=TODAY)
    assert result.outcome == outcome
    assert result.reason and result.start_date is None and result.recurrence is None


def test_unparsed_title_or_model_fields_cannot_authorize_untimed_task():
    for text, title, recurrence in [
        ("Add reading hold off until conditions change", "reading hold off until conditions change", None),
        ("Add reading Friday", "reading Friday", None),
        ("Add reading on the weekend", "reading on the weekend", None),
        ("Add reading within two weeks", "reading within two weeks", None),
        ("Add reading once we get back", "reading once we get back", None),
        ("Add reading when you get back", "reading", None),
        ("Add watch Friday Night Lights Due Friday", "watch Friday Night Lights Due Friday", None),
        ("Add reading", "reading", "daily"),
    ]:
        assert resolve_task_schedule(text, title=title, recurrence=recurrence, today=TODAY).outcome != "supported"


def test_resolver_uses_the_captured_task_day_instead_of_wall_date():
    text = "Add reading daily starting tomorrow"
    before = date.fromisoformat(task_day(datetime(2026, 11, 26, 3, 59)))
    after = date.fromisoformat(task_day(datetime(2026, 11, 26, 4)))
    assert resolve_task_schedule(text, title="reading", starts="tomorrow", today=before).start_date == "2026-11-26"
    assert resolve_task_schedule(text, title="reading", starts="tomorrow", today=after).start_date == "2026-11-27"


@pytest.mark.parametrize("title", ["call the bank at half past five", "water plants on the first of the month"])
def test_unsupported_clause_cannot_be_accepted_as_a_matching_title(title):
    plan = resolve_task_schedule(f"Add {title}", title=title, today=date(2026, 11, 25))
    assert plan.outcome == "reject"


def test_media_title_can_be_lowercase_in_a_transcript():
    title = "watch friday night lights"
    plan = resolve_task_schedule(f"Add {title}", title=title, today=date(2026, 11, 25))
    assert plan.outcome == "supported" and plan.recurrence is None
