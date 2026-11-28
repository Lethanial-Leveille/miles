"""Follow up review fixes for the request intent step. Offline, fake names only."""

import asyncio
import threading
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import brain
import pending_action as pa
import task_intent
import task_scheduling
from task_intent import Intent
from test_task_intent import Stream, active, block, turn  # noqa: F401  (turn is a fixture)
from test_voice_action_safety import voice  # noqa: F401  (a fixture)
from voice_action_safety import _intent_matches

WEDNESDAY = date(2026, 11, 25)


def _ran(monkeypatch, output="ok"):
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append((name, args)) or output)
    return ran


# ── F1 and F2: an external change is bound to what he named, one at a time ──

DELETE = Intent("other", "dentist", "", tool="delete_calendar_event")


def test_a_delete_for_a_different_event_than_he_named_is_refused(turn, monkeypatch):
    ran = _ran(monkeypatch)
    said = "delete the dentist appointment on monday"
    for args in ({"title": "dentist follow up", "day": "monday"},
                 {"title": "dentist", "day": "friday"},
                 {"title": "orthodontist", "day": "monday"}):
        result = asyncio.run(turn.run(said, [block("delete_calendar_event", args)], intent=DELETE))
        assert ran == [] and "Nothing was changed" in result.text, args
    asyncio.run(turn.run(said, [block("delete_calendar_event", {"title": "dentist", "day": "monday"})],
                         intent=DELETE))
    assert [name for name, _ in ran] == ["delete_calendar_event"]


def test_two_distinct_calls_to_the_authorized_external_tool_are_refused_together(turn, monkeypatch):
    ran = _ran(monkeypatch)
    result = asyncio.run(turn.run(
        "delete the dentist appointment on monday and tuesday",
        [block("delete_calendar_event", {"title": "dentist", "day": "monday"}, id="a"),
         block("delete_calendar_event", {"title": "dentist", "day": "tuesday"}, id="b")],
        intent=DELETE))
    assert ran == [] and "one change at a time" in result.text


# ── F4: a stored change is reported from its result, not the model's sentence ──

def test_a_timer_is_reported_from_its_result_not_the_draft(turn, monkeypatch):
    _ran(monkeypatch, "Timer set for 10 minutes.")
    result = asyncio.run(turn.run("set a timer for ten minutes",
                                  [block("set_timer", {"amount": 10, "unit": "minutes"})],
                                  draft="Okay, a twenty minute timer is running."))
    assert "10 minutes" in result.text and "twenty" not in result.text.casefold()
    assert not any("twenty" in str(line).casefold() for line in turn.audio)
    assert not any("twenty" in str(e[1]).casefold() for e in turn.events if e[0] == "delta")


# ── F11: the timer length and reminder words are bound ──

def test_a_timer_of_a_different_length_is_refused(turn, monkeypatch):
    ran = _ran(monkeypatch)
    for args in ({"amount": 10, "unit": "hours"}, {"amount": 100, "unit": "minutes"}):
        asyncio.run(turn.run("set a timer for ten minutes", [block("set_timer", args)]))
    assert ran == []
    asyncio.run(turn.run("set a timer for ten minutes", [block("set_timer", {"amount": 10, "unit": "minutes"})]))
    asyncio.run(turn.run("set a timer for an hour", [block("set_timer", {"amount": 60, "unit": "minutes"})]))
    assert [a for _, a in ran] == [{"amount": 10, "unit": "minutes"}, {"amount": 60, "unit": "minutes"}]


def test_a_timer_with_no_length_asks_instead_of_guessing(turn, monkeypatch):
    ran = _ran(monkeypatch)
    result = asyncio.run(turn.run("set a timer", [block("set_timer", {"amount": 5, "unit": "minutes"})]))
    assert ran == [] and "How long" in result.text


def test_a_reminder_must_carry_his_content_and_time(turn, monkeypatch):
    ran = _ran(monkeypatch)
    five = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0).isoformat()
    six = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0).isoformat()
    said = "remind me to call the bank at 5 pm"
    for args in ({"content": "take out the trash", "due": five}, {"content": "call the bank", "due": six},
                 {"content": "call the bank"}):
        asyncio.run(turn.run(said, [block("set_reminder", args)]))
    assert ran == []
    asyncio.run(turn.run(said, [block("set_reminder", {"content": "call the bank", "due": five})]))
    assert [name for name, _ in ran] == ["set_reminder"]


# ── F10: a list change is bound to the item and the list ──

def test_a_list_check_cannot_name_another_item_or_list(turn, monkeypatch):
    ran = _ran(monkeypatch)
    said = "check off eggs off my grocery list"
    for args in ({"item": "milk", "list_name": "grocery"}, {"item": "eggs", "list_name": "hardware"}):
        asyncio.run(turn.run(said, [block("list_check", args)]))
    assert ran == []
    asyncio.run(turn.run(said, [block("list_check", {"item": "eggs", "list_name": "grocery"})]))
    asyncio.run(turn.run("check off eggs", [block("list_check", {"item": "eggs"})]))
    assert len(ran) == 2


def test_a_list_add_to_a_named_list_must_name_it(turn, monkeypatch):
    ran = _ran(monkeypatch)
    asyncio.run(turn.run("add screws to my hardware list", [block("list_add", {"item": "screws"})]))
    assert ran == []
    asyncio.run(turn.run("add screws to my hardware list",
                         [block("list_add", {"item": "screws", "list_name": "hardware"})]))
    assert len(ran) == 1


# ── F6 and F7: completion needs an affirmative statement about today ──

@pytest.mark.parametrize("said", [
    "I didn't finish reading", "I have not finished reading", "did I finish reading?",
    "I'll finish reading later", "if I finish reading mark it", "never finished reading"])
def test_a_completion_is_not_recorded_from_a_negative_or_uncertain_statement(turn, db, said):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run(said, [block("task_complete", {"task": "reading"})],
                                  intent=Intent("task_complete", "reading")))
    assert not db.task_completed_on(task, db.task_day()) and "Nothing was changed" in result.text


@pytest.mark.parametrize("said, action, timing", [
    ("I finished reading yesterday", "task_complete", "yesterday"),
    ("I finished reading yesterday", "task_complete", ""),
    ("skip reading tomorrow", "task_skip_today", "tomorrow"),
    ("skip reading on friday", "task_skip_today", ""),
    ("stop reading starting tomorrow", "task_archive", "starting tomorrow")])
def test_another_day_is_refused_never_moved_to_today(turn, db, said, action, timing):
    task = db.add_task("reading", "daily")
    tool = {"task_complete": "task_complete", "task_skip_today": "task_skip_today",
            "task_archive": "archive_task"}[action]
    result = asyncio.run(turn.run(said, [block(tool, {"task": "reading"})],
                                  intent=Intent(action, "reading", timing)))
    assert "Nothing was changed" in result.text and "today" in result.text
    assert not db.task_completed_on(task, db.task_day()) and not db.task_skipped_on(task, db.task_day())
    assert db.get_task(task)["archived_at"] is None


def test_a_task_whose_title_names_a_day_is_not_mistaken_for_another_day(turn, db):
    task = db.add_task("monday review", "daily")
    asyncio.run(turn.run("I finished monday review", [block("task_complete", {"task": "monday review"})],
                         intent=Intent("task_complete", "monday review")))
    assert db.task_completed_on(task, db.task_day())


def test_the_validated_day_is_carried_to_the_write(db, monkeypatch):
    task = db.add_task("reading", "daily")
    verdict = task_intent.validate(Intent("task_skip_today", "reading"), "skip reading today", WEDNESDAY)
    assert verdict.kind == "act" and verdict.gate.plan == {"day": "2026-11-25"}
    import lists_tasks
    from tools import AUTHORIZED_DAY, AUTHORIZED_TASK, ToolNotice
    tokens = (AUTHORIZED_TASK.set(task), AUTHORIZED_DAY.set("2026-11-25"))  # the clock says another day
    try:
        with pytest.raises(ToolNotice, match="day changed"):
            lists_tasks.task_skip_today_tool("reading")
    finally:
        AUTHORIZED_DAY.reset(tokens[1])
        AUTHORIZED_TASK.reset(tokens[0])
    assert not db.task_skipped_on(task, "2026-11-25")


# ── F9: the permission floor comes before any lookup ──

def test_a_lower_tier_learns_nothing_about_tasks_or_their_titles(db):
    db.add_task("reading chapter one")
    db.add_task("reading chapter two")
    for tier in ("genin", "chunin"):
        for action in ("task_archive", "task_complete", "task_skip_today"):
            verdict = task_intent.validate(Intent(action, "reading"), "stop reading done", WEDNESDAY, tier=tier)
            assert verdict.kind == "refuse" and "chapter" not in verdict.text
            assert "clearance" in verdict.text and verdict.choice is None
    missing = task_intent.validate(Intent("task_archive", "zebra"), "stop zebra", WEDNESDAY, tier="genin")
    assert missing.text == task_intent.validate(
        Intent("task_archive", "reading"), "stop reading", WEDNESDAY, tier="genin").text


# ── F12: scope words are read with their negation and without the title ──

@pytest.mark.parametrize("said, recurrence, title, scope", [
    ("cancel reading today, not for good", "daily", "reading", "skip"),
    ("cancel reading today but not forever", "daily", "reading", "skip"),
    ("cancel reading today for good", "daily", "reading", "ask"),
    ("cancel stop smoking today", "daily", "stop smoking", "skip"),
    ("cancel delete old files", "daily", "delete old files", "ask"),
    ("cancel reading for good", "daily", "reading", "archive"),
    ("cancel bank call", None, "bank call", "archive")])
def test_cancel_scope_honours_negation_and_ignores_the_title(said, recurrence, title, scope):
    assert task_intent._cancel_scope(said, recurrence, title) == scope


# ── F13: any quantity before a time unit is a relative date ──

@pytest.mark.parametrize("phrase", ["water plants in thirteen days", "water plants in seventeen weeks",
                                    "water plants in 400 days"])
def test_an_unlisted_quantity_is_still_a_relative_date(phrase):
    plan = task_scheduling.resolve_task_schedule(f"add {phrase}", title=phrase, today=WEDNESDAY)
    assert plan.outcome != "supported"
    assert not _intent_matches("task_add", {"task": phrase}, f"add {phrase}")


# ── F14: a supported first day is carried through the voice path ──

def test_a_first_day_is_stored_and_today_is_null(db):
    said = "add reading every day starting Monday"
    verdict = task_intent.validate(Intent("task_add", "reading", "every day starting Monday"), said, WEDNESDAY)
    assert verdict.kind == "act"
    assert verdict.gate.permits("task_add", {"task": "reading", "recurrence": "daily", "start_date": "2026-11-30"})
    assert not verdict.gate.permits("task_add", {"task": "reading", "recurrence": "daily"})
    assert not verdict.gate.permits("task_add", {"task": "reading", "recurrence": "daily",
                                                 "start_date": "2026-12-01"})
    today = task_intent.validate(Intent("task_add", "reading", "every day starting today"),
                                 "add reading every day starting today", WEDNESDAY)
    assert today.kind == "act" and today.gate.permits("task_add", {"task": "reading", "recurrence": "daily"})
    import lists_tasks
    later = (date.fromisoformat(db.task_day()) + timedelta(days=30)).isoformat()
    lists_tasks.task_add_tool("reading", "daily", start_date=later)
    [row] = db.active_tasks_matching("reading")
    assert db.get_task(row[0])["start_date"] == later


def test_a_first_day_in_the_past_or_without_daily_is_refused_at_the_write(db):
    import lists_tasks
    from tools import ToolNotice
    for args in (("reading", "daily", "2000-01-01"), ("reading", None, "2099-01-05"),
                 ("reading", "daily", "not a date")):
        with pytest.raises(ToolNotice):
            lists_tasks.task_add_tool(*args)
    assert not db.active_tasks_matching("reading")


# ── F15: a skip or stop question is live only once it was delivered, to one tier ──

def test_an_undelivered_question_cannot_be_answered(turn, db, monkeypatch):
    task = db.add_task("reading", "daily")
    monkeypatch.setattr(pa, "present_choice", lambda: None)  # interrupted before he heard it
    asyncio.run(turn.run("cancel reading", intent=Intent("task_archive", "reading")))
    result = asyncio.run(turn.run("stop it", [block("archive_task", {"task": "reading"})],
                                  intent=Intent("task_archive", "reading")))
    assert db.get_task(task)["archived_at"] is None and "Nothing was changed" in result.text


def test_a_delivered_question_is_answered_only_by_the_same_tier(monkeypatch):
    monkeypatch.setattr(pa, "_conversations", {})
    scope = ("pi", "voice", None, None)
    for tier, expected in (("hokage", (7, "reading")), ("genin", None)):
        with pa.committed_turn(scope, "hokage", "cancel reading", threading.Event()):
            pa.ask_choice(7, "reading")
            pa.present_choice()
        with pa.committed_turn(scope, tier, "skip today", threading.Event()):
            assert pa.take_choice() == expected


def test_a_cancelled_turn_leaves_its_question_unanswerable(monkeypatch):
    monkeypatch.setattr(pa, "_conversations", {})
    scope = ("pi", "voice", None, None)
    cancelled = threading.Event()
    with pa.committed_turn(scope, "hokage", "cancel reading", cancelled):
        pa.ask_choice(7, "reading")
        cancelled.set()
        pa.present_choice()
    with pa.committed_turn(scope, "hokage", "skip today", threading.Event()):
        assert pa.take_choice() is None


# ── F17: the write goes to the validated row ──

def test_a_replacement_with_the_same_title_is_not_written_to(db):
    import lists_tasks
    from tools import AUTHORIZED_TASK, ToolNotice
    original = db.add_task("reading", "daily")
    db.archive_task(original)
    replacement = db.add_task("reading", "daily")
    token = AUTHORIZED_TASK.set(original)
    try:
        for tool in (lists_tasks.archive_task_tool, lists_tasks.task_complete_tool,
                     lists_tasks.task_skip_today_tool):
            with pytest.raises(ToolNotice, match="no longer active"):
                tool("reading")
    finally:
        AUTHORIZED_TASK.reset(token)
    assert db.get_task(replacement)["archived_at"] is None
    assert not db.task_completed_on(replacement, db.task_day())
    assert not db.task_skipped_on(replacement, db.task_day())


# ── F21: the intent record must be exactly one complete record ──

def _message(*payloads):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="record_intent", input=p)
                                    for p in payloads])


GOOD = {"action": "timer", "target": "", "timing": "ten minutes", "more_than_one_change": False}


def test_the_parser_requires_every_field_and_exactly_one_record():
    assert task_intent.parse_intent(_message(GOOD)) == Intent("timer", "", "ten minutes")
    assert task_intent.parse_intent(_message({"action": "timer"})) is None
    for missing in ("target", "timing", "more_than_one_change"):
        assert task_intent.parse_intent(_message({k: v for k, v in GOOD.items() if k != missing})) is None
    assert task_intent.parse_intent(_message(GOOD, GOOD)) is None
    assert task_intent.parse_intent(_message(GOOD, {**GOOD, "action": "none"})) is None
    assert task_intent.parse_intent(_message()) is None


# ── F5: a pending confirmation's model text is held until the result is known ──

def test_a_confirmation_stream_never_shows_a_success_draft(voice, monkeypatch):
    changes = []
    actual = brain.registry.call

    def call(name, args):
        if name == "create_calendar_event":
            return pa.propose("Create the synthetic event", lambda: changes.append(1) or "Event created.")
        return actual(name, args)

    monkeypatch.setattr(brain.registry, "call", call)
    propose = block("create_calendar_event", {
        "summary": "Synthetic", "start_time": "Tomorrow at two", "duration_minutes": 60})
    asyncio.run(voice.run("Add the synthetic event to my calendar", [propose]))
    voice.events.clear()
    # The model claims success, but the confirmation is not applied (nothing matches).
    result = asyncio.run(voice.run("Yes", [], draft="Done, I created it."))
    shown = [text for kind, text in voice.events if kind == "delta"]
    assert "Done, I created it." not in " ".join(shown)
    assert changes == [] and "created it" not in result.text
