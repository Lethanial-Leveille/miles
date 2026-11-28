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


# ── F1 and F2: the intent step has no "other change" action ──

def test_an_other_change_action_is_not_part_of_the_schema_or_the_parser():
    assert "other" not in task_intent.ACTIONS
    assert "other" not in task_intent.INTENT_TOOL["input_schema"]["properties"]["action"]["enum"]
    assert "tool" not in task_intent.INTENT_TOOL["input_schema"]["properties"]
    assert task_intent.parse_intent(_message({"action": "other", "target": "dentist", "timing": "",
                                              "more_than_one_change": False})) is None


@pytest.mark.parametrize("name", ["delete_calendar_event", "create_calendar_event", "remember"])
def test_an_other_action_authorizes_no_tool_through_the_intent_step(name):
    verdict = task_intent.validate(Intent("other", "dentist"), "delete the dentist appointment on monday",
                                   WEDNESDAY)
    assert verdict.kind == "refuse" and verdict.gate is None
    assert name not in task_intent.COVERED_TOOLS


def test_a_calendar_delete_on_a_flagged_turn_keeps_its_own_confirm_first_flow(turn, monkeypatch):
    # The model called the turn no_change, so the intent step offers no covered tool, but the
    # calendar tool is still there and still only proposes; his yes is what would run it.
    proposals = []
    actual = brain.registry.call

    def call(name, args):
        if name == "delete_calendar_event":
            return pa.propose("Delete the dentist appointment", lambda: proposals.append(1) or "Deleted.")
        return actual(name, args)

    monkeypatch.setattr(brain.registry, "call", call)
    said = "delete the dentist appointment on monday"
    result = asyncio.run(turn.run(said, [block("delete_calendar_event", {"title": "dentist", "day": "monday"})],
                                  intent=Intent("no_change")))
    offered = {schema["name"] for schema in turn.offered[0]}
    assert "delete_calendar_event" in offered and "remember" in offered
    assert not offered & {"task_add", "task_complete", "archive_task", "task_skip_today", "list_add",
                          "list_check", "set_timer", "set_reminder"}
    assert proposals == [] and "Delete the dentist appointment" in result.text


def test_a_covered_tool_is_refused_on_a_no_change_turn(turn, monkeypatch):
    ran = _ran(monkeypatch)
    result = asyncio.run(turn.run("what is on my list", [block("set_timer", {"amount": 5, "unit": "minutes"})],
                                  intent=Intent("no_change")))
    assert ran == [] and "Nothing was changed" in result.text


def test_two_distinct_calls_to_one_covered_tool_are_refused_together(turn, monkeypatch):
    ran = _ran(monkeypatch)
    result = asyncio.run(turn.run(
        "add eggs to my grocery list",
        [block("list_add", {"item": "eggs", "list_name": "grocery"}, id="a"),
         block("list_add", {"item": "milk", "list_name": "grocery"}, id="b")]))
    assert ran == [] and "Nothing was changed" in result.text


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


# ── F10: a list change is bound to the item, the list and the row ──

def _item(db, name, text):
    return db.add_list_item(db.get_or_create_list(name), text)[0]


def test_a_list_check_cannot_name_another_item_or_list(turn, db, monkeypatch):
    ran = _ran(monkeypatch)
    _item(db, "grocery", "eggs")
    said = "check off eggs off my grocery list"
    for args in ({"item": "milk", "list_name": "grocery"}, {"item": "eggs", "list_name": "hardware"}):
        asyncio.run(turn.run(said, [block("list_check", args)]))
    assert ran == []
    asyncio.run(turn.run(said, [block("list_check", {"item": "eggs", "list_name": "grocery"})]))
    asyncio.run(turn.run("check off eggs", [block("list_check", {"item": "eggs"})]))
    assert len(ran) == 2


def test_a_list_check_writes_the_validated_row_not_the_words_matched_later(turn, db):
    first = _item(db, "grocery", "eggs")
    asyncio.run(turn.run("check off eggs off my grocery list",
                         [block("list_check", {"item": "eggs", "list_name": "grocery"})]))
    assert db.get_list_item(first)[2] == 1


def test_the_row_is_the_one_validated_even_when_a_lookalike_arrives_first(db):
    import lists_tasks
    from tools import AUTHORIZED_ITEM
    wanted = _item(db, "grocery", "eggs")
    verdict = task_intent.validate(Intent("list_check", "eggs"), "check off eggs", WEDNESDAY)
    assert verdict.gate.plan == {"item_id": wanted}
    db.check_list_item(wanted)  # a phone checked it first
    other = _item(db, "grocery", "free range eggs")
    token = AUTHORIZED_ITEM.set(wanted)
    try:
        from tools import ToolNotice
        with pytest.raises(ToolNotice, match="already checked"):
            lists_tasks.list_check_tool("eggs")
    finally:
        AUTHORIZED_ITEM.reset(token)
    assert db.get_list_item(other)[2] == 0


def test_two_items_that_match_ask_which_and_check_nothing(turn, db):
    one, two = _item(db, "grocery", "free range eggs"), _item(db, "grocery", "duck eggs")
    result = asyncio.run(turn.run("check off eggs", [block("list_check", {"item": "eggs"})]))
    assert "Which one" in result.text and "Nothing was changed" in result.text
    assert db.get_list_item(one)[2] == 0 and db.get_list_item(two)[2] == 0


def test_a_list_named_with_several_words_is_bound_whole(turn, db, monkeypatch):
    ran = _ran(monkeypatch)
    said = "add screws to my hardware store list"
    for args in ({"item": "screws", "list_name": "hardware"}, {"item": "screws", "list_name": "grocery"}):
        asyncio.run(turn.run(said, [block("list_add", args)], intent=Intent("list_add", "screws")))
    assert ran == []
    asyncio.run(turn.run(said, [block("list_add", {"item": "screws", "list_name": "hardware store"})],
                         intent=Intent("list_add", "screws")))
    assert len(ran) == 1


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


def test_the_validated_day_is_carried_to_the_write(db):
    task = db.add_task("reading", "daily")
    verdict = task_intent.validate(Intent("task_skip_today", "reading"), "skip reading today", WEDNESDAY)
    assert verdict.kind == "act" and verdict.gate.plan == {"day": "2026-11-25"}
    import lists_tasks
    from tools import AUTHORIZED_DAY, AUTHORIZED_TASK
    tokens = (AUTHORIZED_TASK.set(task), AUTHORIZED_DAY.set("2026-11-25"))
    try:
        lists_tasks.task_skip_today_tool("reading")  # the clock may say another day by now
    finally:
        AUTHORIZED_DAY.reset(tokens[1])
        AUTHORIZED_TASK.reset(tokens[0])
    assert db.task_skipped_on(task, "2026-11-25")
    assert not db.task_skipped_on(task, db.task_day()) or db.task_day() == "2026-11-25"


def test_a_completion_across_the_rollover_lands_on_the_validated_day(turn, db, monkeypatch):
    # He spoke on the 25th. By the time the write runs the clock says the 26th (past 04:00).
    task = db.add_task("reading", "daily")
    verdict = task_intent.validate(Intent("task_complete", "reading"), "I finished reading", WEDNESDAY)
    assert verdict.gate.plan["day"] == "2026-11-25"
    import lists_tasks
    from tools import AUTHORIZED_DAY, AUTHORIZED_TASK
    monkeypatch.setattr(lists_tasks, "task_day", lambda now=None: "2026-11-26")
    tokens = (AUTHORIZED_TASK.set(task), AUTHORIZED_DAY.set(verdict.gate.plan["day"]))
    try:
        lists_tasks.task_complete_tool("reading")
    finally:
        AUTHORIZED_DAY.reset(tokens[1])
        AUTHORIZED_TASK.reset(tokens[0])
    assert db.task_completed_on(task, "2026-11-25") and not db.task_completed_on(task, "2026-11-26")


def test_a_skip_across_the_rollover_lands_on_the_validated_day(db, monkeypatch):
    task = db.add_task("reading", "daily")
    import lists_tasks
    from tools import AUTHORIZED_DAY, AUTHORIZED_TASK
    monkeypatch.setattr(lists_tasks, "task_day", lambda now=None: "2026-11-26")
    tokens = (AUTHORIZED_TASK.set(task), AUTHORIZED_DAY.set("2026-11-25"))
    try:
        lists_tasks.task_skip_today_tool("reading")
    finally:
        AUTHORIZED_DAY.reset(tokens[1])
        AUTHORIZED_TASK.reset(tokens[0])
    assert db.task_skipped_on(task, "2026-11-25") and not db.task_skipped_on(task, "2026-11-26")


def test_an_untimed_one_time_add_keeps_the_validated_day(db, monkeypatch):
    import lists_tasks
    from tools import AUTHORIZED_DAY
    monkeypatch.setattr(lists_tasks, "task_day", lambda now=None: "2026-11-26")
    token = AUTHORIZED_DAY.set("2026-11-25")
    try:
        lists_tasks.task_add_tool("call the bank")
    finally:
        AUTHORIZED_DAY.reset(token)
    [row] = db.active_tasks_matching("call the bank")
    assert db.get_task(row[0])["scheduled_for"] == "2026-11-25"


def test_the_dispatch_hands_the_validated_day_to_every_covered_write(turn, db, monkeypatch):
    seen = []
    from tools import AUTHORIZED_DAY
    monkeypatch.setattr(brain.registry, "call", lambda name, args: seen.append(AUTHORIZED_DAY.get()) or "ok")
    asyncio.run(turn.run("set a timer for ten minutes", [block("set_timer", {"amount": 10, "unit": "minutes"})]))
    asyncio.run(turn.run("add call the bank", [block("task_add", {"task": "call the bank"})]))
    assert seen[1] == task_intent.validate(Intent("task_add", "call the bank"), "add call the bank").gate.plan["day"]


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
    assert task_intent.parse_intent(_message(GOOD, {**GOOD, "action": "no_change"})) is None
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


# ── F8: no_change answers normally, a declared change that did nothing says so ──

COVERED = {"task_add", "task_complete", "archive_task", "task_skip_today", "list_add", "list_check",
           "set_timer", "set_reminder"}


@pytest.mark.parametrize("said, wrong", [
    ("what is on my list", Intent("list_add", "list")),
    ("did I finish reading", Intent("task_complete", "reading")),
    ("what is on my tasks today", Intent("task_add", "tasks")),
    ("have I got a timer going", Intent("timer", "timer", "going"))])
def test_a_read_question_never_gets_the_nothing_was_changed_line(turn, db, said, wrong):
    db.add_task("reading", "daily")
    result = asyncio.run(turn.run(said, [], draft="You have reading left.", intent=wrong))
    assert result.text == "You have reading left." and "Nothing was changed" not in result.text
    assert not COVERED & {schema["name"] for schema in turn.offered[0]}


def test_a_read_question_gets_the_ordinary_answer_when_the_model_says_no_change(turn, db):
    db.add_task("reading", "daily")
    result = asyncio.run(turn.run("did I finish reading", [], draft="Not yet.", intent=Intent("no_change")))
    assert result.text == "Not yet."
    assert not COVERED & {schema["name"] for schema in turn.offered[0]}


@pytest.mark.parametrize("draft", ["Reading is done.", "Reading is off your list.", "I added that yesterday."])
def test_passive_success_wording_on_a_no_change_turn_is_not_an_unresolved_action(turn, db, draft):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run("take reading off my task list", [], draft=draft, intent=Intent("no_change")))
    assert result.text == draft and "Nothing was changed" not in result.text
    assert db.get_task(task)["archived_at"] is None


def test_a_declared_change_that_ran_nothing_says_so_with_the_reason(turn, db):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run("cancel reading today", [], draft="Done, skipped.",
                                  intent=Intent("task_skip_today", "reading")))
    assert result.text.startswith("Nothing was changed.") and "could not carry out" in result.text
    assert not db.task_skipped_on(task, db.task_day())


# ── F6: the verb must be the intent's action about the task, not a word anywhere ──

@pytest.mark.parametrize("said", [
    "I almost finished reading", "I nearly finished reading", "I barely finished reading",
    "I finished reading the book", "finishing reading is hard", "my friend finished reading",
    "I want to finish reading", "mark reading done and stop"])
def test_a_completion_verb_elsewhere_in_the_sentence_does_not_complete(turn, db, said):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run(said, [block("task_complete", {"task": "reading"})],
                                  intent=Intent("task_complete", "reading")))
    assert not db.task_completed_on(task, db.task_day()) and "Nothing was changed" in result.text


@pytest.mark.parametrize("said", [
    "I finished reading", "I just finished reading", "I've finished reading", "mark reading done",
    "mark reading as done", "mark my reading done", "complete reading", "please complete reading",
    "can you mark reading done", "I'm done with reading", "check reading off", "reading is done",
    "I finished reading today"])
def test_a_plain_completion_still_completes(turn, db, said):
    task = db.add_task("reading", "daily")
    asyncio.run(turn.run(said, [block("task_complete", {"task": "reading"})],
                         intent=Intent("task_complete", "reading")))
    assert db.task_completed_on(task, db.task_day())


@pytest.mark.parametrize("said, action", [
    ("I might skip reading today", "task_skip_today"), ("don't cancel reading today", "task_skip_today"),
    ("I almost stopped reading for good", "task_archive"), ("never skip reading today", "task_skip_today"),
    ("my sister can skip reading today", "task_skip_today")])
def test_a_removal_verb_elsewhere_in_the_sentence_changes_nothing(turn, db, said, action):
    task = db.add_task("reading", "daily")
    tool = "task_skip_today" if action == "task_skip_today" else "archive_task"
    result = asyncio.run(turn.run(said, [block(tool, {"task": "reading"})],
                                  intent=Intent(action, "reading")))
    assert not db.task_skipped_on(task, db.task_day()) and db.get_task(task)["archived_at"] is None
    assert "Nothing was changed" in result.text


def test_a_plain_cancel_today_still_skips(turn, db):
    task = db.add_task("reading", "daily")
    asyncio.run(turn.run("cancel reading today", [block("task_skip_today", {"task": "reading"})],
                         intent=Intent("task_archive", "reading")))
    assert db.task_skipped_on(task, db.task_day())


# ── F13: timing the intent did not capture writes nothing and asks ──

@pytest.mark.parametrize("intent", [
    Intent("task_add", "water plants"),
    Intent("task_add", "Water plants in four days"),
    Intent("task_add", "water plants in four days", "")])
def test_add_water_plants_in_four_days_writes_nothing_and_asks(turn, db, intent):
    result = asyncio.run(turn.run("add water plants in four days",
                                  [block("task_add", {"task": intent.target})], intent=intent))
    assert not db.active_tasks_matching("water plants")
    assert "Nothing was changed" in result.text and "What timing" in result.text and "four days" in result.text


@pytest.mark.parametrize("said", [
    "add water plants in half a dozen days", "add water plants in a couple of weeks",
    "add call the bank at dawn", "add call the bank at 5pm", "add pay rent in thirteen days",
    "add water plants in about ten minutes", "add stretch at midnight", "add review next week"])
def test_other_uncaptured_timing_writes_nothing_and_asks(turn, db, said):
    result = asyncio.run(turn.run(said, [block("task_add", {"task": "x"})],
                                  intent=Intent("task_add", said.removeprefix("add "))))
    assert "Nothing was changed" in result.text and "What timing" in result.text
    assert not db.active_tasks_matching("x")


def test_supported_timing_and_untimed_adds_still_work(turn, db):
    asyncio.run(turn.run("add water plants", [block("task_add", {"task": "water plants"})],
                         intent=Intent("task_add", "water plants")))
    [row] = db.active_tasks_matching("water plants")
    task = db.get_task(row[0])
    assert task["scheduled_for"] == db.task_day() and task["due_date"] is None and task["recurrence"] is None
    asyncio.run(turn.run("add reading every day", [block("task_add", {"task": "reading", "recurrence": "daily"})],
                         intent=Intent("task_add", "reading", "every day")))
    [row] = db.active_tasks_matching("reading")
    assert db.get_task(row[0])["recurrence"] == "daily"


def test_a_quoted_title_may_hold_a_time_word(turn, db):
    asyncio.run(turn.run('add "call at 5 pm club"', [block("task_add", {"task": "call at 5 pm club"})],
                         intent=Intent("task_add", "call at 5 pm club")))
    assert db.active_tasks_matching("call at 5 pm club")


# ── F11: the reminder content and its date are bound exactly ──

def test_a_reminder_content_must_equal_the_named_words_not_a_piece_of_them(turn, monkeypatch):
    ran = _ran(monkeypatch)
    five = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    if five < datetime.now():
        five += timedelta(days=1)
    said = "remind me to call the bank at 5 pm"
    for content in ("the bank", "call", "call the bank and the dentist"):
        asyncio.run(turn.run(said, [block("set_reminder", {"content": content, "due": five.isoformat()})]))
    assert ran == []
    asyncio.run(turn.run(said, [block("set_reminder", {"content": "to call the bank", "due": five.isoformat()})]))
    assert len(ran) == 1


def test_tomorrow_at_five_is_tomorrows_date_never_a_later_one(turn, monkeypatch):
    ran = _ran(monkeypatch)
    base = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
    said = "remind me to call the bank tomorrow at 5 pm"
    for due in (base + timedelta(days=21), base, base + timedelta(days=2)):
        asyncio.run(turn.run(said, [block("set_reminder", {"content": "call the bank", "due": due.isoformat()})]))
    assert ran == []
    asyncio.run(turn.run(said, [block("set_reminder",
                                      {"content": "call the bank", "due": (base + timedelta(days=1)).isoformat()})]))
    assert len(ran) == 1


def test_a_reminder_without_a_time_cannot_carry_one(turn, monkeypatch):
    ran = _ran(monkeypatch)
    soon = (datetime.now() + timedelta(hours=1)).isoformat()
    asyncio.run(turn.run("remind me to call the bank", [block("set_reminder", {"content": "call the bank", "due": soon})]))
    assert ran == []
    asyncio.run(turn.run("remind me to call the bank", [block("set_reminder", {"content": "call the bank"})]))
    assert len(ran) == 1


# ── The phrases that must keep working through the intent step ──

def test_the_everyday_phrases_still_work(turn, db, monkeypatch):
    ran = _ran(monkeypatch, "ok")
    asyncio.run(turn.run("set a timer for 10 minutes", [block("set_timer", {"amount": 10, "unit": "minutes"})]))
    asyncio.run(turn.run("add eggs to my grocery list",
                         [block("list_add", {"item": "eggs", "list_name": "grocery"})]))
    asyncio.run(turn.run("add reading", [block("task_add", {"task": "reading"})]))
    assert [name for name, _ in ran] == ["set_timer", "list_add", "task_add"]
    task = db.add_task("walking", "daily")
    asyncio.run(turn.run("cancel walking today", [block("task_skip_today", {"task": "walking"})]))
    assert ("task_skip_today", {"task": "walking"}) in [(n, a) for n, a in ran]


# ── Title matching by meaning ──

@pytest.mark.parametrize("said, title", [
    ("add water to the cactus", "Water the cactus"),
    ("add pick up the dry cleaning", "Pick up dry cleaning"),
    ("add call mom", "Call Mom"),
    ("add take out the trash", "Take out trash"),
    ("add buy a gift for Jordan", "Buy gift for Jordan"),
    ("add email the professor about the project", "Email professor about project"),
    ("add walk the dogs", "Walk the dog"),
    ("add reading every day", "reading"),
])
def test_a_title_that_means_what_he_said_is_accepted(db, said, title):
    verdict = task_intent.validate(Intent("task_add", title, "every day" if "every day" in said else ""),
                                   said, date(2026, 11, 25))
    assert verdict.kind == "act", verdict.text
    recurrence = "daily" if "every day" in said else None
    assert verdict.gate.permits("task_add", {"task": title, **({"recurrence": "daily"} if recurrence else {})})


@pytest.mark.parametrize("said, title", [
    ("add water the cactus", "Water the garden"),
    ("add call mom", "Call dad"),
    ("add buy milk", "Buy milk and eggs"),
    ("add reading", "Reading every day"),
    ("add call mom", "Call"),
])
def test_a_title_that_adds_or_drops_a_word_is_refused(db, said, title):
    verdict = task_intent.validate(Intent("task_add", title), said, date(2026, 11, 25))
    assert verdict.kind != "act"
    assert verdict.text.startswith("Nothing was changed.")


def test_the_models_call_title_is_held_to_the_same_rule(db):
    verdict = task_intent.validate(Intent("task_add", "water the cactus"), "add water to the cactus",
                                   date(2026, 11, 25))
    assert verdict.kind == "act"
    assert verdict.gate.permits("task_add", {"task": "Water the Cactus"})
    assert not verdict.gate.permits("task_add", {"task": "Water the garden"})
    assert not verdict.gate.permits("task_add", {"task": "Water the cactus", "recurrence": "daily"})
