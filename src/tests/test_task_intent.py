"""The request intent step. Offline, with a fake model and fake names only."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

import brain
import pending_action as pa
import task_intent
import task_scheduling
import timing
from datetime import date, datetime
from intent_fake import fake_intent
from task_intent import Intent

REAL_REQUEST = task_intent.request_intent
from voice_action_safety import NO_ACTION, TurnSafety


def block(name, args=None, id=None):
    return SimpleNamespace(type="tool_use", name=name, input=args or {}, id=id or name)


class Stream:
    def __init__(self, blocks, draft=""):
        self.blocks, self.draft = blocks, draft

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def __aiter__(self):
        if self.draft:
            yield SimpleNamespace(type="content_block_delta",
                                  delta=SimpleNamespace(type="text_delta", text=self.draft))

    async def get_final_message(self):
        return SimpleNamespace(content=self.blocks, usage=SimpleNamespace())


@pytest.fixture
def turn(db, monkeypatch):
    audio, events, rounds, offered, intent_calls = [], [], [], [], []
    monkeypatch.setattr(brain, "build_request",
                        lambda *a, **kw: brain.TurnRequest("synthetic", "system", [], "fixture"))
    monkeypatch.setattr(brain, "_history_for", lambda *a, **kw: [])
    monkeypatch.setattr(brain.memory_pass, "notice_later", lambda *a, **kw: None)
    monkeypatch.setattr(brain.alerts, "take_for_fold", lambda: [])
    monkeypatch.setattr(pa, "_conversations", {})

    def stream(**kwargs):
        offered.append(kwargs.get("tools", []))
        return rounds.pop(0)

    monkeypatch.setattr(brain, "claude", SimpleNamespace(messages=SimpleNamespace(stream=stream)))

    async def consumer(queue, parts, leaks, heard):
        while (item := await queue.get()) is not None:
            audio.append(item)
            parts.append(item)
            heard.played([item])
            heard.confirmable_audio = True

    monkeypatch.setattr(brain, "_consumer_for", lambda *a: consumer)
    monkeypatch.setattr(brain, "speak", audio.append)

    async def run(utterance, blocks=(), *, draft="", intent=None, more=(), scope=None):
        async def request(client, text, pending_title=None):
            intent_calls.append(text)
            return intent if intent is not None else fake_intent(text)
        monkeypatch.setattr(task_intent, "request_intent", request)
        rounds.clear()  # a refused turn never reads its stream
        rounds.extend([Stream(list(blocks), draft), *more])
        return await brain.ask_nova_async(utterance, tier="hokage", on_text=lambda *e: events.append(e))

    return SimpleNamespace(run=run, audio=audio, events=events, rounds=rounds, offered=offered,
                           intent_calls=intent_calls)


def active(db, words):
    return db.active_tasks_matching(words)


# ── Which turns could change something ──

@pytest.mark.parametrize("text", [
    "add water plants", "cancel reading", "set a timer for ten minutes", "remind me at five",
    "add eggs to my grocery list", "I took my vitamins", "put a meeting on my calendar",
    "remember that my cat is Sam", "stop tracking the bank task", "skip reading today"])
def test_the_wide_check_flags_requests(text):
    assert task_intent.could_change(text)


@pytest.mark.parametrize("text", [
    "how is the weather", "tell me a joke", "what is the capital of France",
    "good morning", "how was my sleep", "what time is it"])
def test_ordinary_conversation_is_not_flagged(text):
    assert not task_intent.could_change(text)


def test_a_miss_by_the_wide_check_exposes_no_mutating_tool_and_makes_no_extra_call(turn, db, monkeypatch):
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append(name) or "ok")
    # The model tries a timer anyway, on a turn that is just chat.
    result = asyncio.run(turn.run("what a lovely afternoon", [block("set_timer", {"amount": 5, "unit": "minutes"})]))
    assert turn.intent_calls == [] and ran == []
    names = {schema["name"] for schema in turn.offered[0]}
    assert "set_timer" not in names and "task_add" not in names and "archive_task" not in names
    assert "get_weather" in names and "tasks_today" in names
    assert "Nothing was changed" in result.text


def test_ordinary_chat_costs_no_intent_call_and_keeps_model_text(turn):
    result = asyncio.run(turn.run("good morning", [], draft="Good morning to you."))
    assert turn.intent_calls == [] and result.text == "Good morning to you."


def test_a_flagged_turn_costs_exactly_one_intent_call(turn, db):
    asyncio.run(turn.run("Add water plants", [block("task_add", {"task": "water plants"})]))
    assert turn.intent_calls == ["Add water plants"]


# ── Timing the amendment does not support ──

@pytest.mark.parametrize("utterance, target, timing_words", [
    ("add water plants in four days", "water plants", "in four days"),
    ("add water plants in four days", "water plants in four days", ""),
    ("add call the bank at dawn", "call the bank", "at dawn"),
    ("add call the bank at dawn", "call the bank at dawn", ""),
    ("add call the bank at half past five", "call the bank", "at half past five"),
    ("add water plants Saturdays", "water plants Saturdays", ""),
])
def test_unsupported_timing_writes_nothing_and_asks(turn, db, utterance, target, timing_words):
    result = asyncio.run(turn.run(utterance, intent=Intent("task_add", target, timing_words)))
    assert "Nothing was changed" in result.text and "?" in result.text
    assert not active(db, "water plants") and not active(db, "call the bank")
    assert not turn.offered, "no model call was made after a refusal"
    # It never defaults to today without his answer.
    assert db.tasks_today() == []


def test_a_model_that_invents_timing_is_not_believed(turn, db):
    result = asyncio.run(turn.run("add water plants", intent=Intent("task_add", "water plants", "every day")))
    assert "Nothing was changed" in result.text and not active(db, "water plants")


@pytest.mark.parametrize("utterance, timing_words, recurrence", [
    ("add water plants", "", None),
    ("add water plants every day", "every day", "daily"),
    ("I need to take my vitamins every day", "every day", "daily"),
])
def test_supported_adds_still_work_through_the_intent_step(turn, db, utterance, timing_words, recurrence):
    target = "take my vitamins" if "vitamins" in utterance else "water plants"
    result = asyncio.run(turn.run(utterance, [block("task_add", {"task": target, **(
        {"recurrence": recurrence} if recurrence else {})})],
        intent=Intent("task_add", target, timing_words), draft="Done."))
    [row] = active(db, target)
    assert row[2] == recurrence and "Added the" in result.text
    if recurrence is None:
        # Untimed one time add: today's task day, no due date.
        task = db.get_task(row[0])
        assert task["scheduled_for"] == db.task_day() and task["due_date"] is None


def test_a_first_day_is_refused_because_voice_cannot_store_it(turn, db):
    result = asyncio.run(turn.run("add reading every day starting Monday",
                                  intent=Intent("task_add", "reading", "every day starting Monday")))
    assert "Nothing was changed" in result.text and not active(db, "reading")


def test_the_validator_sees_spoken_counts_and_dawn():
    today = date(2026, 11, 25)
    for said in ("add water plants in four days", "add call the bank at dawn",
                 "add water plants in four days"):
        plan = task_scheduling.resolve_task_schedule(said, title=said.removeprefix("add "), today=today)
        assert plan.outcome != "supported", said


# ── The gate ──

def test_cancel_with_timer_and_reminder_in_the_first_batch_dispatches_nothing(turn, db, monkeypatch):
    task = db.add_task("reading")
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append(name) or "ok")
    batch = [block("set_timer", {"amount": 4, "unit": "minutes"}),
             block("set_reminder", {"content": "x"}),
             block("tasks_today")]
    result = asyncio.run(turn.run("cancel reading", batch, more=[Stream([block("archive_task", {"task": "reading"})])]))
    assert "set_timer" not in ran and "set_reminder" not in ran and "archive_task" not in ran
    assert "Nothing was changed" in result.text
    assert db.get_task(task)["archived_at"] is None


def test_the_gate_holds_in_every_round(turn, db, monkeypatch):
    db.add_task("reading")
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append(name) or "ok")
    # Round one is a read; round two tries a change that was never asked for.
    result = asyncio.run(turn.run(
        "show my tasks and cancel nothing", [], intent=Intent("none")))
    assert result.text
    turn.rounds.clear()
    result = asyncio.run(turn.run("Add water plants", [block("tasks_today")],
                                  intent=Intent("task_add", "water plants"),
                                  more=[Stream([block("set_timer", {"amount": 1, "unit": "minutes"})])]))
    assert "set_timer" not in ran
    assert "Nothing was changed" in result.text


@pytest.mark.parametrize("wrong", [
    block("task_complete", {"task": "reading"}), block("task_add", {"task": "reading"}),
    block("archive_task", {"task": "someone else"})])
def test_only_the_matching_tool_and_task_may_run(turn, db, wrong):
    task = db.add_task("reading")
    db.add_task("someone else")
    result = asyncio.run(turn.run("archive reading", [wrong], intent=Intent("task_archive", "reading")))
    assert "Nothing was changed" in result.text
    assert db.get_task(task)["archived_at"] is None and db.get_task(task)["status"] == "pending"


def test_cancel_reading_with_no_tool_call_never_claims_it_archived(turn, db):
    task = db.add_task("reading")
    result = asyncio.run(turn.run("cancel reading", [], draft="Done. I archived Reading.",
                                  intent=Intent("task_archive", "reading")))
    assert result.text == NO_ACTION and "archived" not in result.text.lower()
    assert db.get_task(task)["archived_at"] is None
    assert not any("archived" in line.lower() for line in turn.audio)


def test_a_turn_that_ran_no_action_cannot_say_it_did(turn, db):
    result = asyncio.run(turn.run("what do I need to add", [], draft="Done. I added it.", intent=Intent("none")))
    assert result.text == "Nothing was changed."


# ── Cancel ──

def test_bare_cancel_on_a_daily_task_asks_and_writes_nothing(turn, db):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run("cancel reading", intent=Intent("task_archive", "reading")))
    assert "Skip today or stop it for good?" in result.text and "Nothing was changed" in result.text
    assert db.get_task(task)["archived_at"] is None and not db.task_skipped_on(task, db.task_day())
    assert not turn.offered


def test_cancel_with_today_skips_only_today(turn, db):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run("cancel reading today", [block("task_skip_today", {"task": "reading"})],
                                  intent=Intent("task_archive", "reading")))
    assert result.text == "Skipped reading for today."
    assert db.get_task(task)["archived_at"] is None and db.task_skipped_on(task, db.task_day())
    assert db.tasks_today() == []


def test_cancel_with_today_cannot_be_turned_into_an_archive(turn, db):
    task = db.add_task("reading", "daily")
    asyncio.run(turn.run("cancel reading tonight", [block("archive_task", {"task": "reading"})],
                         intent=Intent("task_skip_today", "reading")))
    assert db.get_task(task)["archived_at"] is None


def test_bare_cancel_on_a_one_time_task_archives_it(turn, db):
    task = db.add_task("bank call")
    result = asyncio.run(turn.run("cancel bank call", [block("archive_task", {"task": "bank call"})],
                                  intent=Intent("task_archive", "bank call")))
    assert db.get_task(task)["archived_at"] is not None and "Stopped tracking" in result.text


@pytest.mark.parametrize("words", ["stop reading", "archive reading", "delete reading", "cancel reading for good"])
def test_a_clear_stop_request_archives_a_daily_task(turn, db, words):
    task = db.add_task("reading", "daily")
    asyncio.run(turn.run(words, [block("archive_task", {"task": "reading"})],
                         intent=Intent("task_archive", "reading")))
    assert db.get_task(task)["archived_at"] is not None


@pytest.mark.parametrize("answer, tool, intent_action, archived", [
    ("skip today", "task_skip_today", "task_skip_today", False),
    ("stop it for good", "archive_task", "task_archive", True)])
def test_the_answer_is_bound_to_the_question_and_goes_through_the_intent_step(
        turn, db, answer, tool, intent_action, archived):
    task = db.add_task("reading", "daily")
    asyncio.run(turn.run("cancel reading", intent=Intent("task_archive", "reading")))
    assert db.get_task(task)["archived_at"] is None
    asyncio.run(turn.run(answer, [block(tool, {"task": "reading"})], intent=Intent(intent_action, "Reading")))
    assert (db.get_task(task)["archived_at"] is not None) == archived
    assert db.task_skipped_on(task, db.task_day()) == (not archived)


def test_an_answer_with_no_question_pending_is_not_an_action(turn, db):
    task = db.add_task("reading", "daily")
    result = asyncio.run(turn.run("skip today", [block("task_skip_today", {"task": "reading"})],
                                  intent=Intent("task_skip_today", "Reading")))
    assert "Nothing was changed" in result.text and not db.task_skipped_on(task, db.task_day())


def test_the_question_expires_after_one_unrelated_turn(turn, db):
    task = db.add_task("reading", "daily")
    asyncio.run(turn.run("cancel reading", intent=Intent("task_archive", "reading")))
    asyncio.run(turn.run("good morning", [], draft="Morning."))
    result = asyncio.run(turn.run("skip today", [block("task_skip_today", {"task": "reading"})],
                                  intent=Intent("task_skip_today", "Reading")))
    assert not db.task_skipped_on(task, db.task_day()) and "Nothing was changed" in result.text


def test_two_matching_tasks_ask_which_one(turn, db):
    db.add_task("reading chapter one")
    db.add_task("reading chapter two")
    result = asyncio.run(turn.run("cancel reading", intent=Intent("task_archive", "reading")))
    assert "Which one" in result.text


# ── Unaffected requests still work through the intent step ──

def test_timer_reminder_and_list_requests_still_work(turn, db, monkeypatch):
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append(name) or "ok")
    asyncio.run(turn.run("set a timer for 10 minutes",
                         [block("set_timer", {"amount": 10, "unit": "minutes"})], draft="Timer set."))
    five = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0).isoformat()
    asyncio.run(turn.run("remind me to call the bank at 5",
                         [block("set_reminder", {"content": "call the bank", "due": five})], draft="Okay."))
    result = asyncio.run(turn.run("add eggs to my grocery list",
                                  [block("list_add", {"item": "eggs", "list_name": "grocery"})]))
    assert ran == ["set_timer", "set_reminder", "list_add"]
    assert result.text == "ok"


def test_a_timer_cannot_ride_along_with_a_list_add(turn, db, monkeypatch):
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append(name) or "ok")
    asyncio.run(turn.run("add eggs to my grocery list", [
        block("list_add", {"item": "eggs", "list_name": "grocery"}),
        block("set_timer", {"amount": 1, "unit": "minutes"})]))
    assert ran == []


def test_an_unclear_model_answer_refuses(turn, db, monkeypatch):
    async def broken(client, text, pending_title=None):
        return None

    monkeypatch.setattr(task_intent, "request_intent", broken)
    result = asyncio.run(turn.run("cancel reading"))
    assert "Nothing was changed" in result.text and not turn.offered


def test_two_changes_in_one_request_are_refused(turn, db):
    result = asyncio.run(turn.run("add milk to my grocery list and set a timer for five minutes",
                                  intent=Intent("list_add", "milk")))
    assert "one change at a time" in result.text


# ── What the client sees ──

def test_a_flagged_turn_sends_no_text_until_the_actions_finish(turn, db):
    result = asyncio.run(turn.run("Add water plants", [block("task_add", {"task": "water plants"})],
                                  draft="Done. I saved the task.", intent=Intent("task_add", "water plants")))
    deltas = [e for e in turn.events if e[0] == "delta"]
    assert deltas == [("delta", "Added the task: Water plants.")] and ("reset", "") not in turn.events
    assert result.text == "Added the task: Water plants."


@pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
def test_streaming_route_never_shows_provisional_success(turn, db, monkeypatch, route):
    from fastapi.testclient import TestClient
    import server

    monkeypatch.setattr(brain, "_claude", lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: turn.rounds.pop(0))))
    monkeypatch.setattr(brain, "effective_tier", lambda: "hokage")
    monkeypatch.setitem(server.app.dependency_overrides, server.get_current_user, lambda: "fake-actor")
    turn.rounds.append(Stream([], "Done. I saved the task."))
    body = {"message": "add water plants Saturdays", "channel": "text",
            "conversation_id": str(uuid4()), "request_id": str(uuid4())}
    client = TestClient(server.app)
    if route == "/chat":
        text = client.post(route, json=body).json()["response"]
        assert "Done" not in text and "Nothing was changed" in text
        return
    with client.stream("POST", route, json=body) as streamed:
        frames = [json.loads(line[6:]) for line in streamed.iter_lines() if line.startswith("data: ")]
    shown = [frame["text"] for frame in frames]
    assert shown and all("Done" not in text and "saved the task" not in text for text in shown)
    # A connection that drops after any prefix of the frames shows no success.
    assert "Nothing was changed" in shown[-1]


# ── The intent call itself ──

class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[block("record_intent", self.payload)])


def test_the_intent_call_is_forced_schema_with_the_production_model_and_is_timed():
    from config import MODEL_A
    client = FakeClient({"action": "task_add", "target": "water plants", "timing": "",
                         "more_than_one_change": False})
    timing.begin_turn("text")
    intent = asyncio.run(REAL_REQUEST(client, "add water plants"))
    assert intent == Intent("task_add", "water plants")
    [call] = client.calls
    assert call["model"] == MODEL_A
    assert call["tool_choice"] == {"type": "tool", "name": "record_intent"}
    assert [tool["name"] for tool in call["tools"]] == ["record_intent"]
    assert timing.last_intent_ms() is not None and timing.last_intent_ms() >= 0
    timing.abandon_turn()


@pytest.mark.parametrize("payload", [
    {"action": "launch missiles", "target": "", "timing": "", "more_than_one_change": False},
    {"action": "task_add", "target": 5, "timing": "", "more_than_one_change": False},
    {"action": "task_add", "target": "x", "timing": "", "more_than_one_change": "no"}])
def test_an_answer_outside_the_schema_is_rejected(payload):
    assert asyncio.run(REAL_REQUEST(FakeClient(payload), "add x")) is None


def test_a_failed_call_is_none_and_never_raises():
    assert asyncio.run(REAL_REQUEST(FakeClient(error=RuntimeError("synthetic")), "add x")) is None


def test_other_may_name_only_a_registered_mutating_tool():
    ok = task_intent.validate(Intent("other", tool="create_calendar_event"), "add a calendar event", date(2026, 11, 25))
    assert ok.kind == "act" and ok.gate.offers("create_calendar_event") and not ok.gate.offers("set_timer")
    for tool in ("get_weather", "nonsense", "task_add", "confirm_pending_action", ""):
        assert task_intent.validate(Intent("other", tool=tool), "do something", date(2026, 11, 25)).kind == "refuse"


def test_the_closed_gate_offers_nothing_but_a_pending_confirmation():
    assert not task_intent.closed_gate().offers("set_timer")
    assert task_intent.closed_gate(confirm=True).offers("confirm_pending_action")
    safety = TurnSafety("hello", gate=task_intent.closed_gate())
    names = {s["name"] for s in safety.tool_schemas(brain.registry.api_schemas(), brain.registry)}
    assert "set_timer" not in names and "confirm_pending_action" not in names and "get_weather" in names
