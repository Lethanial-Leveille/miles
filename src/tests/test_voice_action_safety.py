"""Synthetic voice dispatch tests. No provider, microphone or calendar traffic."""

import asyncio
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

import brain
import pending_action as pa
from tools import Permission
from voice_action_safety import TurnSafety


def block(name, args=None, id=None):
    return SimpleNamespace(type="tool_use", name=name, input=args or {}, id=id or name)


def state(db, task):
    return db.get_task(task)["archived_at"], db.get_task(task)["status"]


class Stream:
    def __init__(self, blocks, draft):
        self.blocks = blocks
        self.draft = draft

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def __aiter__(self):
        yield SimpleNamespace(type="content_block_delta",
                              delta=SimpleNamespace(type="text_delta", text=self.draft))

    async def get_final_message(self):
        return SimpleNamespace(content=self.blocks, usage=SimpleNamespace())


@pytest.fixture
def voice(db, monkeypatch):
    audio, events, rounds = [], [], []
    monkeypatch.setattr(brain, "build_request",
                        lambda *a, **kw: brain.TurnRequest("synthetic", "system", [], "fixture"))
    monkeypatch.setattr(brain, "_history_for", lambda *a, **kw: [])
    monkeypatch.setattr(brain.memory_pass, "notice_later", lambda *a, **kw: None)
    monkeypatch.setattr(brain.alerts, "take_for_fold", lambda: [])
    monkeypatch.setattr(brain, "claude", SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: rounds.pop(0))))
    monkeypatch.setattr(pa, "_conversations", {})

    async def consumer(queue, parts, leaks, heard):
        while (item := await queue.get()) is not None:
            audio.append(item)
            parts.append(item)
            heard.played([item])
            heard.confirmable_audio = True

    monkeypatch.setattr(brain, "_consumer_for", lambda *a: consumer)
    monkeypatch.setattr(brain, "speak", audio.append)

    async def run(utterance, blocks, *, tier="hokage", draft="Done. I saved the task.", extra=()):
        rounds.extend([Stream(blocks, draft), *extra])
        return await brain.ask_nova_async(utterance, tier=tier, on_text=lambda *e: events.append(e))

    return SimpleNamespace(run=run, audio=audio, events=events)


@pytest.mark.parametrize("reverse", [False, True])
def test_refused_schedule_cannot_speak_success_or_let_archive_escape(voice, db, reverse):
    existing = db.add_task("water plants")
    batch = [block("task_add", {"task": "reading", "recurrence": "daily"}),
             block("archive_task", {"task": "water plants"})]
    if reverse:
        batch.reverse()
    result = asyncio.run(voice.run("Add reading daily starting Monday", batch))
    assert state(db, existing) == (None, "pending")
    assert not db.active_tasks_matching("reading")
    assert "Nothing was changed" in result.text
    assert all("Done" not in line and "saved the task" not in line for line in voice.audio)
    assert ("reset", "") in voice.events


def test_valid_guarded_outcome_is_only_spoken_after_write(voice, db, monkeypatch):
    real_call = brain.registry.call
    checked = []

    def dispatch(name, args):
        assert voice.audio == []
        result = real_call(name, args)
        assert db.active_tasks_matching("reading")
        checked.append(name)
        return result

    monkeypatch.setattr(brain.registry, "call", dispatch)
    result = asyncio.run(voice.run("Add reading", [block("task_add", {"task": "reading"})]))
    assert checked == ["task_add"]
    assert voice.audio == [result.text]
    assert result.text == "Added the task: Reading."


def test_failed_mutation_and_duplicate_never_claim_success(voice, db, monkeypatch):
    attempts = []

    def fail(name, args):
        attempts.append(name)
        raise RuntimeError("synthetic hidden detail")

    monkeypatch.setattr(brain.registry, "call", fail)
    args = {"task": "reading"}
    result = asyncio.run(voice.run("Add reading", [block("task_add", args, "first"),
                                                   block("task_add", args, "duplicate")]))
    assert attempts == ["task_add"] and not db.active_tasks_matching("reading")
    assert "could not confirm" in result.text and "Nothing was repeated" in result.text
    assert "synthetic hidden detail" not in result.text and not any("Done" in s for s in voice.audio)


def test_clearance_and_invalid_sibling_preflight_precede_any_mutation(voice, db):
    task = db.add_task("reading")
    result = asyncio.run(voice.run("Archive reading", [block("archive_task", {"task": "reading"})],
                                   tier="genin"))
    assert "clearance" in result.text and state(db, task) == (None, "pending")
    result = asyncio.run(voice.run("Archive reading", [
        block("archive_task", {"task": "reading"}), block("task_add", {"task": 45})]))
    assert "Nothing was changed" in result.text and state(db, task) == (None, "pending")


def test_mixed_batch_preserves_vetted_read_but_not_a_refused_write(voice, db):
    existing = db.add_task("ordinary", scheduled_for=db.task_day())
    result = asyncio.run(voice.run("Add reading after Monday", [
        block("tasks_today"), block("task_add", {"task": "reading"})]))
    assert state(db, existing) == (None, "pending")
    assert "ordinary" in result.text.casefold() and "Nothing was changed" in result.text
    assert not db.active_tasks_matching("reading")


def test_two_vetted_reads_overlap_and_keep_original_order(monkeypatch):
    simultaneous = Barrier(2)

    def call(name, args):
        simultaneous.wait(timeout=3)
        return name

    monkeypatch.setattr(brain.registry, "call", call)
    monkeypatch.setattr(brain, "log_tool_call", lambda *a, **k: None)
    result = asyncio.run(brain._dispatch_tools([block("tasks_today"), block("list_show", {"list_name": "grocery"})],
                                               "synthetic", "hokage", TurnSafety("Read")))
    assert [row["output"] for row in result] == ["tasks_today", "list_show"]


def test_followup_draft_is_discarded_before_refused_mutation(voice, db):
    task = db.add_task("reading")
    result = asyncio.run(voice.run("Show tasks", [block("tasks_today")], draft="Done. I archived the task.",
                                   extra=[Stream([block("archive_task", {"task": "reading"})],
                                                 "Done. I archived the task.")]))
    assert state(db, task) == (None, "pending")
    assert "Nothing was changed" in result.text
    assert all("archived the task" not in s for s in voice.audio)


def test_cancellation_before_dispatch_and_during_worker(monkeypatch):
    started, release, done = Event(), Event(), Event()
    calls = []

    def call(name, args):
        calls.append(name)
        started.set()
        release.wait(timeout=3)
        done.set()
        return "Recorded"

    monkeypatch.setattr(brain.registry, "call", call)
    monkeypatch.setattr(brain, "log_tool_call", lambda *a, **kw: None)

    async def run():
        action = block("task_add", {"task": "reading"})
        first = TurnSafety("Add reading")
        first.cancelled.set()
        refused = await brain._dispatch_tools([action], "synthetic", "hokage", first)
        assert refused[0]["is_error"] and calls == []

        second = TurnSafety("Add reading")
        work = asyncio.create_task(brain._dispatch_tools([action], "synthetic", "hokage", second))
        assert await asyncio.to_thread(started.wait, 3)
        work.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await work
        finally:
            release.set()
        assert await asyncio.to_thread(done.wait, 3)
        again = await brain._dispatch_tools([action], "synthetic", "hokage", second)
        assert again[0]["is_error"] and calls == ["task_add"]

    asyncio.run(run())


def test_confirmation_uses_presented_proposal_channel_tier_and_explicit_answer(monkeypatch):
    monkeypatch.setattr(pa, "_conversations", {})
    scope = ("synthetic-actor", "room")
    other = ("synthetic-actor", "app")
    spec = brain.registry.get("confirm_pending_action")
    callback = []
    source = Event()
    delivery = Event()
    with pa.committed_turn(scope, "hokage", "propose a change", source, delivery), pa.proposing_tool(spec):
        question = pa.propose("Change the synthetic appointment", lambda: callback.append(1) or "Changed.")
        assert not pa.confirmation_allowed(True)
        pa.mark_presented(question)
    delivery.set()
    with pa.committed_turn(other, "hokage", "yes", Event()):
        assert not pa.confirmation_allowed(True)
    with pa.committed_turn(scope, "hokage", "yes, but change the time", Event()):
        assert not pa.confirmation_allowed(True)
        assert "Nothing was changed" in pa.resolve(True)
    assert callback == []


def test_confirmation_partial_outcome_is_reported_without_replaying(monkeypatch):
    monkeypatch.setattr(pa, "_conversations", {})
    scope = ("synthetic-actor", "room")
    spec = brain.registry.get("confirm_pending_action")
    callbacks = []

    def fail():
        callbacks.append("failed")
        raise RuntimeError("synthetic hidden detail")

    with pa.committed_turn(scope, "hokage", "propose two changes", Event()), pa.proposing_tool(spec):
        pa.propose("Change first synthetic appointment", lambda: callbacks.append("first") or "First changed.")
        question = pa.propose("Change second synthetic appointment", fail)
        pa.mark_presented(question)
    with pa.committed_turn(scope, "hokage", "yes", Event()):
        assert pa.confirmation_allowed(True)
        outcome = pa.resolve(True)
        assert "First changed." in outcome and "Could not confirm" in outcome
        assert "hidden detail" not in outcome and not pa.confirmation_allowed(True)
    assert callbacks == ["first", "failed"]


@pytest.mark.parametrize("mode", ["undelivered", "cancelled", "denied", "expired", "wrong tier"])
def test_unusable_proposals_do_not_mutate(monkeypatch, mode):
    monkeypatch.setattr(pa, "_conversations", {})
    scope = ("synthetic", "app")
    spec = brain.registry.get("confirm_pending_action")
    source, delivery = Event(), Event()
    calls = []
    base = [10.0]
    monkeypatch.setattr(pa.time, "monotonic", lambda: base[-1])
    with pa.committed_turn(scope, "hokage", "propose", source, delivery), pa.proposing_tool(spec):
        question = pa.propose("Change synthetic item", lambda: calls.append(True))
        pa.mark_presented(question)
    if mode != "undelivered":
        delivery.set()
    if mode == "cancelled":
        source.set()
    if mode == "expired":
        base.append(10.0 + pa.CONFIRM_WINDOW_S + 1)
    with pa.committed_turn(scope, "genin" if mode == "wrong tier" else "hokage",
                           "no" if mode == "denied" else "yes", Event()):
        if mode == "denied":
            assert not pa.confirmation_allowed(True)
            assert pa.confirmation_allowed(False)
            assert pa.resolve(False).startswith("Cancelled")
        else:
            assert not pa.confirmation_allowed(True)
            assert "Nothing was changed" in pa.resolve(True)
    assert calls == []


def test_model_without_a_tool_result_cannot_announce_completion(voice):
    result = asyncio.run(voice.run("Add reading", []))
    assert result.text == "I could not confirm that action."
    assert voice.audio == [result.text] and ("reset", "") in voice.events


def test_clear_task_request_cannot_be_diverted_into_unrelated_mutation(voice, monkeypatch):
    def no_tool(*args):
        pytest.fail("Unrelated write was dispatched")

    monkeypatch.setattr(brain.registry, "call", no_tool)
    result = asyncio.run(voice.run("Add a daily task to read", [
        block("set_timer", {"amount": 4, "unit": "minutes"})]))
    assert "Nothing was changed" in result.text and not any("Done" in line for line in voice.audio)


def test_new_request_in_same_conversation_cancels_active_authority(monkeypatch):
    monkeypatch.setattr(pa, "_conversations", {})
    scope = ("synthetic", "same conversation")
    older, newer = Event(), Event()
    with pa.committed_turn(scope, "hokage", "first", older):
        with pa.committed_turn(scope, "hokage", "next", newer):
            assert older.is_set() and not newer.is_set()


def test_model_cannot_hide_a_second_requested_change_in_one_title(voice, db):
    result = asyncio.run(voice.run("Add reading and archive planning", [
        block("task_add", {"task": "reading and archive planning"})]))
    assert "one change at a time" in result.text
    assert not db.active_tasks_matching("reading and archive planning")


def test_only_a_delivered_question_can_authorize_the_confirmed_result(voice, monkeypatch):
    actual = brain.registry.call
    changes = []

    def call(name, args):
        if name == "create_calendar_event":
            return pa.propose("Create the synthetic event", lambda: changes.append(1) or "Event created.")
        return actual(name, args)

    monkeypatch.setattr(brain.registry, "call", call)
    propose = block("create_calendar_event", {
        "summary": "Synthetic", "start_time": "Tomorrow at two", "duration_minutes": 60})
    result = asyncio.run(voice.run("Propose the synthetic event", [propose]))
    assert result.text == "Create the synthetic event?"
    assert changes == [] and voice.audio == [result.text]
    result = asyncio.run(voice.run("Yes, go ahead", [
        block("confirm_pending_action", {"approved": True})]))
    assert result.text == "Event created." and changes == [1]
    assert voice.audio[-1] == "Event created."
    result = asyncio.run(voice.run("Yes", [block("confirm_pending_action", {"approved": True})]))
    assert "Nothing was changed" in result.text and changes == [1]


def test_dependent_task_read_sees_committed_mutation_and_audit_failure_cannot_hide_it(voice, db, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic audit store unavailable")

    monkeypatch.setattr(brain, "log_tool_call", fail)
    result = asyncio.run(voice.run("Add reading", [block("tasks_today"), block("task_add", {"task": "reading"})]))
    assert db.active_tasks_matching("reading")
    assert "Added the task: Reading." in result.text
    assert result.text.casefold().count("reading") == 2


def test_absent_mutation_result_never_authorizes_success(voice, monkeypatch):
    monkeypatch.setattr(brain.registry, "call", lambda *args: None)
    result = asyncio.run(voice.run("Add reading", [block("task_add", {"task": "reading"})]))
    assert "could not confirm" in result.text
    assert "Done" not in " ".join(voice.audio)
