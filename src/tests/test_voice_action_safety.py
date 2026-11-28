"""Synthetic voice dispatch tests. No provider, microphone or calendar traffic."""

import asyncio
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

import brain
import pending_action as pa
from tools import Permission
from voice_action_safety import NO_ACTION, TurnSafety


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
        rounds.clear()  # a turn the intent step refused never reads its stream
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
    # The refusal is the only text the client ever sees: no provisional success.
    assert ("reset", "") not in voice.events
    assert [e for e in voice.events if e[0] == "delta"] == [("delta", result.text)]


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
    # The intent is a plain add, so the read may run, but the timer riding along
    # is refused and the add is refused with it, before either runs.
    result = asyncio.run(voice.run("Add reading", [
        block("tasks_today"), block("set_timer", {"amount": 4, "unit": "minutes"}),
        block("task_add", {"task": "reading"})]))
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
    assert result.text == NO_ACTION
    assert " ".join(voice.audio) == result.text
    assert [e for e in voice.events if e[0] == "delta"] == [("delta", result.text)]


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
    result = asyncio.run(voice.run("Add the synthetic event to my calendar", [propose]))
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


@pytest.mark.parametrize("title", [
    "water plants Saturdays", "call the bank at half past five",
    "water plants on the first of the month", "water plants fortnightly",
    "send the report EOD",
])
def test_unsupported_timing_does_not_become_an_ordinary_task(voice, db, title):
    result = asyncio.run(voice.run(f"Add {title}", [block("task_add", {"task": title})]))
    assert not db.active_tasks_matching(title)
    assert "Nothing was changed" in result.text
    assert all("saved the task" not in text for text in voice.audio)


@pytest.mark.parametrize(("utterance", "args"), [
    ("Add call the bank to my tasks", {"task": "call the bank"}),
    ("I need to take my vitamins every day", {"task": "take my vitamins", "recurrence": "daily"}),
    ("Create a daily task to read", {"task": "read", "recurrence": "daily"}),
    ("Add watch friday night lights", {"task": "watch friday night lights"}),
])
def test_ordinary_task_add_forms_keep_working(voice, db, utterance, args):
    result = asyncio.run(voice.run(utterance, [block("task_add", args)]))
    [task] = db.active_tasks_matching(args["task"])
    assert task[2] == args.get("recurrence")
    assert "Added the" in result.text


@pytest.mark.parametrize(("utterance", "blocks"), [
    ("Add reading daily", [block("dismiss")]),
    ("I need to take my vitamins every day", []),
])
def test_control_refusal_and_missing_tool_cannot_speak_success(voice, db, utterance, blocks):
    result = asyncio.run(voice.run(utterance, blocks))
    assert not result.dismissed
    assert result.text and "saved the task" not in result.text
    assert all("saved the task" not in line for line in voice.audio)
    assert not db.active_tasks_matching("reading") and not db.active_tasks_matching("vitamins")


@pytest.mark.parametrize("utterance", ["Add reading", "Could you add reading?"])
def test_unrelated_earlier_round_cannot_change_state_before_task_refusal(voice, db, monkeypatch, utterance):
    actual = brain.registry.call

    def call(name, args):
        if name == "set_timer":
            db.add_task("Synthetic side effect marker")
            return "Synthetic timer set."
        return actual(name, args)

    monkeypatch.setattr(brain.registry, "call", call)
    result = asyncio.run(voice.run(utterance, [
        block("set_timer", {"amount": 4, "unit": "minutes"}), block("tasks_today"),
    ], extra=[Stream([block("task_add", {"task": "reading"})], "Done.")]))
    assert "Nothing was changed" in result.text
    assert not db.active_tasks_matching("side effect marker")
    assert not db.active_tasks_matching("reading")


def test_confirmation_reports_absent_results_by_action(voice, monkeypatch):
    actual = brain.registry.call

    def call(name, args):
        if name == "create_calendar_event":
            pa.propose("Create synthetic first", lambda: None)
            return pa.propose("Create synthetic second", lambda: False)
        return actual(name, args)

    monkeypatch.setattr(brain.registry, "call", call)
    asyncio.run(voice.run("Add synthetic changes to my calendar", [block("create_calendar_event", {
        "summary": "Synthetic", "start_time": "Tomorrow at two", "duration_minutes": 60})]))
    result = asyncio.run(voice.run("Yes", [block("confirm_pending_action", {"approved": True})]))
    assert result.text == ("Could not confirm the result of: Create synthetic first. "
                           "Could not confirm the result of: Create synthetic second.")
    assert voice.audio[-1] == result.text


@pytest.fixture
def app_chat(voice, monkeypatch):
    from fastapi.testclient import TestClient
    import server

    rounds = []
    monkeypatch.setattr(brain, "_claude", lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: rounds.pop(0))))
    monkeypatch.setattr(brain, "effective_tier", lambda: "hokage")
    monkeypatch.setitem(server.app.dependency_overrides, server.get_current_user, lambda: "fake-actor")
    return server.app, TestClient(server.app), rounds


def test_app_replay_across_transports_returns_recorded_reply_once(app_chat, db):
    import json
    from uuid import uuid4

    _, client, rounds = app_chat
    rounds.append(Stream([block("task_add", {"task": "reading"}, "same-event")], ""))
    body = {"message": "Add reading", "channel": "text",
            "conversation_id": str(uuid4()), "request_id": str(uuid4())}
    with client.stream("POST", "/chat/stream", json=body) as streamed:
        frames = [json.loads(line[6:]) for line in streamed.iter_lines() if line.startswith("data: ")]
    assert frames[-1]["text"] == "Added the task: Reading."
    assert client.post("/chat", json=body).json() == {"response": frames[-1]["text"]}
    assert client.post("/chat", json={**body, "message": "Add writing"}).status_code == 409
    assert not rounds
    assert len(db.active_tasks_matching("reading")) == 1
    assert not db.active_tasks_matching("writing")


def test_concurrent_or_unfinished_app_replay_is_not_resubmitted(app_chat, db):
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4
    import server

    _, client, _ = app_chat
    entered, release = Event(), Event()
    body = {"message": "Add reading", "channel": "text",
            "conversation_id": str(uuid4()), "request_id": str(uuid4())}

    def failed_worker(*args, **kwargs):
        db.add_task("Reading")
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("Synthetic failure before acknowledgement")

    with pytest.MonkeyPatch.context() as patch, ThreadPoolExecutor(max_workers=1) as executor:
        patch.setattr(server, "ask_nova", failed_worker)
        first = executor.submit(client.post, "/chat", json=body)
        try:
            assert entered.wait(timeout=5)
            assert client.post("/chat", json=body).status_code == 409
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="Synthetic failure"):
            first.result(timeout=5)
    assert client.post("/chat", json=body).status_code == 409
    assert len(db.active_tasks_matching("reading")) == 1


@pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
@pytest.mark.parametrize("confirmation", ["delivered", "wrong conversation", "wrong receipt", "lost body"])
def test_app_approval_binds_exact_delivery_and_conversation(app_chat, monkeypatch, route, confirmation):
    import json
    from uuid import uuid4
    from fastapi.testclient import TestClient

    app, client, rounds = app_chat
    actual = brain.registry.call
    changes = []

    def call(name, args):
        if name == "create_calendar_event":
            return pa.propose("Create the synthetic event", lambda: changes.append(1) or "Event created.")
        return actual(name, args)

    monkeypatch.setattr(brain.registry, "call", call)
    rounds.extend([
        Stream([block("create_calendar_event", {
            "summary": "Synthetic", "start_time": "Tomorrow at two", "duration_minutes": 60})], ""),
        Stream([block("confirm_pending_action", {"approved": True})], ""),
    ])
    conversation = str(uuid4())
    proposal = {"message": "Add the synthetic event to my calendar", "channel": "text",
                "conversation_id": conversation, "request_id": str(uuid4())}
    receipt = None
    if confirmation == "lost body":
        async def lose_body(scope, receive, send):
            async def fail(message):
                if message["type"] == "http.response.body":
                    raise OSError("Synthetic disconnect")
                await send(message)

            await app(scope, receive, fail)

        TestClient(lose_body, raise_server_exceptions=False).post(route, json=proposal)
    elif route == "/chat/stream":
        with client.stream("POST", route, json=proposal) as streamed:
            frames = [json.loads(line[6:]) for line in streamed.iter_lines() if line.startswith("data: ")]
        assert all("proposal_receipt" not in frame for frame in frames[:-1])
        assert frames[-1]["text"] == "Create the synthetic event?"
        receipt = frames[-1]["proposal_receipt"]
    else:
        response = client.post(route, json=proposal).json()
        assert response["response"] == "Create the synthetic event?"
        receipt = response["proposal_receipt"]
    assert not changes
    reply = {"message": "Yes", "channel": "text", "request_id": str(uuid4()),
             "conversation_id": str(uuid4()) if confirmation == "wrong conversation" else conversation,
             "proposal_receipt": str(uuid4()) if confirmation == "wrong receipt" else receipt}
    second_client = TestClient(app)
    answer = second_client.post("/chat", json=reply).json()["response"]
    if confirmation == "delivered":
        assert answer == "Event created." and changes == [1]
        assert second_client.post("/chat", json=reply).json()["response"] == answer
        assert changes == [1]
    else:
        assert "no matching confirmation" in answer and not changes


def test_legacy_chat_keeps_conversation_but_refuses_unidentifiable_mutation(app_chat, db):
    _, client, rounds = app_chat
    rounds.extend([
        Stream([], "Hello."),
        Stream([block("task_add", {"task": "reading"})], "Done."),
    ])
    assert client.post("/chat", json={"message": "Hi"}).json() == {"response": "Hello."}
    answer = client.post("/chat", json={"message": "Add reading"}).json()["response"]
    assert "send this change safely" in answer
    assert not db.active_tasks_matching("reading")


def test_prestart_completion_explains_the_actual_first_day(voice, db, monkeypatch):
    from datetime import datetime
    import lists_tasks

    captured = datetime(2026, 11, 25, 12)
    task = db.add_task("Reading", "daily", now=captured, start_date="2026-11-26")
    monkeypatch.setattr(lists_tasks, "complete_task", lambda task_id, day=None: db.complete_task(task_id, captured))
    result = asyncio.run(voice.run("Complete reading", [block("task_complete", {"task": "reading"})]))
    assert result.text == "Reading starts 2026-11-26. Nothing was marked done."
    assert voice.audio == [result.text]
    assert not db.task_completed_on(task, "2026-11-25")


def test_confirmation_without_tool_result_cannot_speak_success(voice, monkeypatch):
    def propose(name, args):
        return pa.propose("Create the synthetic event", lambda: pytest.fail("Ran without confirmation"))

    monkeypatch.setattr(brain.registry, "call", propose)
    asyncio.run(voice.run("Add a synthetic event to my calendar", [block("create_calendar_event", {
        "summary": "Synthetic", "start_time": "Tomorrow at two", "duration_minutes": 60})]))
    result = asyncio.run(voice.run("Yes", [], draft="Done. I created the event."))
    assert result.text == NO_ACTION
    assert voice.audio[-1] in result.text


def test_cancelled_app_worker_leaves_an_uncertain_reservation(app_chat, db, monkeypatch):
    from uuid import uuid4
    import server

    _, client, _ = app_chat
    body = {"message": "Add reading", "channel": "text",
            "conversation_id": str(uuid4()), "request_id": str(uuid4())}

    def cancelled_worker(*args, cancelled, **kwargs):
        db.add_task("Reading")
        cancelled.set()
        return brain.TurnResult(text="")

    monkeypatch.setattr(server, "ask_nova", cancelled_worker)
    assert client.post("/chat", json=body).json() == {"response": ""}
    assert client.post("/chat", json=body).status_code == 409
    assert len(db.active_tasks_matching("reading")) == 1
