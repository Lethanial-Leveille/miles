"""An answer to the assistant's own question about a change gets the intent step.
Offline, fake model, fake names only."""

import asyncio

import pytest

import brain
import task_intent
from task_intent import Intent
from test_task_intent import REAL_REQUEST, block, turn  # noqa: F401  (turn is a fixture)

ASK = "Which one, reading or stretching?"
SKIP_OR_STOP = "Skip today or stop it for good?"


@pytest.fixture
def tasks(db):
    return db.add_task("Reading", "daily"), db.add_task("Stretching", "daily")


def _ask_which(turn, text="Can you cancel that actually", question=ASK, **kw):
    return asyncio.run(turn.run(text, [], draft=question, **kw))


def _archived(db, task_id):
    return db.get_task(task_id)["archived_at"] is not None


def test_tonights_transcript_asks_skip_or_stop_and_then_archives_through_the_tool(turn, db, tasks):
    reading, stretching = tasks
    first = _ask_which(turn)
    assert first.text == ASK
    second = asyncio.run(turn.run("Stretching", [], draft="Stopped the daily task: Stretching."))
    assert not _archived(db, stretching) and not _archived(db, reading)
    assert second.text == f"Nothing was changed. Stretching is a daily task. {SKIP_OR_STOP}"
    assert not any(text.startswith("Stopped") for text in turn.audio)
    third = asyncio.run(turn.run("stop it for good", [block("archive_task", {"task": "stretching"})],
                                 intent=Intent("task_archive", "Stretching")))
    assert _archived(db, stretching) and not _archived(db, reading)
    assert "Stretching" in third.text


@pytest.mark.parametrize("answer", ["Stretching", "stretching please", "yeah stretching", "the second one"])
@pytest.mark.parametrize("channel, device", [("voice", "pi"), ("text", "app")])
def test_variations_resolve_to_stretching_and_ask(turn, db, tasks, answer, channel, device):
    _ask_which(turn, channel=channel, device=device)
    result = asyncio.run(turn.run(answer, [], draft="Stopped it.", channel=channel, device=device))
    assert result.text.endswith(SKIP_OR_STOP)
    assert not any(_archived(db, task_id) for task_id in tasks)


def test_the_first_one_names_reading(turn, db, tasks):
    _ask_which(turn)
    result = asyncio.run(turn.run("the first one", [], draft="Okay."))
    assert result.text.startswith("Nothing was changed. Reading is a daily task.")


def test_a_cancel_of_a_one_time_task_archives_after_the_answer(turn, db):
    water = db.add_task("Water plants")
    db.add_task("Call the bank")
    _ask_which(turn, question="Which one, water plants or call the bank?")
    result = asyncio.run(turn.run("water plants", [block("archive_task", {"task": "water plants"})]))
    assert _archived(db, water) and "Water plants" in result.text


def test_both_writes_nothing(turn, db, tasks):
    _ask_which(turn)
    result = asyncio.run(turn.run("both", [], draft="Stopped both."))
    assert result.text == "Nothing was changed. Please ask for one change at a time."
    assert not any(_archived(db, task_id) for task_id in tasks)


@pytest.mark.parametrize("answer", ["what's the weather", "never mind"])
def test_an_unrelated_answer_is_answered_normally(turn, db, tasks, answer):
    _ask_which(turn)
    result = asyncio.run(turn.run(answer, [block("set_timer", {"amount": 5, "unit": "minutes"})],
                                  draft="It is sunny."))
    names = {schema["name"] for schema in turn.offered[-1]}
    assert result.text == "It is sunny." or result.text.startswith("Nothing was changed")
    assert "task_add" not in names and "archive_task" not in names and "set_timer" not in names
    assert not any(_archived(db, task_id) for task_id in tasks)


def test_a_question_older_than_five_minutes_triggers_nothing(turn, db, tasks, monkeypatch):
    _ask_which(turn)
    now = brain._exchange_clock()
    monkeypatch.setattr(brain, "_exchange_clock", lambda: now + brain.FOLLOWUP_WINDOW_S + 1)
    calls = len(turn.intent_calls)
    result = asyncio.run(turn.run("Stretching", [], draft="Stretching is a good habit."))
    assert turn.intent_calls[calls:] == [] and result.text == "Stretching is a good habit."


def test_a_question_inside_five_minutes_still_counts(turn, db, tasks, monkeypatch):
    _ask_which(turn)
    now = brain._exchange_clock()
    monkeypatch.setattr(brain, "_exchange_clock", lambda: now + brain.FOLLOWUP_WINDOW_S - 1)
    assert asyncio.run(turn.run("Stretching", [])).text.endswith(SKIP_OR_STOP)


def test_a_question_in_ordinary_chat_triggers_nothing(turn, db, tasks):
    asyncio.run(turn.run("good morning", [], draft="Good morning. Which one do you like, tea or coffee?"))
    result = asyncio.run(turn.run("Tea", [], draft="Nice choice."))
    assert turn.intent_calls == [] and result.text == "Nice choice."


def test_a_statement_reply_is_not_a_question(turn, db, tasks):
    asyncio.run(turn.run("Can you cancel that actually", [], draft="Tell me which one."))
    asyncio.run(turn.run("Stretching", [], draft="Okay."))
    assert len(turn.intent_calls) == 1


def test_another_device_does_not_answer_the_question(turn, db, tasks):
    _ask_which(turn, device="app", channel="text")
    asyncio.run(turn.run("Stretching", [], draft="Nice.", device="pi", channel="voice"))
    assert len(turn.intent_calls) == 1


def test_the_model_gets_the_earlier_request_the_question_and_the_answer():
    seen = {}

    class Client:
        class messages:
            @staticmethod
            def create(**kwargs):
                seen.update(kwargs)
                return None

    followup = task_intent.FollowUp("cancel that", ASK, "Stretching")
    asyncio.run(REAL_REQUEST(Client, "Stretching", None, followup=followup))
    content = seen["messages"][0]["content"]
    assert "cancel that" in content and ASK in content and "Stretching" in content
    assert "no_change" in seen["system"] and len(seen["messages"]) == 1


def test_the_journal_line_says_followup(turn, db, tasks, capsys):
    _ask_which(turn, device="app", channel="text")
    asyncio.run(turn.run("Stretching", [], device="app", channel="text"))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("App turn")]
    assert 'followup="no"' in lines[0] and 'followup="yes"' in lines[1]


@pytest.mark.parametrize("said, intent, blocks", [
    ("add reading every day", Intent("task_add", "reading", "every day"),
     [block("task_add", {"task": "reading", "recurrence": "daily"})]),
    ("add water to the cactus", Intent("task_add", "water the cactus"),
     [block("task_add", {"task": "Water the cactus"})]),
    ("add eggs to my grocery list", Intent("list_add", "eggs"),
     [block("list_add", {"item": "eggs", "list_name": "grocery"})]),
    ("set a timer for ten minutes", Intent("timer", "timer", "ten minutes"),
     [block("set_timer", {"amount": 10, "unit": "minutes"})]),
])
def test_ordinary_requests_after_a_question_still_run(turn, db, said, intent, blocks):
    db.add_task("Stretching", "daily")
    _ask_which(turn)
    result = asyncio.run(turn.run(said, blocks, intent=intent))
    assert not result.text.startswith("Nothing was changed")


def test_a_backslash_in_a_model_written_target_does_not_crash_the_turn():
    """The target is model text. It must be placed in the rebuilt command as is,
    never read as a regular expression template."""
    follow = task_intent.FollowUp("Can you cancel that actually", "Which one?", "stretching")
    command = task_intent._followup_command(follow, Intent("task_archive", r"stretch\1 \g<0>"))
    assert command == r"Can you cancel stretch\1 \g<0> actually"
