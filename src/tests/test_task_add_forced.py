"""A validated add does not depend on the main model choosing to call the tool.
Offline, fake model, fake names."""

import asyncio

import pytest

import brain
from task_intent import Intent
from test_task_intent import Stream, block, turn  # noqa: F401  (turn is a fixture)

TASKS = [("reading", "Reading"), ("stretching", "Stretching")]


@pytest.fixture
def seen(turn, monkeypatch):
    """The keyword arguments of every main model call this test makes."""
    calls = []
    real = brain.claude.messages.stream
    monkeypatch.setattr(brain.claude.messages, "stream", lambda **kw: calls.append(kw) or real(**kw))
    return calls


def _daily_rows(db, words):
    return [row for row in db.active_tasks_matching(words) if row[2] == "daily"]


# The four things the main model was seen doing on "add reading every day".
BEHAVIOURS = {
    "calls the tool correctly": lambda word: ([block("task_add", {"task": word, "recurrence": "daily"})], ""),
    "text only claiming success": lambda word: ([], f"Added the daily task: {word.title()}."),
    "text saying it has no tool": lambda word: ([], f"I don't have a tool to add {word} as a daily task."),
    "wrong recurrence": lambda word: ([block("task_add", {"task": word, "recurrence": "weekly"})], "Done."),
    "missing recurrence": lambda word: ([block("task_add", {"task": word})], "Done."),
    "other tool": lambda word: ([block("tasks_today")], "That task is already on your list."),
}


@pytest.mark.parametrize("word, title", TASKS)
@pytest.mark.parametrize("behaviour", list(BEHAVIOURS))
def test_a_validated_daily_add_always_creates_the_task_once(turn, seen, db, word, title, behaviour):
    blocks, draft = BEHAVIOURS[behaviour](word)
    result = asyncio.run(turn.run(f"Add {word} every day", blocks, draft=draft))
    assert result.text == f"Added the daily task: {title}."
    assert len(_daily_rows(db, word)) == 1
    assert len(db.active_tasks_matching(word)) == 1


@pytest.mark.parametrize("word, title", TASKS)
def test_the_first_round_is_forced_to_the_one_validated_tool(turn, seen, db, word, title):
    asyncio.run(turn.run(f"Add {word} every day", [block("task_add", {"task": word, "recurrence": "daily"})]))
    assert seen[0]["tool_choice"] == {"type": "tool", "name": "task_add"}


def test_an_ordinary_turn_is_not_forced(turn, seen, db):
    asyncio.run(turn.run("good morning", [], draft="Good morning."))
    assert "tool_choice" not in seen[0]


def test_a_flagged_turn_with_no_validated_tool_is_not_forced(turn, seen, db):
    asyncio.run(turn.run("What do you remember about my commute?", [], draft="Twenty minutes.",
                         intent=Intent("no_change")))
    assert "tool_choice" not in seen[0]


@pytest.mark.parametrize("said, intent, blocks", [
    ("I finished reading", Intent("task_complete", "reading"), [block("task_complete", {"task": "reading"})]),
    ("archive reading", Intent("task_archive", "reading"), [block("archive_task", {"task": "reading"})]),
    ("skip reading today", Intent("task_skip_today", "reading"), [block("task_skip_today", {"task": "reading"})]),
    ("add eggs to my grocery list", Intent("list_add", "eggs"),
     [block("list_add", {"item": "eggs", "list_name": "grocery"})]),
    ("set a timer for 10 minutes", Intent("timer", "timer", "10 minutes"),
     [block("set_timer", {"amount": 10, "unit": "minutes"})]),
])
def test_every_validated_single_tool_gate_is_forced(turn, seen, db, said, intent, blocks):
    db.add_task("reading", "daily")
    asyncio.run(turn.run(said, blocks, intent=intent))
    assert seen[0]["tool_choice"] == {"type": "tool", "name": blocks[0].name}


def test_a_wrong_call_for_a_stop_is_refused_not_replaced(turn, db):
    task = db.add_task("reading")
    db.add_task("someone else")
    result = asyncio.run(turn.run("archive reading", [block("archive_task", {"task": "someone else"})],
                                  intent=Intent("task_archive", "reading")))
    assert "Nothing was changed" in result.text
    assert db.get_task(task)["archived_at"] is None


def test_the_started_day_comes_from_the_plan_not_the_model(turn, db):
    result = asyncio.run(turn.run("Add reading every day starting Monday",
                                  [block("task_add", {"task": "reading", "recurrence": "daily",
                                                      "start_date": "2030-01-01"})]))
    [(task_id, _, _)] = _daily_rows(db, "reading")
    assert db.get_task(task_id)["start_date"] != "2030-01-01"
    assert "starting" in result.text


@pytest.mark.parametrize("said, blocks, text", [
    ("add water plants", [], "Added the task: Water plants."),
    ("add reading", [block("task_add", {"task": "reading"})], "Added the task: Reading."),
])
def test_one_time_adds_still_work(turn, db, said, blocks, text):
    result = asyncio.run(turn.run(said, blocks))
    assert result.text == text
    assert db.active_tasks_matching(said.removeprefix("add "))[0][2] is None


def test_water_the_cactus_is_added_as_the_model_titled_it(turn, db):
    result = asyncio.run(turn.run("Add water to the cactus", [block("task_add", {"task": "Water the cactus"})],
                                  intent=Intent("task_add", "water the cactus")))
    assert result.text == "Added the task: Water the cactus."
    assert len(db.active_tasks_matching("cactus")) == 1


def test_a_title_that_changes_meaning_is_refused_before_any_model_round(turn, db):
    result = asyncio.run(turn.run("Add water the cactus", [], intent=Intent("task_add", "water the garden")))
    assert result.text.startswith("Nothing was changed.")
    assert not db.active_tasks_matching("garden")


# ── No duplicates (owner decision, F1) ──

def test_adding_the_same_daily_task_twice_writes_it_once(turn, db):
    first = asyncio.run(turn.run("Add reading every day", [block("task_add", {"task": "reading", "recurrence": "daily"})]))
    assert first.text == "Added the daily task: Reading."
    second = asyncio.run(turn.run("Add reading every day", [block("task_add", {"task": "reading", "recurrence": "daily"})]))
    assert second.text == "Nothing was changed. Reading is already a daily task on your list."
    assert len(db.active_tasks_matching("reading")) == 1


def test_adding_the_same_one_time_task_twice_writes_it_once(turn, db):
    asyncio.run(turn.run("Add water plants", [block("task_add", {"task": "water plants"})]))
    again = asyncio.run(turn.run("Add water plants", [block("task_add", {"task": "water plants"})]))
    assert again.text == "Nothing was changed. Water plants is already a task on your list."
    assert len(db.active_tasks_matching("water plants")) == 1


def test_the_same_title_by_meaning_is_still_the_same_task(turn, db):
    asyncio.run(turn.run("Add water the cactus", [block("task_add", {"task": "water the cactus"})]))
    again = asyncio.run(turn.run("Add water to the cactus", []))
    assert "already" in again.text
    assert len(db.active_tasks_matching("cactus")) == 1


def test_a_daily_task_and_a_one_time_task_with_one_title_are_not_duplicates(turn, db):
    asyncio.run(turn.run("Add reading", [block("task_add", {"task": "reading"})]))
    daily = asyncio.run(turn.run("Add reading every day", [block("task_add", {"task": "reading", "recurrence": "daily"})]))
    assert daily.text == "Added the daily task: Reading."
    assert len(db.active_tasks_matching("reading")) == 2


def test_a_different_title_is_not_a_duplicate(turn, db):
    asyncio.run(turn.run("Add reading every day", [block("task_add", {"task": "reading", "recurrence": "daily"})]))
    other = asyncio.run(turn.run("Add reading the news every day",
                                 [block("task_add", {"task": "reading the news", "recurrence": "daily"})]))
    assert other.text == "Added the daily task: Reading the news."


# ── The code adds the task itself only for "add" or "create" (owner decision, F2) ──

def test_i_need_to_is_not_added_by_the_code_when_the_models_call_is_missing(turn, db):
    result = asyncio.run(turn.run("I need to buy milk", [], draft="Added the task: Buy milk."))
    assert result.text == "Nothing was changed. I couldn't do that. Please try again."
    assert db.active_tasks_matching("buy milk") == []


def test_i_need_to_is_not_added_by_the_code_when_the_models_call_is_wrong(turn, db):
    result = asyncio.run(turn.run("I need to buy milk", [block("task_add", {"task": "buy bread"})], draft="Done."))
    assert result.text == "Nothing was changed. That is not the change you asked for."
    assert db.active_tasks_matching("buy") == []


def test_i_need_to_still_works_when_the_models_call_is_valid(turn, db):
    result = asyncio.run(turn.run("I need to buy milk", [block("task_add", {"task": "buy milk"})]))
    assert result.text == "Added the task: Buy milk."
    assert len(db.active_tasks_matching("buy milk")) == 1


@pytest.mark.parametrize("said", ["Add buy milk", "Create buy milk"])
def test_an_explicit_add_or_create_is_still_added_by_the_code(turn, db, said):
    result = asyncio.run(turn.run(said, [], draft="Added the task: Buy milk."))
    assert result.text == "Added the task: Buy milk."
    assert len(db.active_tasks_matching("buy milk")) == 1


# ── The model repeats the validated timing inside the title (seen live, 3 of 3) ──

def test_i_need_to_with_the_timing_repeated_in_the_title_is_added_with_the_clean_title(turn, db):
    result = asyncio.run(turn.run("I need to take my vitamins every day",
                                  [block("task_add", {"task": "take my vitamins every day", "recurrence": "daily"})]))
    assert result.text == "Added the daily task: Take my vitamins."
    [row] = db.active_tasks_matching("vitamins")
    assert row[1] == "Take my vitamins" and row[2] == "daily"


def test_an_explicit_add_with_the_timing_repeated_in_the_title_is_stored_clean(turn, db):
    result = asyncio.run(turn.run("Add reading every day",
                                  [block("task_add", {"task": "reading every day", "recurrence": "daily"})]))
    assert result.text == "Added the daily task: Reading."
    [row] = db.active_tasks_matching("reading")
    assert row[1] == "Reading"


@pytest.mark.parametrize("title", ["take my vitamins at night", "take my vitamins every week", "take my vitamins daily"])
def test_other_words_after_the_title_are_still_refused(turn, db, title):
    result = asyncio.run(turn.run("I need to take my vitamins every day",
                                  [block("task_add", {"task": title, "recurrence": "daily"})]))
    assert result.text.startswith("Nothing was changed.")
    assert db.active_tasks_matching("vitamins") == []
