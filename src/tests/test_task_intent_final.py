"""The final fix round for the request intent step. Offline, fake names only."""

import asyncio
from datetime import date

import pytest

import brain
import task_intent
from task_intent import Intent
from test_task_intent import Stream, block, turn  # noqa: F401  (turn is a fixture)

WEDNESDAY = date(2026, 11, 25)


# ── F8: a read that holds two verbs is not "more than one change" ──

def test_a_read_with_two_verbs_gets_a_normal_answer_when_the_model_says_no_change(turn, db):
    db.add_list_item(db.get_or_create_list("grocery"), "milk")
    said = "Can you show my grocery list and check whether milk is on it?"
    result = asyncio.run(turn.run(said, [block("list_show", {"list_name": "grocery"})],
                                  intent=Intent("no_change"), more=[Stream([], "Yes, milk is on it.")]))
    assert "Nothing was changed" not in result.text and "Yes, milk is on it." in result.text


def test_two_real_changes_are_still_refused_when_the_model_names_a_change():
    verdict = task_intent.validate(Intent("list_add", "eggs"), "add eggs to my grocery list and check off milk",
                                   WEDNESDAY)
    assert verdict.kind == "refuse" and "one change at a time" in verdict.text


# ── F1: a question never offers remember ──

def test_a_question_turn_does_not_offer_remember(turn, db):
    asyncio.run(turn.run("What do you remember about my commute?",
                         [], draft="You bike twenty minutes.", intent=Intent("no_change")))
    names = {schema["name"] for schema in turn.offered[0]}
    assert "remember" not in names and "tasks_today" in names


def test_a_remember_request_that_is_not_a_question_still_offers_remember(turn, db):
    asyncio.run(turn.run("remember that my cat is named Sam", [], draft="Okay.", intent=Intent("no_change")))
    assert "remember" in {schema["name"] for schema in turn.offered[0]}


def test_a_remember_call_on_a_question_turn_is_refused(turn, db, monkeypatch):
    ran = []
    monkeypatch.setattr(brain.registry, "call", lambda name, args: ran.append(name) or "ok")
    result = asyncio.run(turn.run("What do you remember about my commute?",
                                  [block("remember", {"content": "Commute is long"})], intent=Intent("no_change")))
    assert ran == [] and "Nothing was changed" in result.text


# ── F10: a list name that holds a preposition is read whole ──

def test_a_list_name_with_a_preposition_inside_it_is_read_whole():
    verdict = task_intent.validate(Intent("list_add", "tape"), "add tape to my back to school list", WEDNESDAY)
    assert verdict.kind == "act"
    assert verdict.gate.permits("list_add", {"item": "tape", "list_name": "back to school"})
    assert not verdict.gate.permits("list_add", {"item": "tape", "list_name": "school"})
    assert not verdict.gate.permits("list_add", {"item": "tape", "list_name": "grocery"})


@pytest.mark.parametrize("said, item, name", [
    ("put eggs on the grocery list", "eggs", "grocery"),
    ("add milk to my hardware store list", "milk", "hardware store"),
    ("add things to remember to my errands list", "things to remember", "errands"),
])
def test_the_list_name_comes_after_the_item(said, item, name):
    assert task_intent._spoken_list(task_intent._words(said), item) == name


# ── V2: the stored item is looked up as he said it ──

def test_a_punctuated_item_is_found_by_its_own_words(db):
    db.add_list_item(db.get_or_create_list("grocery"), "half-and-half")
    verdict = task_intent.validate(Intent("list_check", "half-and-half"), "check off half-and-half", WEDNESDAY)
    assert verdict.kind == "act"
    assert verdict.gate.permits("list_check", {"item": "half-and-half"})


# ── V3: ordinary reports are accepted ──

@pytest.mark.parametrize("said", ["I finally finished reading", "I've really finished reading",
                                  "I just finished reading", "I finished reading", "mark reading done",
                                  "reading is done", "I finished reading finally"])
def test_an_ordinary_completion_report_is_accepted(db, said):
    db.add_task("Reading", "daily")
    verdict = task_intent.validate(Intent("task_complete", "reading"), said, WEDNESDAY)
    assert verdict.kind == "act", verdict.text


@pytest.mark.parametrize("said", ["I almost finished reading", "I didn't finish reading", "did I finish reading",
                                  "I will finish reading"])
def test_a_report_that_is_not_a_completion_is_still_refused(db, said):
    db.add_task("Reading", "daily")
    assert task_intent.validate(Intent("task_complete", "reading"), said, WEDNESDAY).kind != "act"


# ── F6: the verb must be the task's own act ──

def test_buying_vitamins_does_not_complete_taking_them(db):
    db.add_task("Take vitamins", "daily")
    verdict = task_intent.validate(Intent("task_complete", "vitamins"), "I bought my vitamins", WEDNESDAY)
    assert verdict.kind == "refuse"
    for said in ("I took my vitamins", "mark vitamins done", "I finished my vitamins"):
        assert task_intent.validate(Intent("task_complete", "vitamins"), said, WEDNESDAY).kind == "act", said


def test_a_task_without_a_verb_name_keeps_its_old_reports(db):
    db.add_task("Vitamins", "daily")
    db.add_task("Pay rent", None)
    assert task_intent.validate(Intent("task_complete", "vitamins"), "I took my vitamins", WEDNESDAY).kind == "act"
    assert task_intent.validate(Intent("task_complete", "rent"), "I paid rent", WEDNESDAY).kind == "act"
    assert task_intent.validate(Intent("task_complete", "rent"), "I called rent", WEDNESDAY).kind == "refuse"


# ── E8: a title with a hypothetical word is still a plain report ──

def test_a_title_with_a_hypothetical_word_does_not_block_a_completion(db):
    db.add_task("Call when home", None)
    verdict = task_intent.validate(Intent("task_complete", "call when home"), "I finished call when home", WEDNESDAY)
    assert verdict.kind == "act", verdict.text
    assert task_intent.validate(Intent("task_complete", "call when home"),
                                "did I finish call when home", WEDNESDAY).kind != "act"


# ── E9: "anymore" means stop, not skip today ──

@pytest.mark.parametrize("said", ["I don't track reading anymore", "cancel reading, I don't do it anymore",
                                  "stop reading, not anymore", "I never do reading anymore"])
def test_a_negated_anymore_is_a_stop_not_a_skip(said):
    assert task_intent._cancel_scope(said, "daily", "Reading") == "archive"


def test_not_for_good_is_still_a_skip():
    assert task_intent._cancel_scope("stop reading, not for good", "daily", "Reading") == "skip"


# ── F13: a working day count is still a relative date ──

@pytest.mark.parametrize("phrase", ["follow up in four business days", "follow up in two working days",
                                    "follow up in three calendar weeks"])
def test_a_business_day_count_is_uncaptured_timing(phrase):
    assert task_intent._uncaptured_timing(Intent("task_add", phrase), f"add {phrase}")


# ── E10: compound timer lengths ──

@pytest.mark.parametrize("words, seconds", [
    ("an hour and a half", 5400), ("one hour thirty minutes", 5400), ("two and a half hours", 9000),
    ("1 hour 15 minutes", 4500), ("ten minutes", 600), ("half an hour", 1800), ("ninety seconds", 90),
    ("twenty five minutes", 1500), ("for 5 minutes", 300)])
def test_compound_timer_lengths_are_read(words, seconds):
    assert task_intent.spoken_seconds(words) == seconds


@pytest.mark.parametrize("words", ["", "soon", "ten", "ten minutes later", "a half", "banana minutes"])
def test_other_words_are_not_a_timer_length(words):
    assert task_intent.spoken_seconds(words) is None


def test_a_compound_timer_request_goes_through(turn, db):
    said = "set a timer for an hour and a half"
    result = asyncio.run(turn.run(said, [block("set_timer", {"amount": 90, "unit": "minutes"})],
                                  intent=Intent("timer", "timer", "an hour and a half")))
    assert "Nothing was changed" not in result.text


# ── E7: a compact clock time is understood ──

@pytest.mark.parametrize("timing", ["at 5pm", "at 5 pm", "at 5:30pm", "5pm"])
def test_a_compact_clock_time_is_accepted(timing):
    from datetime import datetime, timedelta
    now = datetime.now()
    minute = 30 if ":30" in timing else 0
    when = (now + timedelta(days=1)).replace(hour=17, minute=minute, second=0, microsecond=0)
    assert task_intent._due_matches(when.isoformat(), timing, now)
    assert not task_intent._due_matches(when.replace(hour=16).isoformat(), timing, now)
