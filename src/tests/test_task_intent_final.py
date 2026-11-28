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

