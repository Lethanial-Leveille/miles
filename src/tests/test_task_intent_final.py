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
