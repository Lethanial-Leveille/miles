"""The note that tells her the history is old.

Regression for Sep 28 2026. At 1:46 PM she said his last workout was six days
old; at 6:57 PM, two minutes after he saved one, she repeated it without
calling a tool, because the history carries no times and read as a moment ago.
"""

from datetime import datetime, timedelta

import brain

NOW = datetime(2026, 9, 28, 18, 57)
MESSAGES = [{"role": "user", "content": "What did I do at the gym?"},
            {"role": "assistant", "content": "Upper A on Tuesday, six days ago."},
            {"role": "user", "content": "How did my workout go today?"}]


def test_a_gap_of_hours_is_said_on_the_newest_message():
    out = brain._with_gap_note(MESSAGES, NOW - timedelta(hours=5, minutes=20), now=NOW)
    assert out[:-1] == MESSAGES[:-1], "only the newest message changes"
    assert out[-1]["content"].startswith("How did my workout go today?")
    assert "ended 5 hours ago" in out[-1]["content"]
    assert "check again" in out[-1]["content"]


def test_a_follow_up_gets_no_note():
    out = brain._with_gap_note(MESSAGES, NOW - timedelta(seconds=20), now=NOW)
    assert out == MESSAGES


def test_no_reply_yet_gets_no_note():
    assert brain._with_gap_note(MESSAGES, None, now=NOW) == MESSAGES


def test_the_message_list_is_not_changed_in_place():
    before = [dict(m) for m in MESSAGES]
    brain._with_gap_note(MESSAGES, NOW - timedelta(hours=3), now=NOW)
    assert MESSAGES == before


def test_gap_words():
    assert brain._gap_words(45) == "45 minutes"
    assert brain._gap_words(5 * 60 + 20) == "5 hours"
    assert brain._gap_words(3 * 24 * 60) == "3 days"
