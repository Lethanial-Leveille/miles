"""The intent call is told his active task titles, so a task named like an
everyday word ("Reading") is read as a task when he gives a command."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

import task_intent
from task_intent import Intent
from test_task_intent import REAL_REQUEST  # the real function, before conftest fakes it


def _capture(monkeypatch):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(task_intent, "parse_intent", lambda message: Intent("task_archive", "reading"))
    return seen, SimpleNamespace(messages=SimpleNamespace(create=create))


def test_the_active_task_titles_are_in_the_intent_prompt(db, monkeypatch):
    now = datetime.now()
    db.add_task("Reading", "daily", now=now)
    db.add_task("Water the cactus", None, now=now, scheduled_for=db.task_day(now))
    seen, client = _capture(monkeypatch)
    asyncio.run(REAL_REQUEST(client, "Stop reading for good"))
    assert "Reading (daily)" in seen["system"] and "Water the cactus (one time)" in seen["system"]
    assert "everyday word" in seen["system"]
    assert seen["messages"] == [{"role": "user", "content": "Stop reading for good"}]


def test_no_tasks_leaves_the_prompt_as_it_was(db, monkeypatch):
    seen, client = _capture(monkeypatch)
    asyncio.run(REAL_REQUEST(client, "Stop reading for good"))
    assert "His active tasks" not in seen["system"]


def test_an_archived_task_is_not_listed(db, monkeypatch):
    now = datetime.now()
    db.add_task("Reading", "daily", now=now)
    db.add_task("Stretching", "daily", now=now)
    [(task_id, _text, _rec)] = db.active_tasks_matching("reading")
    db.archive_task(task_id)
    seen, client = _capture(monkeypatch)
    asyncio.run(REAL_REQUEST(client, "hello"))
    assert "Stretching (daily)" in seen["system"] and "Reading (daily)" not in seen["system"]


def test_a_failure_reading_the_tasks_does_not_stop_the_call(db, monkeypatch):
    def boom(_words):
        raise RuntimeError("database busy")
    monkeypatch.setattr(task_intent.database, "active_tasks_matching", boom)
    seen, client = _capture(monkeypatch)
    asyncio.run(REAL_REQUEST(client, "Stop reading for good"))
    assert "His active tasks" not in seen["system"] and seen["system"].startswith(task_intent._SYSTEM)


def test_the_list_is_capped(db, monkeypatch):
    now = datetime.now()
    for number in range(60):
        db.add_task(f"Task number {number}", "daily", now=now)
    seen, client = _capture(monkeypatch)
    asyncio.run(REAL_REQUEST(client, "hello"))
    assert seen["system"].count("(daily)") == 40
