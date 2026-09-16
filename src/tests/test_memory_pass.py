"""The background pass that notices facts. Nothing here reaches Claude."""

from datetime import datetime
from types import SimpleNamespace

import pytest

import database
import memory_pass
import memory_tool


def _call(**inputs):
    return SimpleNamespace(type="tool_use", name="remember", input=inputs)


@pytest.fixture
def run_pass(db, monkeypatch):
    """Runs one pass against a throwaway database with a scripted reply."""
    sent = {}

    def go(*blocks, text="Charley is a girl, can you move her lesson to 3"):
        def create(**request):
            sent.update(request)
            return SimpleNamespace(content=list(blocks))
        monkeypatch.setattr(memory_pass, "client",
                            SimpleNamespace(messages=SimpleNamespace(create=create)))
        database.save_message("user", "earlier question")
        database.save_message("assistant", "earlier answer")
        database.save_message("user", text)
        database.save_message("assistant", "Moved it.")
        return memory_pass.notice(text, datetime(2026, 9, 15, 23, 29)), sent
    return go


def _pending():
    return [row[1] for row in database.get_pending_memories()]


def test_what_it_notices_waits_for_review_even_if_called_asked(run_pass):
    stored, _ = run_pass(_call(content="Charley is a girl.", certainty="asked"))
    assert stored == ["Charley is a girl."]
    assert _pending() == ["Charley is a girl."]


def test_it_never_supersedes(run_pass, db):
    """A supersede activates the replacement at once, skipping review."""
    keep = database.save_memory("Charley is a boy.", source="seed")
    run_pass(_call(content="Charley is a girl.", supersedes=keep))
    assert database.memory_status(keep) == "active"
    assert _pending() == ["Charley is a girl."]


def test_empty_calls_are_dropped(run_pass):
    stored, _ = run_pass(_call(content=""), _call(certainty="inferred"),
                         _call(content="  Andrew's test is Saturday.  "))
    assert stored == ["Andrew's test is Saturday."]


def test_at_most_three_from_one_message(run_pass):
    stored, _ = run_pass(*[_call(content=f"Fact {i}.") for i in range(5)])
    assert stored == ["Fact 0.", "Fact 1.", "Fact 2."]


def test_the_same_fact_is_not_queued_twice(run_pass):
    database.save_memory("Charley is a girl.", source="implicit", status="pending")
    stored, _ = run_pass(_call(content="charley is a girl."), _call(content="Charley is a girl."))
    assert stored == []
    assert _pending() == ["Charley is a girl."]


def test_a_temporary_fact_keeps_its_end_date(run_pass, db):
    run_pass(_call(content="Andrew's test is Saturday.", temporary=True, until="2026-09-19"))
    row = database.get_pending_memories()[0]
    conn = __import__("sqlite3").connect(database.DB_PATH)
    volatile, until = conn.execute(
        "SELECT volatile, references_date FROM memories WHERE id = ?", (row[0],)).fetchone()
    assert (volatile, until) == (1, "2026-09-19")


def test_only_the_latest_message_is_the_source(run_pass):
    """The earlier turns are labelled reference only, and the latest message is
    given once, after them, with the date it was sent."""
    _, sent = run_pass()
    prompt = sent["messages"][0]["content"]
    before, latest = prompt.split("His latest message, sent Tuesday, September 15, 2026:")
    assert "earlier question" in before and "Moved it." not in prompt
    assert latest.strip() == "Charley is a girl, can you move her lesson to 3"
    assert sent["tools"][0]["name"] == "remember" and len(sent["tools"]) == 1
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_failure_never_reaches_the_turn(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("API down")
    monkeypatch.setattr(memory_pass, "notice", boom)
    memory_pass.notice_later("hello", datetime.now()).join(timeout=2)


def test_switched_off_it_does_nothing(monkeypatch):
    monkeypatch.setattr(memory_pass, "MEMORY_PASS", False)
    monkeypatch.setattr(memory_pass, "notice", lambda *a: pytest.fail("ran while off"))
    assert memory_pass.notice_later("hello", datetime.now()) is None


def test_the_turn_starts_it_only_for_him():
    import inspect
    import brain
    source = inspect.getsource(brain.ask_nova_async)
    assert 'if tier == "hokage" and not ignored:' in source
    assert "memory_pass.notice_later(user_text" in source
