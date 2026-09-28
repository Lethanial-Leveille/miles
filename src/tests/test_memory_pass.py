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
def run_pass(db, monkeypatch, fake_embeddings):
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


def test_a_rewording_of_a_stored_fact_is_not_queued(run_pass):
    """Sep 28 2026: the turn stored his top three films and, two seconds later,
    the pass queued the same three in other words."""
    database.save_memory("Andrew's test is on Saturday.", source="explicit")
    stored, _ = run_pass(_call(content="On Saturday is Andrew's test."))
    assert stored == [] and _pending() == []


def test_a_different_fact_about_the_same_thing_is_still_queued(run_pass):
    database.save_memory("Andrew's test is on Saturday.", source="explicit")
    stored, _ = run_pass(_call(content="Andrew's test is on Sunday."))
    assert stored == ["Andrew's test is on Sunday."]


def test_a_failed_comparison_still_queues_the_fact(run_pass, monkeypatch):
    """A missed duplicate costs one discard; a missed fact is gone."""
    def broken(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(memory_pass.embeddings, "nearest", broken)
    stored, _ = run_pass(_call(content="Andrew's test is on Saturday."))
    assert stored == ["Andrew's test is on Saturday."]


def test_similar_pairs_lists_close_active_memories_closest_first(db, fake_embeddings):
    import embeddings
    a = database.save_memory("Leg extensions stay in his programme.", source="explicit")
    b = database.save_memory("Leg extensions stay in his programme for knees.", source="explicit")
    database.save_memory("Obsession is a favourite film.", source="explicit")
    pairs = embeddings.similar_pairs(0.75, db_path=database.DB_PATH)
    assert [(p[1][0], p[2][0]) for p in pairs] == [(a, b)]


# ── what the turn itself stored ──
# The real pairs from tool_call_log, Sep 16 to 28 2026.

TURN_INJURY = ("Ring finger injury affecting pull ups and grip work on exercises. Toe injury "
               "from last week, expected to fully heal in about one more week from September 28.")
TURN_CHANGES = ["No longer tutoring Charley and Andrew.", "Quit the busboy job at Pi Beta Phi.",
                "Not pledging D9 this year.", "Not pledging this year."]


@pytest.mark.parametrize("piece,stored", [
    ("Ring finger injury affecting pull ups and grip work on exercises.", TURN_INJURY),
    ("Toe injury from last week, expected to fully heal in about one more week from September 28.",
     TURN_INJURY),
    ("Obsession, Odyssey, and Spider-Man: Brand New Day are his top three movies of 2026.",
     "Obsession is one of his top three movies of 2026, along with Odyssey and Spider-Man: "
     "Brand New Day."),
])
def test_a_piece_of_what_the_turn_stored_is_skipped(piece, stored):
    assert memory_pass._stored_by_the_turn(piece, [stored])


def test_a_correction_is_not_swallowed_by_the_facts_it_corrects():
    """Against the four together it scores 0.89, which is why each is compared alone."""
    correction = "Still tutoring, but no longer the busboy job and not pledging this year."
    assert not memory_pass._stored_by_the_turn(correction, TURN_CHANGES)
    assert memory_pass.overlap(correction, " ".join(TURN_CHANGES)) >= memory_pass.SAME_MESSAGE_OVERLAP


def test_nothing_stored_by_the_turn_skips_nothing():
    assert not memory_pass._stored_by_the_turn("Ring finger injury.", [])
