"""Every memory can be found: by meaning as soon as it is stored, and by recall
once it falls out of the prompt. A bag of words stands in for the model."""

import sqlite3

import brain


def _vectors(db):
    return {row[0] for row in sqlite3.connect(db.DB_PATH).execute(
        "SELECT memory_id FROM memory_embeddings")}


def test_a_new_memory_is_embedded_by_the_next_search(db, fake_embeddings):
    """Sep 28 2026: nothing called backfill, so no memory stored after Sep 13
    had a vector and search by meaning could not find any of them."""
    old = db.save_memory("Leg extensions stay in his programme.", source="explicit")
    waiting = db.save_memory("Obsession is a favourite film.", source="implicit",
                             status="pending")
    db.search_memories("programme")
    assert _vectors(db) == {old, waiting}, "into the test database, pending included"


def test_excluded_memories_do_not_take_places_in_the_ranking(db, fake_embeddings):
    ids = [db.save_memory(f"Gym session number {n} was good.", source="explicit")
           for n in range(6)]
    hits = db.search_memories("gym session", limit=3, exclude=ids[:4])
    assert {row[0] for row in hits} == set(ids[4:])


def test_recall_looks_past_what_the_prompt_already_holds(monkeypatch):
    """Learned memories are tier one, and only the newest twenty are resident.
    Recall searched tier two, so the twenty first was reachable nowhere."""
    asked = {}
    monkeypatch.setattr(brain, "get_seed_memories", lambda: [(1, "seed", "x")])
    monkeypatch.setattr(brain, "get_episodic_memories", lambda: [(2, "newest")])
    monkeypatch.setattr(brain, "search_memories",
                        lambda query, **k: asked.update(k) or [(3, "older learned", None)])
    messages = brain._with_recalled([{"role": "user", "content": "gym?"}], "gym?", log=False)
    assert asked["exclude"] == {1, 2} and "tier" not in asked
    assert "older learned" in messages[-1]["content"]
