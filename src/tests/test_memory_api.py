"""The app can add, edit and trace memories. Sep 16 2026: he could list them
and approve pending ones, but not fix one that was wrong."""

import os
import tempfile

import database

# server.py calls init_db() at import. Point it somewhere throwaway first, so
# importing it can never touch the real ~/miles/data/miles.db.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402


@pytest.fixture
def client(db, fake_embeddings):
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def _status(memory_id):
    return database.memory_status(memory_id)


def test_a_typed_memory_is_active_and_explicit(client, db):
    response = client.post("/memories", json={"content": "  Last day at work is September 25.  "})
    assert response.status_code == 201
    memory_id = response.json()["id"]
    assert _status(memory_id) == "active"
    listed = client.get("/memories").json()["memories"]
    assert listed[0] == {"id": memory_id, "content": "Last day at work is September 25.",
                         "category": None, "source": "explicit", "confidence": "high",
                         "status": "active"}


def test_the_same_memory_twice_is_refused(client):
    client.post("/memories", json={"content": "Exam is Friday."})
    assert client.post("/memories", json={"content": "Exam is Friday."}).status_code == 409


def test_an_empty_memory_is_refused(client):
    assert client.post("/memories", json={"content": "   "}).status_code == 400


def test_an_edit_keeps_the_old_wording_and_links_it(client):
    old_id = client.post("/memories", json={"content": "Exam is Friday."}).json()["id"]
    edited = client.patch(f"/memories/{old_id}", json={"content": "Exam is Thursday."}).json()
    assert edited["changed"] and edited["replaced"] == old_id
    assert _status(old_id) == "superseded"
    assert _status(edited["id"]) == "active"
    history = client.get(f"/memories/{edited['id']}/history").json()
    assert [h["content"] for h in history] == ["Exam is Thursday.", "Exam is Friday."]
    assert history[1]["superseded_by"] == edited["id"]


def test_editing_a_pending_memory_approves_it_in_his_words(client, db):
    pending_id = database.save_memory("Maybe likes climbing.", source="implicit",
                                      status="pending", confidence="low")
    edited = client.patch(f"/memories/{pending_id}", json={"content": "Climbs on Sundays."}).json()
    assert _status(edited["id"]) == "active"
    assert client.get("/memories/pending").json() == []
    listed = client.get("/memories").json()["memories"]
    assert listed[0]["source"] == "explicit"


def test_an_edit_with_the_same_words_changes_nothing(client):
    memory_id = client.post("/memories", json={"content": "Exam is Friday."}).json()["id"]
    assert client.patch(f"/memories/{memory_id}", json={"content": "Exam is Friday."}).json() == \
        {"id": memory_id, "changed": False}
    assert _status(memory_id) == "active"


def test_a_replaced_memory_cannot_be_edited_again(client):
    """That would give one fact two current answers."""
    old_id = client.post("/memories", json={"content": "Exam is Friday."}).json()["id"]
    client.patch(f"/memories/{old_id}", json={"content": "Exam is Thursday."})
    assert client.patch(f"/memories/{old_id}", json={"content": "Exam is Monday."}).status_code == 404


def test_history_of_an_unknown_memory_is_not_found(client):
    assert client.get("/memories/99999/history").status_code == 404


# ── Duplicates, Sep 28 2026 ──

def _pair(db):
    a = db.save_memory("Leg extensions stay in his programme.", source="seed")
    b = db.save_memory("Leg extensions stay in his programme for knees.", source="explicit")
    return a, b


def test_similar_pairs_are_listed_until_he_keeps_both(client, db):
    a, b = _pair(db)
    db.save_memory("Obsession is a favourite film.", source="explicit")
    pairs = client.get("/memories/similar").json()["pairs"]
    assert [(p["a"]["id"], p["b"]["id"]) for p in pairs] == [(a, b)]

    assert client.post("/memories/similar/dismiss", json={"a_id": b, "b_id": a}).status_code == 200
    assert client.get("/memories/similar").json()["pairs"] == []


def test_a_merge_is_one_memory_with_history_and_the_other_gone(client, db):
    a, b = _pair(db)
    r = client.post("/memories/merge", json={"keep_id": a, "drop_id": b,
                                             "content": "Leg extensions stay, for his knees."})
    new = r.json()["id"]
    assert r.json() == {"id": new, "replaced": a, "deleted": b}
    assert _status(new) == "active" and _status(a) == "superseded" and _status(b) is None
    assert [h["id"] for h in client.get(f"/memories/{new}/history").json()] == [new, a]


def test_a_pending_fact_can_replace_the_one_it_updates(client, db):
    old = db.save_memory("Plans to pledge D9 this fall.", source="seed")
    waiting = db.save_memory("Not pledging D9 this year.", source="implicit", status="pending")
    new = client.post("/memories/merge", json={"keep_id": old, "drop_id": waiting,
                                               "content": "Not pledging D9 this year."}).json()["id"]
    assert _status(new) == "active" and _status(old) == "superseded" and _status(waiting) is None


def test_a_merge_of_something_already_replaced_is_refused(client, db):
    a, b = _pair(db)
    db.supersede_memory(a, "Leg extensions, rewritten.")
    r = client.post("/memories/merge", json={"keep_id": a, "drop_id": b, "content": "x"})
    assert r.status_code == 404 and _status(b) == "active", "nothing half done"


def test_pending_memories_carry_what_they_may_repeat(client, db):
    a, _ = _pair(db)
    waiting = db.save_memory("Leg extensions stay in his programme always.",
                             source="implicit", status="pending")
    unrelated = db.save_memory("Obsession is a favourite film.", source="implicit",
                               status="pending")
    by_id = {p["id"]: p for p in client.get("/memories/pending").json()}
    assert by_id[waiting]["similar_to"]["id"] in {a, _}
    assert by_id[unrelated]["similar_to"] is None


def test_the_draft_is_only_a_suggestion(client, db, monkeypatch):
    a, b = _pair(db)
    monkeypatch.setattr(server.memory_merge, "draft", lambda older, newer: (older + " | " + newer, True))
    r = client.post("/memories/merge/draft", json={"a_id": b, "b_id": a}).json()
    assert r == {"content": "Leg extensions stay in his programme. | Leg extensions stay in his "
                            "programme for knees.", "drafted": True}, "older first"
    assert _status(a) == "active" and _status(b) == "active", "nothing stored"


def test_with_no_model_the_draft_is_the_longer_memory(monkeypatch):
    import memory_merge
    def down(**k):
        raise ConnectionError("offline")
    monkeypatch.setattr(memory_merge, "client",
                        type("C", (), {"messages": type("M", (), {"create": staticmethod(down)})})())
    assert memory_merge.draft("Short.", "A longer one.") == ("A longer one.", False)
