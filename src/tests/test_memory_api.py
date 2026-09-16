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
def client(db):
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
