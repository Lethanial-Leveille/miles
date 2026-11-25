"""Lists and tasks as the app sees them: read lists and today's tasks, check
items off and back on, finish tasks."""

import os
import tempfile

import database

# server.py calls init_db() at import. Point it somewhere throwaway first.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402


@pytest.fixture
def client(db):
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def _grocery(db, *items):
    lid = db.get_or_create_list("grocery")
    return lid, [db.add_list_item(lid, item)[0] for item in items]


def test_lists_with_unchecked_counts(client, db):
    lid, (eggs, _) = _grocery(db, "eggs", "milk")
    db.check_list_item(eggs)
    assert client.get("/lists").json() == [
        {"id": lid, "name": "grocery", "unchecked": 1}]


def test_items_unchecked_only_by_default(client, db):
    lid, (eggs, milk) = _grocery(db, "eggs", "milk")
    db.check_list_item(eggs)
    items = client.get(f"/lists/{lid}/items").json()
    assert [(i["id"], i["text"], i["checked"]) for i in items] == [
        (milk, "milk", False)]


def test_items_with_history(client, db):
    lid, (eggs, milk) = _grocery(db, "eggs", "milk")
    db.check_list_item(eggs)
    items = client.get(f"/lists/{lid}/items?include_checked=true").json()
    assert [(i["text"], i["checked"]) for i in items] == [
        ("milk", False), ("eggs", True)]
    assert items[1]["checked_at"] is not None


def test_items_of_a_missing_list_is_404(client):
    assert client.get("/lists/99/items").status_code == 404


def test_tapping_an_item_checks_it_and_keeps_it(client, db):
    lid, (eggs,) = _grocery(db, "eggs")
    response = client.post(f"/lists/items/{eggs}/check")
    assert response.status_code == 200
    assert response.json() == {"checked": True}
    assert db.list_items(lid) == []
    assert len(db.list_items(lid, include_checked=True)) == 1


def test_tapping_a_checked_item_again_is_still_checked(client, db):
    # A retry from the phone must not look like an error, and must not move
    # checked_at to the retry's time.
    lid, (eggs,) = _grocery(db, "eggs")
    client.post(f"/lists/items/{eggs}/check")
    [before] = db.list_items(lid, include_checked=True)
    response = client.post(f"/lists/items/{eggs}/check")
    assert response.status_code == 200
    assert response.json() == {"checked": True}
    assert db.list_items(lid, include_checked=True) == [before]


def test_checking_a_missing_item_is_404(client):
    assert client.post("/lists/items/99/check").status_code == 404


def test_unchecking_puts_the_item_back(client, db):
    lid, (eggs,) = _grocery(db, "eggs")
    db.check_list_item(eggs)
    response = client.post(f"/lists/items/{eggs}/uncheck")
    assert response.status_code == 200
    assert response.json() == {"checked": False}
    [row] = db.list_items(lid)
    assert row[:4] == (eggs, "eggs", 0, None)
    # Again is fine too, and a missing item is 404.
    assert client.post(f"/lists/items/{eggs}/uncheck").status_code == 200
    assert client.post("/lists/items/99/uncheck").status_code == 404


def test_tasks_today_with_its_day(client, db):
    vitamins = db.add_task("take vitamins", "daily")
    bank = db.add_task("call the bank", scheduled_for=db.task_day())
    body = client.get("/tasks/today").json()
    assert body == {"day": db.task_day(), "tasks": [
        {"id": vitamins, "text": "take vitamins", "recurrence": "daily"},
        {"id": bank, "text": "call the bank", "recurrence": None}]}


def test_finishing_a_task_takes_it_off_today(client, db):
    vitamins = db.add_task("take vitamins", "daily")
    response = client.post(f"/tasks/{vitamins}/complete")
    assert response.json() == {"completed": True, "day": db.task_day()}
    assert client.get("/tasks/today").json()["tasks"] == []


def test_finishing_an_archived_or_missing_task_is_404(client, db):
    vitamins = db.add_task("take vitamins", "daily")
    db.archive_task(vitamins)
    assert client.post(f"/tasks/{vitamins}/complete").status_code == 404
    assert client.post("/tasks/99/complete").status_code == 404


def test_every_route_needs_a_login(db):
    # No override here: the real auth dependency runs and there is no token.
    bare = TestClient(server.app)
    for method, path in (("get", "/lists"), ("get", "/lists/1/items"),
                         ("post", "/lists/items/1/check"),
                         ("post", "/lists/items/1/uncheck"),
                         ("get", "/tasks/today"), ("post", "/tasks/1/complete")):
        assert getattr(bare, method)(path).status_code in (401, 403), path
