"""Quick capture: the rules that pick task or list item, the key that is the
only way in, and the log of every capture."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

import auth
import capture
import server


# ── task or list item ──

@pytest.mark.parametrize("text, expected", [
    ("Email professor about the lab", ("task", None, ["Email professor about the lab"])),
    ("talk to mom", ("task", None, ["talk to mom"])),
    ("add eggs to my grocery list", ("list_item", "grocery", ["eggs"])),
    ("eggs, milk and bread to the grocery list.", ("list_item", "grocery", ["eggs", "milk", "bread"])),
    ("put a hammer on the hardware list", ("list_item", "hardware", ["a hammer"])),
    ("Note: call the bank", ("task", None, ["Note: call the bank"])),
])
def test_classify(text, expected):
    assert capture.classify(text, list_exists=lambda name: None) == expected


def test_a_colon_names_a_list_only_when_it_exists():
    exists = {"grocery": 1}.get
    assert capture.classify("grocery: eggs, milk", list_exists=exists) == (
        "list_item", "grocery", ["eggs", "milk"])
    assert capture.classify("Note: call the bank", list_exists=exists)[0] == "task"


# ── the route ──

@pytest.fixture
def key(db, monkeypatch):
    # A real secret, so a login JWT decodes and the 401s below are the route's
    # own refusals rather than a server that cannot read any token.
    monkeypatch.setenv("MILES_JWT_SECRET", "test secret")
    raw, key_hash = capture.new_key()
    key_id = db.add_capture_key("mac_hotkey", key_hash)
    return raw, key_id


def _post(key, text="Email professor", source="mac_hotkey"):
    return TestClient(server.app).post(
        "/capture", json={"text": text, "source": source},
        headers={"Authorization": f"Bearer {key}"})


def test_a_capture_becomes_a_task_and_is_logged(db, key):
    raw, key_id = key
    response = _post(raw, "email professor about the lab")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "kind": "task",
                               "message": "Added to backlog: Email professor about the lab"}
    # No day, so it waits in the backlog and is not on today's list.
    assert db.tasks_today() == []
    assert [t["text"] for t in db.backlog_tasks()] == ["Email professor about the lab"]
    conn = sqlite3.connect(db.DB_PATH)
    assert conn.execute("SELECT key_id, source, text, kind FROM capture_log").fetchall() == [
        (key_id, "mac_hotkey", "email professor about the lab", "task")]


def test_list_items_and_duplicates(db, key):
    raw, _ = key
    assert _post(raw, "eggs and milk to the grocery list").json()["message"] == \
        "Added eggs, milk to grocery"
    assert _post(raw, "eggs to the grocery list").json() == {
        "ok": True, "kind": "duplicate", "message": "eggs already on grocery"}


def test_a_revoked_key_is_refused(db, key):
    raw, key_id = key
    db.revoke_capture_key(key_id)
    assert _post(raw).status_code == 401


@pytest.mark.parametrize("bad", ["", "mcap_not_a_real_key", "nonsense"])
def test_unknown_keys_are_refused(db, key, bad):
    assert _post(bad).status_code in (401, 403)


def test_a_login_jwt_cannot_capture(db, key):
    login = auth.create_token("Lethanial")
    assert _post(login).status_code == 401


@pytest.mark.parametrize("method, path", [
    ("get", "/memories"), ("get", "/tasks/today"), ("get", "/lists"),
    ("get", "/calendar/events?start=2026-11-24&end=2026-11-25"), ("get", "/sessions"),
    ("get", "/history"), ("post", "/chat"),
])
def test_a_capture_key_reads_nothing(db, key, method, path):
    raw, _ = key
    client = TestClient(server.app)
    headers = {"Authorization": f"Bearer {raw}"}
    if method == "post":
        response = client.post(path, headers=headers, json={"message": "hi"})
    else:
        response = client.get(path, headers=headers)
    assert response.status_code == 401


def test_the_same_routes_open_for_a_login(db, key):
    # The control for the test above: a login gets past auth on them, so their
    # 401 for a key is the key being refused, not the route being broken.
    login = auth.create_token("Lethanial")
    assert TestClient(server.app).get(
        "/tasks/today", headers={"Authorization": f"Bearer {login}"}).status_code == 200


def test_input_is_bounded(db, key):
    raw, _ = key
    assert _post(raw, "x" * 501).status_code == 422
    assert _post(raw, source="Mac Hotkey!").status_code == 422
    assert _post(raw, "").status_code == 422


def test_the_key_is_stored_only_as_a_hash(db, key):
    raw, _ = key
    conn = sqlite3.connect(db.DB_PATH)
    stored = conn.execute("SELECT key_hash FROM capture_keys").fetchone()[0]
    assert raw not in stored and stored == capture.hash_key(raw)
    assert capture.CAPTURE_TIER == "chunin"


def test_a_new_key_goes_to_a_private_file_and_is_never_printed(tmp_path):
    import os
    import stat
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts", "capture_key.py")
    path = tmp_path / "keys" / "mac.key"
    env = {**os.environ, "MILES_DB_PATH": str(tmp_path / "keys.db")}
    out = subprocess.run([sys.executable, script, "new", "mac", "--path", str(path)],
                         capture_output=True, text=True, env=env, check=True)
    key = path.read_text().strip()
    assert key.startswith(capture.KEY_PREFIX)
    assert key not in out.stdout and key not in out.stderr
    assert str(path) in out.stdout and "Key 1" in out.stdout
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    again = subprocess.run([sys.executable, script, "new", "mac", "--path", str(path)],
                           capture_output=True, text=True, env=env)
    assert again.returncode != 0                       # never overwrites a key file
