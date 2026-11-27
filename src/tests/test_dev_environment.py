"""Acceptance coverage for the supported isolated headless entry point."""

from contextlib import closing, contextmanager
import os
import socket
import sqlite3
import subprocess
import sys

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

import apple_keys
import calendar_tools
import database
import deepgram_stt
import evallo_tools
import hevy_tools
import oura_tools
import prowl_tools
import push
import runtime_profile
import server
from dev_guards import BoundaryError, State, directory
from dev_providers import CHAT, SESSION
from dev_runtime import LoopbackUnavailable, seed, serve_loopback


@contextmanager
def rejected(kind):
    """Check intentional denials without hiding an unrelated attempted operation."""
    attempt_log = runtime_profile.current().violations
    first = len(attempt_log)
    try:
        with pytest.raises(BoundaryError):
            yield
    finally:
        if attempt_log[first:] == [kind]:
            attempt_log.pop()


def test_early_app_import_used_the_guarded_worktree_state():
    state = runtime_profile.current()
    with closing(sqlite3.connect(state.db_path)) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(45,)]
    assert os.environ["HOME"] == str(state.root)
    assert os.environ["MILES_MODE"] == "test"
    assert not ({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "ANTHROPIC_API_KEY",
                 "GOOGLE_APPLICATION_CREDENTIALS", "HEVY_API_KEY", "PROWL_TOKEN"}
                & os.environ.keys())


def test_seed_is_repeatable_and_schema_is_file_backed(db):
    state = runtime_profile.current()
    for _ in range(2):
        db.init_db()
        seed(state.auth)
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(45,)]
        conn.execute("CREATE VIRTUAL TABLE fts_acceptance USING fts5(text)")
        for table in ("lists", "list_items", "tasks", "reminders", "sessions", "capture_keys", "capture_log"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (1,)


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_other_state_survives_refused_open_and_reset(tmp_path, suffix, link):
    root, peer = tmp_path / "local", tmp_path / "peer"
    root.mkdir(mode=0o700)
    peer.mkdir(mode=0o700)
    sentinel = peer / "miles.db"
    sentinel.write_bytes(b"unrelated synthetic state")
    points_to_peer = root / ("miles.db" + suffix)
    if link == "symbolic":
        points_to_peer.symlink_to(sentinel)
    else:
        os.link(sentinel, points_to_peer)

    with directory(root, private=True) as fd:
        state = State(root, fd, "dev", None)
        with state.exclusive():
            with pytest.raises(BoundaryError):
                with state.db_target(state.db_path):
                    pytest.fail("unsafe target was accepted")
            with pytest.raises(BoundaryError):
                state.reset_database()
    assert points_to_peer.exists()
    assert sentinel.read_bytes() == b"unrelated synthetic state"


def test_state_aliases_and_other_worktree_paths_are_rejected(tmp_path):
    root, peer = tmp_path / "local", tmp_path / "peer"
    root.mkdir(mode=0o700)
    peer.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(peer, target_is_directory=True)
    with pytest.raises(OSError):
        with directory(alias, private=True):
            pytest.fail("root alias was accepted")
    target = peer / "miles.db"
    target.write_bytes(b"synthetic peer")
    with directory(root, private=True) as fd:
        state = State(root, fd, "dev", None)
        with state.exclusive():
            for unsafe in (target, str(root) + "/../peer/miles.db", str(root) + "//miles.db"):
                with pytest.raises(BoundaryError):
                    with state.db_target(unsafe):
                        pytest.fail("another path spelling was accepted")
            state.reset_database()
    assert target.read_bytes() == b"synthetic peer"


@pytest.mark.parametrize("holder_mode", ["dev", "test"], ids=["api", "operation"])
def test_reset_refuses_while_a_caller_holds_the_state(tmp_path, holder_mode):
    root = tmp_path / "worktree"
    root.mkdir(mode=0o700)
    with directory(root, private=True) as holder_fd, directory(root, private=True) as reset_fd:
        holder = State(root, holder_fd, holder_mode, None)
        resetter = State(root, reset_fd, "dev", None)
        holder.db_path.write_bytes(b"synthetic active state")
        with holder.exclusive():
            with pytest.raises(BoundaryError, match="in use"):
                with resetter.exclusive():
                    resetter.reset_database()
            assert holder.db_path.read_bytes() == b"synthetic active state"
        with resetter.exclusive():
            resetter.reset_database()
        assert not holder.db_path.exists()


@pytest.mark.parametrize("filename", ["runtime.lock", "credentials.json"])
def test_protected_state_rejects_links(tmp_path, filename):
    root = tmp_path / "worktree"
    root.mkdir(mode=0o700)
    sentinel = tmp_path / "untouched.state"
    sentinel.write_bytes(b"synthetic state")
    os.link(sentinel, root / filename)
    with directory(root, private=True) as fd:
        state = State(root, fd, "dev", None)
        with pytest.raises(BoundaryError):
            with state.file(filename, os.O_RDONLY):
                pytest.fail("unsafe credential or lock file opened")
    assert sentinel.read_bytes() == b"synthetic state"


def test_foreign_worktree_state_is_refused_before_python_file_access(tmp_path):
    foreign = tmp_path / "peer" / ".miles-dev" / "miles.db"
    with rejected("foreign runtime state"):
        foreign.read_bytes()
    with rejected("foreign runtime state"):
        foreign.write_bytes(b"synthetic")


def test_network_and_process_guards_catch_refusals_before_the_operation():
    port = runtime_profile.current().port
    assert socket.getaddrinfo("127.0.0.1", port, type=socket.SOCK_STREAM)[0][4] == ("127.0.0.1", port)
    with rejected("DNS lookup"):
        socket.getaddrinfo("provider.invalid", 443)
    for address in [("192.0.2.10", 443), ("127.0.0.1", 1)]:
        with socket.socket() as sock, rejected("socket connection"):
            sock.connect(address)
    with socket.socket() as sock, rejected("socket bind"):
        sock.bind(("0.0.0.0", port))
    with rejected("process launch"):
        subprocess.run([sys.executable, "-c", "raise SystemExit(42)"], check=True)


def test_real_provider_factories_refuse_in_the_headless_profile():
    for callback in (calendar_tools._live_service, deepgram_stt._live_session,
                     apple_keys._key, push._make_client,
                     lambda: oura_tools._get("sleep"), lambda: hevy_tools._get("workouts")):
        with pytest.raises(FileNotFoundError, match="disabled"):
            callback()
    with pytest.raises(prowl_tools.ProwlError, match="disabled"):
        prowl_tools._request("GET", "summary")
    assert not evallo_tools.run_script("Alex", None, "Completed", False)["ok"]


def login(client):
    assert client.post("/auth/login", json={"password": "synthetic-invalid-password"}).status_code == 401
    response = client.post("/auth/login", json={"password": runtime_profile.current().auth["password"]})
    assert response.status_code == 200
    headers = {"Authorization": "Bearer " + response.json()["access_token"]}
    refreshed = client.post("/auth/refresh", headers=headers)
    assert refreshed.status_code == 200
    return {"Authorization": "Bearer " + refreshed.json()["access_token"]}


def test_generated_auth_seeded_routes_and_fakes_work_together(db, monkeypatch):
    from dev_providers import calendar

    state = runtime_profile.current()
    seed(state.auth)
    monkeypatch.setattr(calendar_tools, "_service", calendar)
    with TestClient(server.app) as client:
        headers = login(client)
        assert client.get("/status", headers={"Authorization": "Bearer synthetic-invalid"}).status_code == 401
        assert client.get("/status", headers=headers).status_code == 200
        status = client.get("/status/details", headers=headers).json()
        assert status["core_temp_f"] is None
        assert set(status["services"].values()) == {None}
        for path in ("/tasks/overview", "/lists", "/reminders", "/sessions"):
            assert client.get(path, headers=headers).status_code == 200

        captured = client.post("/capture", headers={"Authorization": "Bearer " + state.auth["capture_key"]},
                               json={"source": "synthetic", "text": "Sample follow up"})
        assert captured.status_code == 200
        assert client.post("/capture", headers=headers,
                           json={"source": "synthetic", "text": "Refused"}).status_code == 401

        message = {"message": "Please describe a fictional mountain.", "channel": "text"}
        reply = client.post("/chat/stream", headers=headers, json=message)
        assert reply.status_code == 200 and CHAT in reply.text
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"token": headers["Authorization"].split()[1]})
            assert ws.receive_json() == {"status": "authenticated"}
            ws.send_json(message)
            assert ws.receive_json() == {"response": CHAT}

        session_id = client.get("/sessions", headers=headers).json()["sessions"][0]["id"]
        turn = client.post(f"/sessions/{session_id}/turns", headers=headers,
                           json={"message": "Review the synthetic fixture."})
        assert turn.status_code == 200 and SESSION in turn.text

        events = client.get("/calendar/events", headers=headers, params={"days": 2})
        assert events.status_code == 200
        assert "Sample appointment" in events.text
        assert client.post("/speak", headers=headers, json={"text": "Sample"}).status_code == 502


def test_real_loopback_api_lifecycle(db):
    port = runtime_profile.current().port
    config = uvicorn.Config("server:app", host="127.0.0.1", port=port,
                            access_log=False, lifespan="on")
    try:
        with serve_loopback(config) as url, httpx.Client(base_url=url, trust_env=False) as client:
            headers = login(client)
            assert client.get("/status", headers=headers).status_code == 200
            assert client.get("/status", headers={"Authorization": "Bearer synthetic-invalid"}).status_code == 401
            reply = client.post("/chat/stream", headers=headers,
                                json={"message": "Describe a fictional mountain.", "channel": "text"})
            assert reply.status_code == 200 and CHAT in reply.text
    except LoopbackUnavailable:
        pytest.skip("Runner denies designated loopback binds")
