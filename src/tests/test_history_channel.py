"""Migration 050, the channel column on conversation_history, and who writes it.
Synthetic data only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier
import sqlite3

import pytest

import database


def _version(conn):
    return conn.execute("SELECT version FROM schema_version").fetchone()[0]


def _columns(conn):
    return [row for row in conn.execute("PRAGMA table_info(conversation_history)") if row[1] == "channel"]


def _at(tmp_path, monkeypatch, version):
    path = str(tmp_path / "history.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS", [e for e in database.MIGRATIONS if e[0] <= version])
        database.init_db()
    return path


def _run(path):
    with closing(sqlite3.connect(path)) as conn:
        database._run_migrations(conn)


def test_050_starts_at_49_keeps_every_row_and_adds_one_nullable_column(tmp_path, monkeypatch):
    path = _at(tmp_path, monkeypatch, 49)
    with closing(sqlite3.connect(path)) as conn, conn:
        assert _version(conn) == 49 and not _columns(conn)
        conn.execute("""INSERT INTO conversation_history (role, content, created_at, source_device)
                        VALUES ('user', 'hello Alex', '2026-11-25T10:00:00', 'app'),
                               ('assistant', 'hi Jordan', '2026-11-25T10:00:01', 'app')""")
        before = conn.execute("SELECT id, role, content, created_at, source_device "
                              "FROM conversation_history ORDER BY id").fetchall()
    _run(path)
    _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 50
        [column] = _columns(conn)
        assert column[2].upper() == "TEXT" and column[3] == 0 and column[4] is None
        assert conn.execute("SELECT id, role, content, created_at, source_device "
                            "FROM conversation_history ORDER BY id").fetchall() == before
        assert conn.execute("SELECT DISTINCT channel FROM conversation_history").fetchall() == [(None,)]
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_050_rolls_back_on_injected_failure_and_retries(tmp_path, monkeypatch):
    path = _at(tmp_path, monkeypatch, 49)
    real = database._migration_050_history_channel

    def fail_after_ddl(conn):
        real(conn)
        raise sqlite3.OperationalError("synthetic failure before version stamp")

    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS",
                      [e for e in database.MIGRATIONS if e[0] < 50] + [(50, fail_after_ddl)])
        with pytest.raises(sqlite3.OperationalError, match="synthetic failure"):
            _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 49 and not _columns(conn)
    _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 50 and len(_columns(conn)) == 1


def test_two_starters_apply_050_once(tmp_path, monkeypatch):
    path = _at(tmp_path, monkeypatch, 49)
    barrier, called = Barrier(2), []

    def counted(conn):
        called.append(50)
        database._migration_050_history_channel(conn)

    def start():
        barrier.wait(timeout=5)
        _run(path)

    monkeypatch.setattr(database, "MIGRATIONS",
                        [e for e in database.MIGRATIONS if e[0] < 50] + [(50, counted)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(start) for _ in range(2)]:
            future.result(timeout=10)
    assert called == [50]
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 50 and len(_columns(conn)) == 1


def test_save_message_stores_the_channel_and_defaults_to_null(db):
    db.save_message("user", "typed this", device="app", channel="text")
    db.save_message("assistant", "said this", channel="voice")
    db.save_message("user", "no channel")
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        rows = conn.execute("SELECT content, channel FROM conversation_history ORDER BY id").fetchall()
    assert rows == [("typed this", "text"), ("said this", "voice"), ("no channel", None)]


from test_task_intent import turn  # noqa: E402,F401  (a fixture)


@pytest.mark.parametrize("channel", ["voice", "text"])
def test_an_app_turn_writes_its_channel_on_both_rows(turn, db, channel):
    import asyncio
    import brain

    from test_task_intent import Stream
    turn.rounds.clear()
    turn.rounds.append(Stream([], "Good morning."))
    asyncio.run(brain.ask_nova_async("good morning", device="app", channel=channel, tier="hokage"))
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        rows = conn.execute("SELECT role, channel FROM conversation_history ORDER BY id").fetchall()
    assert rows == [("user", channel), ("assistant", channel)]


# ── the one journal line per app turn ──

def _app_turn(turn, text, channel, blocks, draft=""):
    import asyncio
    import brain
    from test_task_intent import Stream

    turn.rounds.clear()
    turn.rounds.append(Stream(list(blocks), draft))
    return asyncio.run(brain.ask_nova_async(text, device="app", channel=channel, tier="hokage"))


def _lines(capsys):
    return [line for line in capsys.readouterr().out.splitlines() if line.startswith("App turn: ")]


def test_an_app_turn_prints_one_line_with_intent_verdict_and_cost(turn, db, capsys):
    from test_task_intent import block
    _app_turn(turn, "Add reading every day", "voice", [block("task_add", {"task": "reading", "recurrence": "daily"})])
    [line] = _lines(capsys)
    assert 'channel="voice"' in line and 'action="task_add"' in line and 'target="reading"' in line
    assert 'timing="every day"' in line and 'verdict="act"' in line and "intent_ms=" in line


def test_a_refused_turn_logs_the_refusal_text(turn, db, capsys):
    _app_turn(turn, "add call the bank at dawn", "text", [])
    [line] = _lines(capsys)
    assert 'channel="text"' in line and 'verdict="refuse"' in line
    assert "Nothing was changed." in line and "When do you want that?" in line


def test_an_unflagged_turn_logs_a_skipped_step_with_zero_cost(turn, db, capsys):
    _app_turn(turn, "good morning", "text", [], "Good morning.")
    [line] = _lines(capsys)
    assert 'verdict="skipped"' in line and "intent_ms=0" in line and 'action=""' in line
    assert "good morning" not in line.casefold()


def test_a_pi_turn_prints_no_app_line(turn, db, capsys):
    import asyncio
    import brain
    from test_task_intent import Stream
    turn.rounds.clear()
    turn.rounds.append(Stream([], "Hi."))
    asyncio.run(brain.ask_nova_async("good morning", device="pi", channel="voice", tier="hokage"))
    assert _lines(capsys) == []


def test_the_chat_route_passes_the_requested_channel_through(turn, db, monkeypatch, capsys):
    from fastapi.testclient import TestClient
    import server
    from test_task_intent import Stream

    monkeypatch.setitem(server.app.dependency_overrides, server.get_current_user, lambda: "fake-actor")
    turn.rounds.clear()
    turn.rounds.append(Stream([], "Good morning."))
    reply = TestClient(server.app).post("/chat", json={"message": "good morning", "channel": "text"})
    assert reply.status_code == 200
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        assert conn.execute("SELECT role, channel FROM conversation_history ORDER BY id").fetchall() == [
            ("user", "text"), ("assistant", "text")]
    [line] = _lines(capsys)
    assert 'channel="text"' in line
