from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier
import sqlite3

import pytest

import database


def _version(conn):
    return conn.execute("SELECT version FROM schema_version").fetchone()[0]


def _start_columns(conn):
    return [row for row in conn.execute("PRAGMA table_info(tasks)") if row[1] == "start_date"]


@pytest.fixture(autouse=True)
def isolate_migration_046(monkeypatch):
    monkeypatch.setattr(database, "MIGRATIONS", [
        entry for entry in database.MIGRATIONS if entry[0] <= 46])


def _run(path):
    with closing(sqlite3.connect(path)) as conn:
        database._run_migrations(conn)


def _at_version(tmp_path, monkeypatch, version=45):
    path = str(tmp_path / "tasks.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS", [
            entry for entry in database.MIGRATIONS if entry[0] <= version])
        database.init_db()
    return path


@pytest.mark.parametrize("version", [44, 45])
def test_046_is_additive_and_repeated_execution_preserves_tasks(tmp_path, monkeypatch, version):
    path = _at_version(tmp_path, monkeypatch, version)
    with closing(sqlite3.connect(path)) as conn, conn:
        assert _version(conn) == version
        assert not _start_columns(conn)
        conn.execute("""INSERT INTO tasks
                     (id, text, recurrence, created_at, scheduled_for, due_date, status, archived_at)
                     VALUES (1, 'Legacy daily', 'daily', '2026-11-23', NULL, NULL, 'pending', NULL),
                            (2, 'Legacy one time', NULL, '2026-11-22', '2026-11-25',
                             '2026-11-30', 'done', '2026-11-26')""")
        conn.execute("INSERT INTO task_completions VALUES (1, '2026-11-24'), (2, '2026-11-25')")
        if version == 45:
            conn.execute("INSERT INTO evallo_verified_sessions VALUES ('fake-session')")
            conn.execute("INSERT INTO evallo_verification VALUES (1, 'fake', 'fake', 'fake', 'fake', 1, 0)")
        tasks = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
        completions = conn.execute("SELECT * FROM task_completions ORDER BY task_id").fetchall()
    _run(path)
    _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 46
        assert len(_start_columns(conn)) == 1
        assert _start_columns(conn)[0][2:] == ("TEXT", 0, None, 0)
        assert conn.execute("SELECT * FROM tasks ORDER BY id").fetchall() == [
            (*row, None) for row in tasks]
        assert conn.execute("SELECT * FROM task_completions ORDER BY task_id").fetchall() == completions
        if version == 45:
            assert conn.execute("SELECT * FROM evallo_verified_sessions").fetchall() == [("fake-session",)]
            assert conn.execute("SELECT * FROM evallo_verification").fetchall() == [
                (1, "fake", "fake", "fake", "fake", 1, 0)]
        else:
            assert conn.execute("SELECT count(*) FROM evallo_verification").fetchone() == (0,)
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("definition", ["TEXT", "TEXT DEFAULT NULL", "INTEGER", "TEXT DEFAULT 'synthetic'"])
def test_046_handles_preexisting_column_without_transforming_rows(tmp_path, monkeypatch, definition):
    path = _at_version(tmp_path, monkeypatch)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"ALTER TABLE tasks ADD COLUMN start_date {definition}")
        conn.execute("""INSERT INTO tasks (text, recurrence, created_at, start_date)
                        VALUES ('synthetic', 'daily', '2026-11-25', '2026-11-26')""")
    if definition.startswith("TEXT") and definition != "TEXT DEFAULT 'synthetic'":
        _run(path)
        expected = 46
    else:
        with pytest.raises(sqlite3.OperationalError, match="Incompatible tasks.start_date"):
            _run(path)
        expected = 45
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == expected
        assert conn.execute("SELECT start_date FROM tasks").fetchone() == ("2026-11-26",)


def test_046_atomic_rollback_and_retry(tmp_path, monkeypatch):
    path = _at_version(tmp_path, monkeypatch)
    real_046 = database._migration_046_task_start_date

    def fail_after_ddl(conn):
        real_046(conn)
        raise sqlite3.OperationalError("synthetic failure before version stamp")

    with monkeypatch.context() as patch:
        patch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:-1] + [(46, fail_after_ddl)])
        with pytest.raises(sqlite3.OperationalError, match="synthetic failure"):
            _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 45 and not _start_columns(conn)
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    _run(path)
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 46 and len(_start_columns(conn)) == 1


def test_two_migration_starters_apply_046_once(tmp_path, monkeypatch):
    path = _at_version(tmp_path, monkeypatch)
    barrier = Barrier(2)
    called = []

    def counted(conn):
        called.append(46)
        database._migration_046_task_start_date(conn)

    def start():
        barrier.wait(timeout=5)
        _run(path)

    monkeypatch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:-1] + [(46, counted)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(start) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert called == [46]
    with closing(sqlite3.connect(path)) as conn:
        assert _version(conn) == 46 and len(_start_columns(conn)) == 1
