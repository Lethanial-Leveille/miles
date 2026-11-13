"""Lists and tasks: migration 33 so far. The queries, tools and routes add
their tests here as they land."""

import sqlite3

import database


def _tables(path):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def test_migration_33_leaves_existing_rows_alone(tmp_path, monkeypatch):
    # Stop at 32, the version the Pi is on today, and write rows the way the
    # live database already has them.
    path = str(tmp_path / "miles.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:32])
    database.init_db()
    database.save_message("user", "hey nova")
    database.save_memory("Likes pull ups", source="explicit")
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO reminders (content, due_at, created_at) "
                 "VALUES ('stretch', NULL, '2026-11-12T09:00:00')")
    conn.commit()
    before = {t: conn.execute(f"SELECT * FROM {t}").fetchall()
              for t in ("conversation_history", "memories", "reminders")}
    conn.close()

    # Now the real list, which is 33 long, as a deploy would run it.
    monkeypatch.undo()
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()

    conn = sqlite3.connect(path)
    after = {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in before}
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    conn.close()
    assert after == before
    assert version == 33
    assert {"lists", "list_items", "tasks", "task_completions"} <= _tables(path)


def test_migration_33_runs_once(db):
    # A second start must not try to create the tables again.
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 33
    conn.close()


def test_recurrence_only_accepts_daily_or_nothing(db):
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("INSERT INTO tasks (text, recurrence, created_at) "
                 "VALUES ('vitamins', 'daily', 'x')")
    conn.execute("INSERT INTO tasks (text, recurrence, created_at) "
                 "VALUES ('call the bank', NULL, 'x')")
    try:
        conn.execute("INSERT INTO tasks (text, recurrence, created_at) "
                     "VALUES ('typo', 'Daily', 'x')")
        refused = False
    except sqlite3.IntegrityError:
        refused = True
    conn.close()
    assert refused


def test_a_task_is_completed_once_per_day(db):
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("INSERT INTO tasks (text, recurrence, created_at) "
                 "VALUES ('vitamins', 'daily', 'x')")
    conn.execute("INSERT INTO task_completions VALUES (1, '2026-11-12')")
    try:
        conn.execute("INSERT INTO task_completions VALUES (1, '2026-11-12')")
        refused = False
    except sqlite3.IntegrityError:
        refused = True
    conn.close()
    assert refused


def test_a_new_task_is_active(db):
    # archived_at starts empty, which is what makes a task active.
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("INSERT INTO tasks (text, recurrence, created_at) "
                 "VALUES ('vitamins', 'daily', 'x')")
    archived = conn.execute("SELECT archived_at FROM tasks").fetchone()[0]
    conn.close()
    assert archived is None
