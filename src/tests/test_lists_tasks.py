"""Lists and tasks: migration 33 and the queries. The tools and routes add
their tests here as they land."""

import sqlite3
from datetime import datetime

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



# ── The 4am day ──

def test_one_in_the_morning_counts_toward_the_day_before(db):
    assert db.task_day(datetime(2026, 11, 13, 1, 0)) == "2026-11-12"


def test_the_day_turns_over_at_four(db):
    assert db.task_day(datetime(2026, 11, 13, 3, 59)) == "2026-11-12"
    assert db.task_day(datetime(2026, 11, 13, 4, 0)) == "2026-11-13"


# ── tasks_today ──

def test_daily_task_done_yesterday_but_not_today_shows_up(db):
    tid = db.add_task("take vitamins", "daily", now=datetime(2026, 11, 10, 9, 0))
    db.complete_task(tid, now=datetime(2026, 11, 11, 9, 0))
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == [
        (tid, "take vitamins", "daily")]


def test_daily_task_done_today_does_not_show_up(db):
    tid = db.add_task("take vitamins", "daily")
    db.complete_task(tid, now=datetime(2026, 11, 12, 8, 0))
    assert db.tasks_today(now=datetime(2026, 11, 12, 20, 0)) == []


def test_task_done_at_one_am_counts_toward_the_previous_day(db):
    tid = db.add_task("take vitamins", "daily")
    db.complete_task(tid, now=datetime(2026, 11, 13, 1, 0))
    # Still the 12th until 4am, and it is done for the 12th.
    assert db.tasks_today(now=datetime(2026, 11, 13, 2, 0)) == []
    # The 13th starts at 4am, and nothing has been done on it yet.
    assert db.tasks_today(now=datetime(2026, 11, 13, 9, 0)) == [
        (tid, "take vitamins", "daily")]


def test_open_one_time_task_shows_up(db):
    tid = db.add_task("call the bank")
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == [
        (tid, "call the bank", None)]


def test_finished_one_time_task_never_comes_back(db):
    tid = db.add_task("call the bank")
    db.complete_task(tid, now=datetime(2026, 11, 10, 9, 0))
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == []


def test_archived_daily_task_does_not_show_up(db):
    tid = db.add_task("take vitamins", "daily")
    db.complete_task(tid, now=datetime(2026, 11, 11, 9, 0))
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE tasks SET archived_at = '2026-11-11T20:00:00' WHERE id = ?",
                 (tid,))
    conn.commit()
    history = conn.execute("SELECT date FROM task_completions WHERE task_id = ?",
                           (tid,)).fetchall()
    conn.close()
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == []
    # Retired, not erased: the day it was done is still there.
    assert history == [("2026-11-11",)]


def test_an_archived_task_cannot_be_completed(db):
    tid = db.add_task("take vitamins", "daily")
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE tasks SET archived_at = 'x' WHERE id = ?", (tid,))
    conn.commit()
    conn.close()
    assert db.complete_task(tid) is False
    assert db.complete_task(999) is False


def test_done_twice_in_one_day_is_one_row(db):
    tid = db.add_task("take vitamins", "daily")
    assert db.complete_task(tid, now=datetime(2026, 11, 12, 8, 0)) is True
    assert db.complete_task(tid, now=datetime(2026, 11, 12, 21, 0)) is True
    conn = sqlite3.connect(db.DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM task_completions").fetchone()[0]
    conn.close()
    assert count == 1


def test_tasks_today_lists_both_kinds_in_the_order_added(db):
    a = db.add_task("take vitamins", "daily")
    b = db.add_task("call the bank")
    c = db.add_task("stretch", "daily")
    db.complete_task(c, now=datetime(2026, 11, 12, 7, 0))
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == [
        (a, "take vitamins", "daily"), (b, "call the bank", None)]


def test_matching_skips_archived_tasks(db):
    a = db.add_task("take vitamins", "daily")
    b = db.add_task("buy vitamins")
    conn = sqlite3.connect(db.DB_PATH)
    conn.execute("UPDATE tasks SET archived_at = 'x' WHERE id = ?", (b,))
    conn.commit()
    conn.close()
    assert db.active_tasks_matching("vitamins") == [(a, "take vitamins", "daily")]


# ── Lists ──

def test_first_add_creates_the_list(db):
    assert db.find_list("grocery") is None
    lid = db.get_or_create_list("grocery")
    assert db.get_or_create_list("grocery") == lid
    assert db.find_list("grocery") == lid


def test_adding_the_same_unchecked_item_twice_keeps_one(db):
    lid = db.get_or_create_list("grocery")
    first, added = db.add_list_item(lid, "eggs")
    again, added_again = db.add_list_item(lid, "Eggs")
    assert (added, added_again) == (True, False)
    assert again == first
    assert [r[1] for r in db.list_items(lid)] == ["eggs"]


def test_checked_item_stays_with_when_it_was_checked(db):
    lid = db.get_or_create_list("grocery")
    iid, _ = db.add_list_item(lid, "eggs")
    assert db.check_list_item(iid, now=datetime(2026, 11, 12, 18, 30)) is True
    assert db.list_items(lid) == []
    [row] = db.list_items(lid, include_checked=True)
    assert row[:4] == (iid, "eggs", 1, "2026-11-12T18:30:00")
    # Checking it again is refused: nothing unchecked has that id.
    assert db.check_list_item(iid) is False


def test_needing_a_bought_item_again_is_a_new_row(db):
    lid = db.get_or_create_list("grocery")
    old, _ = db.add_list_item(lid, "eggs")
    db.check_list_item(old)
    new, added = db.add_list_item(lid, "eggs")
    assert added is True and new != old


def test_unchecked_items_come_first_then_newest_bought(db):
    lid = db.get_or_create_list("grocery")
    eggs, _ = db.add_list_item(lid, "eggs")
    milk, _ = db.add_list_item(lid, "milk")
    bread, _ = db.add_list_item(lid, "bread")
    db.check_list_item(eggs, now=datetime(2026, 11, 10, 9, 0))
    db.check_list_item(bread, now=datetime(2026, 11, 11, 9, 0))
    assert [r[1] for r in db.list_items(lid, include_checked=True)] == [
        "milk", "bread", "eggs"]


def test_matching_finds_every_unchecked_item_containing_the_words(db):
    lid = db.get_or_create_list("grocery")
    oat, _ = db.add_list_item(lid, "oat milk")
    whole, _ = db.add_list_item(lid, "whole milk")
    db.add_list_item(lid, "eggs")
    assert db.unchecked_items_matching(lid, "milk") == [
        (oat, "oat milk"), (whole, "whole milk")]


def test_all_lists_counts_only_unchecked(db):
    g = db.get_or_create_list("grocery")
    h = db.get_or_create_list("hardware")
    iid, _ = db.add_list_item(g, "eggs")
    db.add_list_item(g, "milk")
    db.add_list_item(h, "nails")
    db.check_list_item(iid)
    assert db.all_lists() == [(g, "grocery", 1), (h, "hardware", 1)]
