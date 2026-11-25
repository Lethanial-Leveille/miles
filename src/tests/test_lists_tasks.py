"""Lists and tasks: migration 33, the queries and the tools. The routes add
their tests here when they land."""

import sqlite3
from datetime import datetime

import pytest

import database
import lists_tasks
from tools import ToolNotice, permits, registry


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
    assert version == database.MIGRATIONS[-1][0]
    assert {"lists", "list_items", "tasks", "task_completions"} <= _tables(path)


def test_migration_33_runs_once(db):
    # A second start must not try to create the tables again.
    db.init_db()
    conn = sqlite3.connect(db.DB_PATH)
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == \
        database.MIGRATIONS[-1][0]
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
    tid = db.add_task("call the bank", scheduled_for="2026-11-12")
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == [
        (tid, "call the bank", None)]


def test_finished_one_time_task_never_comes_back(db):
    tid = db.add_task("call the bank")
    db.complete_task(tid, now=datetime(2026, 11, 10, 9, 0))
    assert db.tasks_today(now=datetime(2026, 11, 12, 9, 0)) == []


def test_a_dated_one_time_task_shows_from_its_day_while_pending(db):
    """Migration 41: a task planned for Wednesday is not on Tuesday's list, one
    left from Monday still is, and a skipped or moved one is settled."""
    tuesday = datetime(2026, 11, 24, 12, 0)
    ids = database.add_planned_tasks([
        {"text": "Wednesday lab", "scheduled_for": "2026-11-25"},
        {"text": "Monday leftover", "scheduled_for": "2026-11-23"},
        {"text": "Tuesday reading", "scheduled_for": "2026-11-24"},
        {"text": "Skipped laundry", "scheduled_for": "2026-11-24"},
        {"text": "Moved essay", "scheduled_for": "2026-11-24"},
    ], tuesday)
    undated = database.add_task("Call the bank", now=tuesday)   # no day: the backlog
    database.set_task_status(ids[3], "skipped", tuesday)
    database.move_task(ids[4], "2026-11-26", tuesday)
    shown = [t[1] for t in database.tasks_today(tuesday)]
    assert shown == ["Monday leftover", "Tuesday reading"]
    assert "Wednesday lab" in [t[1] for t in database.tasks_today(datetime(2026, 11, 25, 9, 0))]
    assert [t["id"] for t in database.backlog_tasks()] == [undated]


def test_a_one_time_task_done_by_voice_is_done_in_the_review(db):
    task_id = database.add_task("Return the book")
    database.complete_task(task_id)
    assert database.get_task(task_id)["status"] == "done"
    assert task_id not in [t[0] for t in database.tasks_today()]


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
    b = db.add_task("call the bank", scheduled_for="2026-11-12")
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



def test_archive_task_keeps_the_row(db):
    tid = db.add_task("take vitamins", "daily")
    assert db.archive_task(tid, now=datetime(2026, 11, 12, 9, 0)) is True
    # Archiving twice finds nothing active to archive.
    assert db.archive_task(tid) is False
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute("SELECT text, archived_at FROM tasks WHERE id = ?",
                       (tid,)).fetchone()
    conn.close()
    assert row == ("take vitamins", "2026-11-12T09:00:00")


# ── The matching rule: exact wins, then a lone partial, else ask ──

def test_exact_item_match_wins_over_partial(db):
    lid = db.get_or_create_list("grocery")
    milk, _ = db.add_list_item(lid, "milk")
    db.add_list_item(lid, "oat milk")
    assert db.unchecked_items_matching(lid, "Milk") == [(milk, "milk")]


def test_exact_task_match_wins_over_partial(db):
    a = db.add_task("vitamins", "daily")
    db.add_task("buy vitamins")
    assert db.active_tasks_matching("vitamins") == [(a, "vitamins", "daily")]


def test_no_list_searches_every_list(db):
    g = db.get_or_create_list("grocery")
    h = db.get_or_create_list("hardware")
    eggs, _ = db.add_list_item(g, "eggs")
    nails, _ = db.add_list_item(h, "nails")
    assert db.unchecked_items_matching(None, "nails") == [(nails, "nails")]
    assert db.unchecked_items_matching(g, "nails") == []


# ── The tools, called the way Nova calls them ──

def _unchecked(db, name):
    return [row[1] for row in db.list_items(db.find_list(name))]


def test_list_names_ignore_my_the_and_list():
    for said in ("My Grocery List", "the grocery list", "grocery", " grocery "):
        assert lists_tasks.stored_list_name(said) == "grocery"


def test_add_eggs_to_my_grocery_list(db):
    result = registry.call("list_add", {"item": "eggs",
                                        "list_name": "my grocery list"})
    assert result == "Added Eggs to the grocery list."
    assert _unchecked(db, "grocery") == ["Eggs"]


def test_adding_eggs_again_is_refused_out_loud(db):
    registry.call("list_add", {"item": "eggs", "list_name": "grocery"})
    with pytest.raises(ToolNotice, match="already on the grocery list"):
        registry.call("list_add", {"item": "eggs", "list_name": "grocery"})
    assert _unchecked(db, "grocery") == ["Eggs"]


def test_check_off_milk_with_oat_milk_on_the_list_checks_milk(db):
    registry.call("list_add", {"item": "milk", "list_name": "grocery"})
    registry.call("list_add", {"item": "oat milk", "list_name": "grocery"})
    assert registry.call("list_check", {"item": "milk"}) == "Checked off Milk."
    assert _unchecked(db, "grocery") == ["Oat milk"]


def test_one_partial_match_is_checked_without_asking(db):
    registry.call("list_add", {"item": "oat milk", "list_name": "grocery"})
    assert registry.call("list_check", {"item": "oat"}) == "Checked off Oat milk."


def test_several_partial_matches_ask_and_check_nothing(db):
    registry.call("list_add", {"item": "oat milk", "list_name": "grocery"})
    registry.call("list_add", {"item": "whole milk", "list_name": "grocery"})
    with pytest.raises(ToolNotice,
                       match="More than one unchecked item matches 'milk': "
                             "Oat milk, Whole milk. Ask which one."):
        registry.call("list_check", {"item": "milk"})
    assert _unchecked(db, "grocery") == ["Oat milk", "Whole milk"]


def test_checking_off_something_not_on_the_list_is_refused(db):
    registry.call("list_add", {"item": "eggs", "list_name": "grocery"})
    with pytest.raises(ToolNotice, match="No unchecked item matches 'bread'"):
        registry.call("list_check", {"item": "bread"})


def test_checking_on_a_list_that_does_not_exist_is_refused(db):
    with pytest.raises(ToolNotice, match="There is no hardware list"):
        registry.call("list_check", {"item": "nails", "list_name": "hardware"})


def test_whats_on_my_grocery_list(db):
    for item in ("eggs", "milk", "bread"):
        registry.call("list_add", {"item": item, "list_name": "grocery"})
    registry.call("list_check", {"item": "milk"})
    assert registry.call("list_show", {"list_name": "my grocery list"}) == \
        "On the grocery list: Eggs, Bread."


def test_list_show_for_an_empty_or_missing_list(db):
    assert registry.call("list_show", {"list_name": "hardware"}) == \
        "There is no hardware list."
    assert db.find_list("hardware") is None
    registry.call("list_add", {"item": "eggs", "list_name": "grocery"})
    registry.call("list_check", {"item": "eggs"})
    assert registry.call("list_show", {"list_name": "grocery"}) == \
        "The grocery list is empty."


def test_daily_task_by_voice_then_done_then_retired(db):
    assert registry.call("task_add", {"task": "take my vitamins",
                                      "recurrence": "daily"}) == \
        "Added the daily task: Take my vitamins."
    registry.call("task_add", {"task": "call the bank"})
    assert registry.call("tasks_today", {}) == \
        "Still to do today: Take my vitamins, Call the bank."
    assert registry.call("task_complete", {"task": "vitamins"}) == \
        "Marked done: Take my vitamins."
    assert registry.call("tasks_today", {}) == "Still to do today: Call the bank."
    assert registry.call("archive_task", {"task": "bank"}) == \
        "Stopped tracking: Call the bank."
    assert registry.call("tasks_today", {}) == "Nothing left to do today."


def test_stop_tracking_vitamins(db):
    registry.call("task_add", {"task": "take my vitamins", "recurrence": "daily"})
    registry.call("archive_task", {"task": "vitamins"})
    assert registry.call("tasks_today", {}) == "Nothing left to do today."
    with pytest.raises(ToolNotice, match="No task matches 'vitamins'"):
        registry.call("task_complete", {"task": "vitamins"})


# ── Who may use them ──

@pytest.mark.parametrize("name", ["list_add", "list_check", "list_show"])
def test_family_can_use_lists_and_a_guest_cannot(name):
    spec = registry.get(name)
    assert not permits(spec, "genin")
    assert permits(spec, "chunin")



# ── Plurals find the list that exists, and never rename one ──

def test_groceries_adds_to_the_grocery_list(db):
    registry.call("list_add", {"item": "milk", "list_name": "grocery"})
    registry.call("list_add", {"item": "eggs", "list_name": "groceries"})
    [(lid, name, unchecked)] = db.all_lists()
    assert (name, unchecked) == ("grocery", 2)
    assert _unchecked(db, "grocery") == ["Milk", "Eggs"]


def test_a_trailing_s_finds_the_singular_list(db):
    lid = db.get_or_create_list("errand")
    assert db.get_or_create_list("errands") == lid
    assert db.find_list("errands") == lid


def test_a_plural_with_no_singular_list_is_created_as_said(db):
    lid = db.get_or_create_list("supplies")
    assert db.all_lists() == [(lid, "supplies", 0)]


def test_supply_adds_to_the_supplies_list(db):
    registry.call("list_add", {"item": "tape", "list_name": "supplies"})
    registry.call("list_add", {"item": "glue", "list_name": "supply"})
    [(lid, name, unchecked)] = db.all_lists()
    assert (name, unchecked) == ("supplies", 2)
    assert db.find_list("supply") == lid


def test_a_singular_finds_the_list_with_an_s(db):
    lid = db.get_or_create_list("errands")
    assert db.get_or_create_list("errand") == lid


def test_a_singular_with_no_plural_list_is_created_as_said(db):
    db.get_or_create_list("supplies")
    lid = db.get_or_create_list("hardware")
    assert db.find_list("hardware") == lid
    assert {row[1] for row in db.all_lists()} == {"supplies", "hardware"}


def test_whats_on_my_groceries_list_reads_grocery(db):
    registry.call("list_add", {"item": "eggs", "list_name": "grocery"})
    assert registry.call("list_show", {"list_name": "groceries"}) == \
        "On the groceries list: Eggs."


# ── Notices reach Nova like errors and are logged apart from them ──

class _Block:
    def __init__(self, name, arguments):
        self.name = name
        self.input = arguments


def _last_log(db):
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute("SELECT result, is_error, is_notice FROM tool_call_log "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return row


def test_already_on_the_list_reaches_nova_and_logs_as_a_notice(db):
    import brain
    registry.call("list_add", {"item": "eggs", "list_name": "grocery"})
    [result] = brain._run_tools(
        [_Block("list_add", {"item": "eggs", "list_name": "grocery"})],
        "model", "hokage")
    said = ("Eggs is already on the grocery list, so nothing was added. "
            "Say it was already there.")
    # Sent as an error, so the loop makes the second call and she says it.
    assert result["is_error"] is True
    assert result["output"] == said
    assert brain._needs_second_call([result])
    assert _last_log(db) == (said, 0, 1)


def test_a_real_failure_is_still_an_error(db):
    import brain
    [result] = brain._run_tools(
        [_Block("list_add", {"item": "eggs", "list_name": " "})],
        "model", "hokage")
    assert result["is_error"] is True
    assert result["output"] == "list_add failed: a list needs a name, like grocery"
    assert _last_log(db)[1:] == (1, 0)


def test_a_success_is_neither(db):
    import brain
    [result] = brain._run_tools(
        [_Block("list_add", {"item": "eggs", "list_name": "grocery"})],
        "model", "hokage")
    assert result["is_error"] is False
    assert _last_log(db) == ("Added Eggs to the grocery list.", 0, 0)


def test_migration_34_keeps_old_tool_calls_as_they_were(tmp_path, monkeypatch):
    path = str(tmp_path / "miles.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "MIGRATIONS", database.MIGRATIONS[:33])
    database.init_db()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO tool_call_log (created_at, tool_name, result, is_error) "
                 "VALUES ('2026-11-01T09:00:00', 'get_weather', 'down', 1)")
    conn.commit()
    conn.close()

    monkeypatch.undo()
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()

    conn = sqlite3.connect(path)
    row = conn.execute("SELECT tool_name, result, is_error, is_notice "
                       "FROM tool_call_log").fetchone()
    conn.close()
    assert row == ("get_weather", "down", 1, 0)


# ── Items and tasks read as written ──

@pytest.mark.parametrize("said, stored", [
    ("salt", "Salt"),
    ("  chicken broth ", "Chicken broth"),
    ("iPhone charger", "iPhone charger"),
    ("AA batteries", "AA batteries"),
    ("Eggs", "Eggs"),
    ("2 lemons", "2 lemons"),
])
def test_written_form(said, stored):
    assert lists_tasks.written_form(said) == stored


def test_items_are_stored_capitalized(db):
    registry.call("list_add", {"item": "salt", "list_name": "grocery"})
    registry.call("list_add", {"item": "chicken", "list_name": "grocery"})
    assert _unchecked(db, "grocery") == ["Salt", "Chicken"]


def test_a_lowercase_repeat_is_still_a_duplicate(db):
    registry.call("list_add", {"item": "Salt", "list_name": "grocery"})
    with pytest.raises(ToolNotice):
        registry.call("list_add", {"item": "salt", "list_name": "grocery"})
    assert _unchecked(db, "grocery") == ["Salt"]


def test_checking_off_finds_a_capitalized_item(db):
    registry.call("list_add", {"item": "salt", "list_name": "grocery"})
    registry.call("list_check", {"item": "salt"})
    assert _unchecked(db, "grocery") == []


def test_tasks_are_stored_capitalized(db):
    registry.call("task_add", {"task": "take my vitamins", "recurrence": "daily"})
    assert [t[1] for t in db.tasks_today()] == ["Take my vitamins"]
