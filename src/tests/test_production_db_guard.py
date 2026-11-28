"""No test run can reach the production database.

The launcher installs the SQLite boundary before app and pytest imports. These tests prove each layer, so removing one is caught."""

import os
import sqlite3

import pytest

import actions
import config
from runtime_profile import current
import database
import system_state

PRODUCTION = os.path.realpath(os.path.expanduser("~/miles/data/miles.db"))


def test_the_one_path_function_never_answers_production_in_tests():
    assert os.environ.get("MILES_DB_PATH")
    assert os.path.realpath(config.db_path()) != PRODUCTION


def test_import_time_copies_are_not_production():
    # Taken at import, before any fixture ran: these are what a module level
    # init_db() would have used.
    assert os.path.realpath(config.DB_PATH) != PRODUCTION
    assert os.path.realpath(database.DB_PATH) != PRODUCTION


def test_the_db_fixture_moves_every_reader_together(db):
    assert config.db_path() == database.DB_PATH
    assert actions.db_path() == database.DB_PATH
    assert system_state.db_path() == database.DB_PATH


@pytest.mark.parametrize("target", [
    PRODUCTION,
    "~/miles/data/miles.db",
    f"file:{PRODUCTION}?mode=ro",
])
def test_opening_production_raises_and_is_recorded(target, monkeypatch):
    violations = current().violations
    before = len(violations)
    with pytest.raises(RuntimeError, match="SQLite"):
        sqlite3.connect(target, uri=target.startswith("file:"))
    assert len(violations) == before + 1
    # These attempts are the test's own; take them back so the run's final
    # check only fails on attempts nobody meant to make.
    del violations[before:]


def test_other_databases_still_open(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "fine.db"))
    conn.execute("SELECT 1")
    conn.close()
    sqlite3.connect(":memory:").close()
