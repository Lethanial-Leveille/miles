"""No test run can reach the production database.

conftest.py sets MILES_DB_PATH before any import and wraps sqlite3.connect so
the production file cannot be opened; pytest_sessionfinish fails the run if
anything even tried. These tests prove each layer, so removing one is caught."""

import os
import sqlite3

import pytest

import actions
import config
import conftest
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
    before = len(conftest.PRODUCTION_DB_ATTEMPTS)
    with pytest.raises(RuntimeError, match="production database"):
        sqlite3.connect(target, uri=target.startswith("file:"))
    assert len(conftest.PRODUCTION_DB_ATTEMPTS) == before + 1
    # These attempts are the test's own; take them back so the run's final
    # check only fails on attempts nobody meant to make.
    del conftest.PRODUCTION_DB_ATTEMPTS[before:]


def test_other_databases_still_open(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "fine.db"))
    conn.execute("SELECT 1")
    conn.close()
    sqlite3.connect(":memory:").close()
