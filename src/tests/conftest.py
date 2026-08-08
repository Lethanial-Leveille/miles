import os
import sys

# Test files live in src/tests/; the modules under test live in src/, one
# directory up. Insert it so plain `import database` etc. resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import database


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point database.py at a throwaway sqlite file for the duration of one
    test and run migrations against it, so tests never touch the real
    ~/miles/data/miles.db."""
    test_db_path = str(tmp_path / "test_miles.db")
    monkeypatch.setattr(database, "DB_PATH", test_db_path)
    database.init_db()
    return database
