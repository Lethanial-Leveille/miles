import os
import sqlite3
import sys
import tempfile

# Before anything from src/ is imported, so config.db_path() can never answer
# with the production file inside a test run. server.py and modules it pulls in
# call init_db() at import; without this the suite once applied a branch's new
# migrations to the live ~/miles/data/miles.db (Nov 24 2026).
_IMPORT_TIME_DIR = tempfile.mkdtemp(prefix="miles_tests_")
os.environ["MILES_DB_PATH"] = os.path.join(_IMPORT_TIME_DIR, "import_time.db")

# The guard behind that rule: any sqlite3.connect to the production file, from
# any module, raises and is recorded. Recorded as well as raised because some
# code swallows exceptions on purpose (a voiceprint sample "never raises"), and
# a swallowed attempt must still fail the run. pytest_sessionfinish fails the
# whole run if the list is not empty.
PRODUCTION_DB = os.path.realpath(os.path.expanduser("~/miles/data/miles.db"))
PRODUCTION_DB_ATTEMPTS = []
_real_connect = sqlite3.connect


def _target(database):
    path = os.fspath(database) if not isinstance(database, str) else database
    if path.startswith("file:"):
        path = path[len("file:"):].split("?", 1)[0]
    return os.path.realpath(os.path.expanduser(path))


def _guarded_connect(database, *args, **kwargs):
    if database != ":memory:" and _target(database) == PRODUCTION_DB:
        import traceback
        PRODUCTION_DB_ATTEMPTS.append("".join(traceback.format_stack(limit=8)))
        raise RuntimeError(f"a test tried to open the production database {PRODUCTION_DB}")
    return _real_connect(database, *args, **kwargs)


sqlite3.connect = _guarded_connect

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
    # Modules that ask config.db_path() at call time follow it too.
    monkeypatch.setenv("MILES_DB_PATH", test_db_path)
    database.init_db()
    return database


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Words hashed into 64 buckets stand in for the sentence model: sentences
    sharing words point the same way, and nothing loads 80MB of weights."""
    import zlib
    import numpy as np
    import embeddings

    def encode(texts):
        out = []
        for text in texts:
            v = np.zeros(64, dtype=np.float32)
            for word in text.lower().replace(".", "").split():
                v[zlib.crc32(word.encode()) % 64] += 1.0
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(out)

    monkeypatch.setattr(embeddings, "encode", encode)
    return encode


# ── a stand in for the Anthropic client, for the session engine ──

def text_block(text):
    from types import SimpleNamespace
    return SimpleNamespace(type="text", text=text)


def tool_block(name, arguments=None, block_id=None):
    from types import SimpleNamespace
    return SimpleNamespace(type="tool_use", id=block_id or f"toolu_{name}",
                           name=name, input=dict(arguments or {}))


class _FakeSessionStream:
    """messages.stream as the engine uses it: a context manager that yields a
    block start for each block, the text in two pieces, then the final message."""

    def __init__(self, blocks):
        from types import SimpleNamespace
        self._blocks = blocks
        self._final = SimpleNamespace(
            content=list(blocks),
            usage=SimpleNamespace(input_tokens=120, output_tokens=30,
                                  cache_read_input_tokens=100, cache_creation_input_tokens=0))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        from types import SimpleNamespace
        for block in self._blocks:
            yield SimpleNamespace(type="content_block_start", content_block=block)
            if block.type == "text":
                half = len(block.text) // 2
                for piece in (block.text[:half], block.text[half:]):
                    yield SimpleNamespace(type="content_block_delta",
                                          delta=SimpleNamespace(type="text_delta", text=piece))

    def get_final_message(self):
        return self._final


class FakeSessionClaude:
    """Plays back one list of blocks per round and records every request."""

    def __init__(self, rounds, summary="He finished the lab and planned tomorrow."):
        import copy
        self._copy = copy.deepcopy
        self.rounds = list(rounds)
        self.summary = summary
        self.calls = []
        self.summary_calls = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(self._copy(kwargs))
        return _FakeSessionStream(self.rounds.pop(0))

    def create(self, **kwargs):
        from types import SimpleNamespace
        self.summary_calls.append(self._copy(kwargs))
        return SimpleNamespace(content=[text_block(self.summary)])


@pytest.fixture
def fake_claude(monkeypatch):
    """fake_claude(round, round, ...) installs a fake client for sessions.py,
    each round a list of text_block and tool_block, and returns it."""
    import sessions

    def install(*rounds, summary="He finished the lab and planned tomorrow."):
        fake = FakeSessionClaude(rounds, summary)
        monkeypatch.setattr(sessions, "_client", lambda: fake)
        return fake

    return install


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if anything tried to open the production database, even
    an attempt some code caught and ignored."""
    if PRODUCTION_DB_ATTEMPTS:
        print(f"\n{len(PRODUCTION_DB_ATTEMPTS)} attempt(s) to open {PRODUCTION_DB} "
              f"during the test run. First one:\n{PRODUCTION_DB_ATTEMPTS[0]}")
        session.exitstatus = 1
