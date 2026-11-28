import pytest

from runtime_profile import current

if current() is None:
    raise RuntimeError("Use ./scripts/dev test")

import database


@pytest.fixture(autouse=True)
def no_real_google(monkeypatch):
    """No test may read or write his real calendar. The nightly review builds
    its opening from the calendar, so without this a test creating a session
    on the Pi would call Google with the live token. A test that wants a
    calendar installs its own (tests/fake_google.py), which replaces this."""
    import calendar_tools

    def refuse():
        raise FileNotFoundError("no Google token; tests never reach the real calendar")
    monkeypatch.setattr(calendar_tools, "_service", refuse)
    calendar_tools._calendar_cache.clear()


@pytest.fixture(autouse=True)
def quiet_evallo_filter(monkeypatch, request):
    """Calendar tests that never set up a database should neither read one nor
    hear the Evallo "not verified" line. Their state is filter off, outcome
    active. test_evallo_filter.py uses the real state loader."""
    if request.module.__name__.endswith("test_evallo_filter"):
        return
    import evallo_filter

    class Quiet(evallo_filter.State):
        def outcome(self, now):
            return "active"

    monkeypatch.setattr(evallo_filter, "load_state", lambda: Quiet())


@pytest.fixture(autouse=True)
def fake_intent_model(monkeypatch):
    """No test reaches the intent model. This stands in with simple patterns;
    a test that needs another answer patches task_intent.request_intent."""
    import task_intent
    from intent_fake import fake_intent

    async def request(client, utterance, pending_title=None, followup=None):
        return fake_intent(utterance, followup)

    monkeypatch.setattr(task_intent, "request_intent", request)


@pytest.fixture(autouse=True)
def forget_last_exchange():
    """The last exchange per device is process memory; no test inherits another's."""
    import brain

    brain._last_exchange.clear()
    yield
    brain._last_exchange.clear()


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
    import embeddings
    from dev_providers import encode

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
