"""A voice that is not his is a guest: answered, but none of his, and nothing above genin.

Until Sep 26 2026 an unrecognized voice was refused outright. He wanted friends
to be able to ask questions, with Nova knowing it is not him and gating anything
personal. The prompt side existed (GUEST_BLOCK) but guests were unreachable, and
the history still carried his last twenty messages, which would have made the
guest prompt's "those facts are not in front of you" false.
"""

import ast
import asyncio
import pathlib
from types import SimpleNamespace

import pytest

import brain
import database
import prompts
from tools import registry, permits

SRC = pathlib.Path(__file__).resolve().parent.parent
HIS = [{"role": "user", "content": "I interviewed at ReliaQuest yesterday."},
       {"role": "assistant", "content": "How did it go?"}]
GUESTS = [{"role": "user", "content": "Is Trump a good president?"}]


# ── the database ──

def test_messages_after_a_start_are_only_that_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    database.init_db()
    database.save_message("user", "his private thing")
    start = database.latest_message_id()
    database.save_message("user", "guest question")
    database.save_message("assistant", "guest answer")
    assert database.get_messages_after(start) == [
        {"role": "user", "content": "guest question"},
        {"role": "assistant", "content": "guest answer"}]


def test_an_empty_history_starts_at_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    database.init_db()
    assert database.latest_message_id() == 0


# ── what a guest turn sees ──

def test_his_turns_keep_his_history(monkeypatch):
    monkeypatch.setattr(brain, "get_recent_messages", lambda limit=20: HIS)
    assert brain._history_for("hokage", 5, "x") == HIS


def test_a_guest_sees_only_their_conversation(monkeypatch):
    seen = []
    monkeypatch.setattr(brain, "get_messages_after",
                        lambda after, limit=20: (seen.append(after), GUESTS)[1])
    assert brain._history_for("genin", 41, "x") == GUESTS
    assert seen == [41]


def test_a_guest_with_no_start_sees_only_this_message():
    assert brain._history_for("genin", None, "hi") == [{"role": "user", "content": "hi"}]


class _Stream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield SimpleNamespace(type="content_block_delta",
                              delta=SimpleNamespace(type="text_delta", text="Fine."))

    async def get_final_message(self):
        return SimpleNamespace(content=[], usage=SimpleNamespace(
            cache_read_input_tokens=0, cache_creation_input_tokens=0))


def test_a_guest_turn_sends_nothing_of_his(monkeypatch):
    sent = []
    monkeypatch.setattr(brain, "speak", lambda *a, **k: None)
    monkeypatch.setattr(brain, "start_synthesis", lambda text, *a, **k: text)
    monkeypatch.setattr(brain, "_play_with_barge_in", lambda s: (False, 0.0))
    monkeypatch.setattr(brain, "save_message", lambda *a, **k: 7)
    monkeypatch.setattr(brain, "get_recent_messages", lambda limit=20: HIS)
    monkeypatch.setattr(brain, "get_messages_after", lambda after, limit=20: GUESTS)
    monkeypatch.setattr(brain, "get_seed_memories", lambda: [(1, "PRIVATE SEED", "x")])
    monkeypatch.setattr(brain, "get_episodic_memories", lambda: [(2, "PRIVATE EPISODE")])
    monkeypatch.setattr(brain, "memory_manifest", lambda: [("x", 1)])
    monkeypatch.setattr(brain, "search_memories", lambda *a, **k: [(9, "PRIVATE RECALL", "x")])
    monkeypatch.setattr(brain.alerts, "take_for_fold", lambda: [])
    monkeypatch.setattr(brain.memory_pass, "notice_later",
                        lambda *a, **k: pytest.fail("a guest's words are not his to keep"))
    monkeypatch.setattr(brain, "claude", SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **k: (sent.append(k), _Stream())[1])))
    asyncio.run(brain.ask_nova_async("Is Trump a good president?", tier="genin",
                                     conversation_start=41))
    request = str(sent[0]["system"]) + str(sent[0]["messages"])
    for private in ("ReliaQuest", "PRIVATE SEED", "PRIVATE EPISODE", "PRIVATE RECALL"):
        assert private not in request
    assert "This is not Lethanial" in request


def test_guests_stay_quiet_for_speech_not_aimed_at_her():
    """A false wake on friends talking is a guest turn now."""
    prompt = prompts.build_enhanced_prompt([], [], "voice", [], tier="genin")
    assert prompts.GUEST_BLOCK in prompt
    assert prompts.NOT_ADDRESSED_TO_YOU in prompt
    assert prompts.HOW_TO_TREAT_HIM not in prompt


# ── local intent, which skips the permission gate ──

def _local_allowed():
    """voice_main opens the microphone on import, so the function is lifted out
    of its source and run on its own."""
    tree = ast.parse((SRC / "voice_main.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_local_allowed")
    scope = {"registry": registry, "permits": permits}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "voice_main", "exec"), scope)
    return scope["_local_allowed"]


@pytest.mark.parametrize("intent,allowed", [
    ("set_timer", False), ("cancel_reminder", False),
    ("dismiss", True), ("time_of_day", True),
])
def test_a_guest_gets_only_the_local_answers_genin_allows(intent, allowed):
    assert _local_allowed()(SimpleNamespace(name=intent), "genin") is allowed


def test_he_gets_every_local_answer():
    assert _local_allowed()(SimpleNamespace(name="set_timer"), None) is True


# ── the voice loop ──

def test_an_unrecognized_voice_is_answered_not_refused():
    source = (SRC / "voice_main.py").read_text()
    assert "I don't recognize your voiceprint" not in source
    assert "I don't recognize that voice" not in source
    assert "tier=None if session_is_him else GUEST_TIER" in source
    assert source.count("speculative_turn.set_session(session_is_him)") == 2


def test_a_guests_follow_up_is_never_trusted_on_the_session():
    tree = ast.parse((SRC / "voice_main.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "audio.verify_voice"
             and any(k.arg == "turn_type" and ast.unparse(k.value) == "'followup'"
                     for k in n.keywords)]
    assert len(calls) == 1
    trusted = next(k for k in calls[0].keywords if k.arg == "session_trusted")
    assert ast.unparse(trusted.value) == "not said_wake and session_is_him"
