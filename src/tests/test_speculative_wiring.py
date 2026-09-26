"""When a speculative Claude call may start, and how the real turn takes it over.

The gate has to agree with voice_main about every turn: a speculation for text the
real turn would never send is wasted, and one for a voice the real check refuses
would send his personal prompt for someone else. The claim has to agree with the
speculation about the request, or every speculation is silently thrown away.
"""

import ast
import asyncio
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

import alerts
import brain
import config
import speculative_turn as st

SRC = pathlib.Path(__file__).resolve().parent.parent
PRINT = np.array([1.0, 0.0])
HIM = np.array([1.0, 0.0])           # cosine 1.0 with PRINT
SOMEONE = np.array([0.0, 1.0])       # cosine 0.0


# ── the text the real turn would send ──

@pytest.mark.parametrize("text,turn_type,expected", [
    ("What's on my calendar tomorrow?", "initial", ("What's on my calendar tomorrow?", False)),
    ("Hey Nova, what's on my calendar?", "initial", ("what's on my calendar?", False)),
    ("Hey Nova.", "initial", None),                       # voice_main records again
    ("", "initial", None),                                # noise
    ("What about Thursday?", "followup", ("What about Thursday?", True)),
    ("Hey Nova, set a timer", "followup", ("set a timer", False)),  # fresh turn, scored
    ("Hey Nova", "followup", None),
    ("", "followup", None),
    (None, "initial", None),                              # Deepgram failed
])
def test_the_text_is_what_voice_main_would_send(text, turn_type, expected):
    derived = st.user_text_for(text, turn_type)
    if expected is None:
        assert derived is None
    else:
        assert derived[1] == expected[1]
        assert derived[0].lower().strip(" ,.") == expected[0].lower().strip(" ,.")


# ── the gate ──

@pytest.fixture
def gate(monkeypatch):
    started = []

    class _Turn:
        def __init__(self, make_request):
            self.make_request = make_request
            self.cancelled = False
            started.append(self)

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(st, "SpeculativeTurn", _Turn)
    monkeypatch.setattr(st, "SPECULATIVE_CLAUDE", True)
    monkeypatch.setattr(st, "MODEL_AB_TEST", False)
    monkeypatch.setattr(st.local_intent, "is_warm", lambda: True)
    monkeypatch.setattr(st.local_intent, "classify", lambda text: None)
    monkeypatch.setattr(st, "_current", None)
    return started


def _consider(text="What's on my calendar tomorrow?", turn_type="initial",
              seconds=3.0, embedding=HIM):
    return st.consider(text, turn_type, seconds, embedding, PRINT)


def test_a_verified_command_for_claude_starts_one(gate):
    assert _consider() is gate[0]


@pytest.mark.parametrize("why,change", [
    ("switched off", lambda m: m.setattr(st, "SPECULATIVE_CLAUDE", False)),
    ("model unknowable during an A/B", lambda m: m.setattr(st, "MODEL_AB_TEST", True)),
    ("intent examples not encoded yet", lambda m: m.setattr(st.local_intent, "is_warm", lambda: False)),
    ("answered locally", lambda m: m.setattr(st.local_intent, "classify", lambda t: "set_timer")),
])
def test_it_does_not_start_when(gate, monkeypatch, why, change):
    change(monkeypatch)
    assert _consider() is None and gate == [], why


def test_a_voice_that_does_not_verify_gets_nothing(gate):
    """His personal prompt is never sent on behalf of someone else."""
    assert _consider(embedding=SOMEONE) is None and gate == []


def test_too_little_voice_gets_nothing(gate):
    assert _consider(seconds=0.2) is None


def test_a_short_follow_up_is_trusted_on_the_session_like_the_real_check(gate):
    assert _consider("Thursday?", "followup", seconds=1.0, embedding=SOMEONE) is gate[0]


def test_a_newer_speculation_replaces_and_cancels_the_older(gate):
    first = _consider()
    second = _consider("And Friday?")
    assert first.cancelled and not second.cancelled
    assert st.take() is second and st.take() is None


def test_discard_cancels_what_is_pending(gate):
    turn = _consider()
    st.discard()
    assert turn.cancelled and st.take() is None


# ── the request it builds, against the one the real turn builds ──

ROWS = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i}"}
        for i in range(30)]


@pytest.fixture
def memories(monkeypatch):
    monkeypatch.setattr(brain, "get_seed_memories", lambda: [(1, "Studies engineering.", "school")])
    monkeypatch.setattr(brain, "get_episodic_memories", lambda: [(2, "Exam Thursday.")])
    monkeypatch.setattr(brain, "memory_manifest", lambda: [("school", 1)])
    monkeypatch.setattr(brain, "get_shareable_memories", lambda: [])
    monkeypatch.setattr(brain, "build_enhanced_prompt", lambda *a, **k: f"prompt {a[2]} {k}")
    monkeypatch.setattr(brain, "search_memories", lambda *a, **k: [])
    monkeypatch.setattr(brain, "effective_tier", lambda: "hokage")
    monkeypatch.setattr(st, "get_recent_messages", lambda limit: ROWS[-limit:])


def test_unchanged_it_fingerprints_exactly_as_the_real_turn_will(gate, memories):
    """If these ever differ, every speculation is thrown away and nothing says so."""
    text = "What's on my calendar tomorrow?"
    _consider(text)
    spec = gate[0].make_request()
    real_history = (ROWS + [{"role": "user", "content": text}])[-20:]
    # A first turn: the real one is named, as voice_main passes named=True.
    real = brain.turn_fingerprint(text, "voice", "hokage", config.MODEL_A, real_history,
                                  alerts.take_for_fold(), named=True)
    assert spec.fingerprint == real
    assert spec.messages[-1]["content"].startswith(text)


def test_a_message_saved_in_between_changes_the_fingerprint(gate, memories):
    """The app, or another turn, wrote to history after the speculation read it."""
    text = "What's on my calendar tomorrow?"
    _consider(text)
    spec = gate[0].make_request()
    moved = (ROWS + [{"role": "assistant", "content": "from the app"},
                     {"role": "user", "content": text}])[-20:]
    assert spec.fingerprint != brain.turn_fingerprint(
        text, "voice", "hokage", config.MODEL_A, moved, [])


def test_building_it_leaves_alerts_and_the_retrieval_log_alone(gate, memories, monkeypatch):
    logged = []
    monkeypatch.setattr(brain, "search_memories", lambda *a, **k: logged.append(k["log"]) or [])
    fired = SimpleNamespace(summary="Timer done.", fired_at=__import__("time").monotonic(),
                            kind="timer", fired_wall=None)
    monkeypatch.setattr(alerts, "_pending", [fired])
    _consider()
    gate[0].make_request()
    assert alerts._pending == [fired], "peeked, not taken"
    assert logged == [False]


def test_peeking_returns_what_taking_would(monkeypatch):
    now = __import__("time").monotonic()
    fresh = SimpleNamespace(summary="fresh", fired_at=now, kind="timer", fired_wall=None)
    stale = SimpleNamespace(summary="stale", fired_at=now - 3600, kind="timer", fired_wall=None)
    monkeypatch.setattr(alerts, "_pending", [fresh, stale])
    monkeypatch.setattr(alerts, "log_alert", lambda **k: None)
    assert alerts.peek_for_fold() == [fresh]
    assert alerts.take_for_fold() == [fresh]


# ── the real turn taking it over ──

class _Stream:
    def __init__(self, text):
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield SimpleNamespace(type="content_block_delta",
                              delta=SimpleNamespace(type="text_delta", text=self.text))

    async def get_final_message(self):
        return SimpleNamespace(content=[], usage=SimpleNamespace(
            cache_read_input_tokens=0, cache_creation_input_tokens=0))


@pytest.fixture
def turn(monkeypatch, memories):
    calls = []
    recall_later = []
    monkeypatch.setattr(brain, "speak", lambda *a, **k: None)
    monkeypatch.setattr(brain, "start_synthesis", lambda text, *a, **k: text)
    monkeypatch.setattr(brain, "_play_with_barge_in", lambda s: (False, 0.0))
    monkeypatch.setattr(brain, "save_message", lambda *a, **k: 7)
    monkeypatch.setattr(brain, "get_recent_messages", lambda limit=20: ROWS[-limit:])
    monkeypatch.setattr(brain.alerts, "take_for_fold", lambda: [])
    monkeypatch.setattr(brain.memory_pass, "notice_later", lambda *a, **k: None)
    monkeypatch.setattr(brain, "_log_recall_later", recall_later.append)
    monkeypatch.setattr(brain, "claude", SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **k: (calls.append(k), _Stream("called after the endpoint"))[1])))

    def run(speculation):
        result = asyncio.run(brain.ask_nova_async("hi", device="pi", channel="voice",
                                                  tier="hokage", speculation=speculation))
        return result, calls, recall_later
    return run


class _Speculation:
    def __init__(self, fingerprint):
        self.fingerprint = fingerprint
        self.request = brain.TurnRequest("claude-haiku-4-5", "system",
                                         [{"role": "user", "content": "hi"}], fingerprint)
        self.opened_at = __import__("time").monotonic()
        self.claimed_with = None

    def claim(self, fingerprint):
        self.claimed_with = fingerprint
        return _Stream("from the speculation") if fingerprint == self.fingerprint else None


def _real_fingerprint():
    return brain.turn_fingerprint("hi", "voice", "hokage", config.MODEL_A, ROWS[-20:], [])


def test_a_matching_speculation_is_played_and_claude_is_not_called_again(turn):
    speculation = _Speculation(_real_fingerprint())
    result, calls, recall_later = turn(speculation)
    assert result.text == "from the speculation"
    assert calls == []
    assert recall_later == ["hi"], "retrieval_log still gets its row"


def test_a_mismatched_speculation_is_ignored_and_the_call_made_here(turn):
    result, calls, recall_later = turn(_Speculation("stale"))
    assert result.text == "called after the endpoint"
    assert len(calls) == 1 and recall_later == []


def test_no_speculation_is_today_exactly(turn):
    result, calls, _ = turn(None)
    assert result.text == "called after the endpoint" and len(calls) == 1


# ── the wiring, read as text where the module cannot be imported ──

def _function(path, name):
    tree = ast.parse((SRC / path).read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def _calls(node):
    return {ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}


def test_every_recording_start_drops_a_pending_speculation():
    assert "speculative_turn.discard" in _calls(_function("audio.py", "cancel_speculation"))


def test_the_deepgram_transcript_is_what_starts_one():
    source = ast.unparse(_function("audio.py", "_Speculation"))
    assert "on_done=self._on_transcript" in source
    assert "speculative_turn.consider(" in source


def test_speaking_again_cancels_the_claude_call_too():
    cancel = next(n for n in ast.walk(_function("audio.py", "_Speculation"))
                  if isinstance(n, ast.FunctionDef) and n.name == "cancel")
    assert "turn.cancel" in _calls(cancel)


def test_the_voice_turn_hands_it_over_and_always_cancels_it():
    source = ast.unparse(_function("voice_main.py", "run_turn"))
    assert "speculative_turn.take()" in source
    assert "ask_nova(text, speculation=speculation, named=named)" in source
    assert "finally:" in source and "speculation.cancel()" in source.split("finally:")[1]


def test_a_claim_records_how_early_the_call_went_out(turn, monkeypatch):
    marks = {}
    monkeypatch.setattr(brain.timing, "mark", lambda stage, ms: marks.setdefault(stage, ms))
    speculation = _Speculation(_real_fingerprint())
    speculation.opened_at -= 0.5
    turn(speculation)
    assert 450 <= marks["claude_head_start_ms"] < 2000


def test_an_unclaimed_turn_records_no_head_start(turn, monkeypatch):
    marks = {}
    monkeypatch.setattr(brain.timing, "mark", lambda stage, ms: marks.setdefault(stage, ms))
    turn(_Speculation("stale"))
    assert "claude_head_start_ms" not in marks



# ── the named turn marker (Sep 26 2026) ──

@pytest.mark.parametrize("text,turn_type,named", [
    ("What's on my calendar?", "initial", True),
    ("Hey Nova, what's on my calendar?", "initial", True),
    ("What about Thursday?", "followup", False),
    ("Hey Nova, set a timer", "followup", True),
])
def test_named_follows_voice_main(text, turn_type, named):
    assert st.user_text_for(text, turn_type)[2] is named


def test_a_named_turn_says_so_after_the_clock_breakpoint():
    messages = [{"role": "user", "content": "Is Trump a good president?"}]
    noted = brain._with_named_note(messages, True)
    assert noted[-1]["content"].endswith(brain.NAMED_TURN_NOTE)
    assert messages[-1]["content"] == "Is Trump a good president?", "the stored message stays raw"
    assert brain._with_named_note(messages, False) == messages


def test_the_note_reaches_the_request_and_the_fingerprint(memories):
    args = ("Is Trump a good president?", "voice", "hokage", config.MODEL_A,
            [{"role": "user", "content": "Is Trump a good president?"}], [])
    named = brain.build_request(*args, log_recall=False, named=True)
    plain = brain.build_request(*args, log_recall=False, named=False)
    assert brain.NAMED_TURN_NOTE in named.messages[-1]["content"]
    assert brain.NAMED_TURN_NOTE not in plain.messages[-1]["content"]
    assert named.fingerprint != plain.fingerprint


def test_the_voice_loop_names_first_turns_and_wake_word_follow_ups():
    source = (SRC / "voice_main.py").read_text()
    assert "run_turn(user_text, named=True)" in source
    assert "run_turn(followup_text, named=said_wake)" in source


def test_the_prompt_and_the_ignore_tool_both_carry_the_rule():
    import prompts
    from tools import registry
    assert brain.NAMED_TURN_NOTE in prompts.NOT_ADDRESSED_TO_YOU
    assert brain.NAMED_TURN_NOTE in registry.get("ignore").description
