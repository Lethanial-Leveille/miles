"""The request a turn sends comes from build_request, which has no side effects.

build_request pulled the request out of ask_nova_async so a speculative turn can
build the same one before the endpoint without saving, consuming or logging
anything. A speculation only saves time if its request is the one the real turn
would have sent, so the real turn has to send exactly what build_request builds.

Checked once, when the refactor landed on Sep 25 2026: four recorded requests
(voice with recall and an alert, text, jonin, guest) were byte for byte the same
before and after. That recording is not kept, because it holds the whole prompt
and every tool schema and would break on any edit to either.

The memories are invented. This repo is public.
"""

import asyncio
import json
from datetime import datetime as real_datetime
from types import SimpleNamespace

import pytest

import brain
from tools import registry

SEED      = [(1, "Studies computer engineering.", "school"),
             (2, "Lifts on Monday, Wednesday and Friday.", "health")]
EPISODIC  = [(3, "Has a physics exam next Thursday.")]
MANIFEST  = [("school", 4), ("health", 2)]
SHAREABLE = [(1, "Studies computer engineering.", "school")]
RECALLED  = [(9, "Prefers morning study blocks.", "habit"),
             (10, "Tutors on Tuesdays.", "work")]
HISTORY   = [{"role": "user", "content": "what's the weather"},
             {"role": "assistant", "content": " ".join(["word"] * 45)},
             {"role": "user", "content": "how did I sleep"}]

SCENARIOS = {
    "voice_hokage_recall_alert": dict(channel="voice", tier="hokage", recalled=RECALLED,
                                      alerts=["Your ten minute timer is done."]),
    "text_hokage":               dict(channel="text", tier="hokage", recalled=[], alerts=[]),
    "voice_jonin":               dict(channel="voice", tier="jonin", recalled=RECALLED, alerts=[]),
    "voice_guest":               dict(channel="voice", tier="genin", recalled=RECALLED, alerts=[]),
}


class _FixedClock(real_datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 25, 21, 5)


class _Stream:
    def __init__(self):
        self._final = SimpleNamespace(content=[], usage=SimpleNamespace(
            cache_read_input_tokens=0, cache_creation_input_tokens=0))

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
        return self._final


def capture(monkeypatch, channel, tier, recalled, alerts):
    """Run one turn against fakes and return what went to messages.stream."""
    sent = []
    recall_logged = capture.recall_logged = []
    monkeypatch.setattr(brain, "datetime", _FixedClock)
    monkeypatch.setattr(brain, "speak", lambda *a, **k: None)
    monkeypatch.setattr(brain, "start_synthesis", lambda text, *a, **k: text)
    monkeypatch.setattr(brain, "_play_with_barge_in", lambda s: (False, 0.0))
    monkeypatch.setattr(brain, "save_message", lambda *a, **k: 7)
    monkeypatch.setattr(brain, "last_reply_at", lambda: None)
    monkeypatch.setattr(brain, "get_recent_messages", lambda limit=20: list(HISTORY))
    monkeypatch.setattr(brain, "get_seed_memories", lambda: list(SEED))
    monkeypatch.setattr(brain, "get_episodic_memories", lambda: list(EPISODIC))
    monkeypatch.setattr(brain, "memory_manifest", lambda: list(MANIFEST))
    monkeypatch.setattr(brain, "get_shareable_memories", lambda: list(SHAREABLE))
    monkeypatch.setattr(brain, "search_memories",
                        lambda *a, **k: (recall_logged.append(k.get("log")), list(recalled))[1])
    monkeypatch.setattr(brain.alerts, "take_for_fold",
                        lambda: [SimpleNamespace(summary=s) for s in alerts])
    monkeypatch.setattr(brain.memory_pass, "notice_later", lambda *a, **k: None)
    monkeypatch.setattr(brain, "_select_model", lambda: "claude-haiku-4-5")
    monkeypatch.setattr(brain, "claude", SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **kwargs: (sent.append(kwargs), _Stream())[1])))
    asyncio.run(brain.ask_nova_async("how did I sleep", device="pi",
                                     channel=channel, tier=tier))
    return json.loads(json.dumps(sent[0], sort_keys=True, default=str))


def _built(channel, tier, recalled, alerts):
    # Below hokage a turn sees only its own conversation; with no start given,
    # only this message (brain._history_for).
    history = (list(HISTORY) if tier == "hokage"
               else [{"role": "user", "content": "how did I sleep"}])
    request = brain.build_request(
        "how did I sleep", channel, tier, "claude-haiku-4-5", history,
        [SimpleNamespace(summary=a) for a in alerts])
    return request, json.loads(json.dumps(
        {"model": request.model, "system": request.system, "messages": request.messages},
        sort_keys=True, default=str))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_turn_sends_exactly_what_build_request_builds(monkeypatch, name):
    sent = capture(monkeypatch, **SCENARIOS[name])
    _, built = _built(**SCENARIOS[name])
    assert {k: sent[k] for k in ("model", "system", "messages")} == built


def test_building_a_request_changes_nothing(monkeypatch):
    """A speculation that is discarded must leave no trace: no saved message, no
    consumed alert, no confirmation clock tick, no retrieval_log row."""
    capture(monkeypatch, **SCENARIOS["voice_hokage_recall_alert"])
    def forbidden(*a, **k):
        raise AssertionError("build_request has a side effect")
    for name in ("save_message", "delete_message"):
        monkeypatch.setattr(brain, name, forbidden)
    monkeypatch.setattr(brain.alerts, "take_for_fold", forbidden)
    monkeypatch.setattr(brain.pending_action, "begin_turn", forbidden)
    logged = []
    monkeypatch.setattr(brain, "search_memories",
                        lambda *a, **k: (logged.append(k.get("log")), list(RECALLED))[1])
    brain.build_request("how did I sleep", "voice", "hokage", "claude-haiku-4-5",
                        list(HISTORY), [], log_recall=False)
    assert logged == [False]


def test_a_real_turn_still_logs_what_recall_returned(monkeypatch):
    capture(monkeypatch, **SCENARIOS["voice_hokage_recall_alert"])
    assert capture.recall_logged == [True]


def _print(**changes):
    args = dict(user_text="how did I sleep", channel="voice", tier="hokage",
                model="claude-haiku-4-5", prompt="prompt", history=list(HISTORY),
                pending_alerts=[SimpleNamespace(summary="Timer done.")])
    args.update(changes)
    return brain._fingerprint(**args)


def test_the_fingerprint_is_stable_for_the_same_inputs():
    assert _print() == _print()


@pytest.mark.parametrize("change", [
    {"user_text": "how did I sleep last night"},
    {"channel": "text"},
    {"tier": "jonin"},
    {"model": "claude-sonnet-4-5-20250929"},
    {"prompt": "prompt with one more memory"},
    {"history": HISTORY[:-1] + [{"role": "user", "content": "how did you sleep"}]},
    {"pending_alerts": []},
])
def test_the_fingerprint_changes_with_any_input(change):
    assert _print(**change) != _print()


def test_the_clock_is_not_part_of_the_fingerprint(monkeypatch):
    """Stamped to the minute, and a speculation runs under a second early."""
    capture(monkeypatch, **SCENARIOS["text_hokage"])
    first, _ = _built(**SCENARIOS["text_hokage"])
    class _Later(_FixedClock):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 25, 21, 6)
    monkeypatch.setattr(brain, "datetime", _Later)
    second, _ = _built(**SCENARIOS["text_hokage"])
    assert first.fingerprint == second.fingerprint
    assert first.messages != second.messages
