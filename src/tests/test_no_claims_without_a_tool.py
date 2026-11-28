"""A reply that opens by claiming a change is replaced when no mutating tool ran.
Offline, fake model, fake names only."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

import brain
from task_intent import Intent
from test_task_intent import Stream, block, turn  # noqa: F401  (turn is a fixture)
from voice_action_safety import ClaimGate, claim_state, opens_with_claim

CLAIMS = ["Stopped the daily task: Stretching.", "Added milk to the list.", "Archived it.", "Skipped it for today.",
          "Saved it.", "Removed that.", "Marked it done.", "Set a timer for ten minutes.", "Done.",
          "Done, all set.", "Done", "stopped it.", "  \"Archived it.\"", "[calmly] Added the task."]
HONEST = ["Yes, I added milk earlier.", "I didn't add it twice.", "Set reminders are listed below.",
          "Done with the dishes? Good.", "Addedum noted.", "The task was stopped earlier.", "Okay."]


@pytest.mark.parametrize("text", CLAIMS)
def test_every_claim_opening_is_recognised(text):
    assert opens_with_claim(text)


@pytest.mark.parametrize("text", HONEST)
def test_other_openings_are_not_claims(text):
    assert not opens_with_claim(text)


def test_the_gate_decides_at_the_first_word_and_never_leaks_a_claim():
    gate = ClaimGate()
    assert gate.push("Sto") == "" and gate.push("pped it. ") == "" and gate.push("More") == ""
    assert gate.finish() == ""
    gate = ClaimGate()
    assert gate.push("Sur") == "" and gate.push("e, ok") == "Sure, ok" and gate.push(" fine") == " fine"
    assert ClaimGate().push("A very long first word indeed") != ""
    assert claim_state("Supercalifragilistic") == "clear"
    short = ClaimGate()
    assert short.push("Okay") == "" and short.finish() == "Okay"


@pytest.mark.parametrize("channel, device", [("voice", "pi"), ("text", "app")])
@pytest.mark.parametrize("text", CLAIMS)
def test_an_unflagged_claim_without_a_tool_becomes_nothing_was_changed(turn, db, text, channel, device):
    result = asyncio.run(turn.run("hello there", [], draft=text, channel=channel, device=device))
    assert result.text == "Nothing was changed."
    shown = "".join(e[1] for e in turn.events if e[0] == "delta")
    assert "Nothing was changed." in shown and not any(w in shown for w in ("Stopped", "Added", "Done"))
    assert not any(part.lower().startswith(("stopped", "added", "archived", "skipped", "saved", "removed",
                                           "marked", "set", "done", "[calmly]")) for part in turn.audio)


@pytest.mark.parametrize("text", HONEST)
def test_honest_replies_are_untouched(turn, db, text):
    assert asyncio.run(turn.run("hello there", [], draft=text)).text == text


def test_a_flagged_no_change_turn_with_a_claim_is_replaced(turn, db):
    result = asyncio.run(turn.run("cancel that actually", [], draft="Stopped the daily task: Stretching.",
                                  intent=Intent("no_change")))
    assert result.text == "Nothing was changed."


def test_a_correct_answer_to_a_question_about_a_past_change_is_untouched(turn, db):
    result = asyncio.run(turn.run("did you add milk?", [], draft="Yes, I added milk earlier."))
    assert result.text == "Yes, I added milk earlier."


def test_a_reply_after_a_real_tool_call_is_the_code_written_one(turn, db):
    result = asyncio.run(turn.run("add water plants", [block("task_add", {"task": "water plants"})],
                                  draft="Done."))
    assert result.text == "Added the task: Water plants." and db.active_tasks_matching("water plants")


def test_a_read_tool_does_not_excuse_a_claim(turn, db):
    result = asyncio.run(turn.run("hello there", [block("tasks_today")], draft="Let me check.",
                                  more=[Stream([], "Stopped the daily task: Stretching.")]))
    assert result.text.endswith("Nothing was changed.") and "Stopped" not in result.text
    assert not any(part.startswith("Stopped") for part in turn.audio)


def test_a_read_tool_then_an_honest_answer_is_untouched(turn, db):
    result = asyncio.run(turn.run("hello there", [block("tasks_today")], draft="Let me check.",
                                  more=[Stream([], "You have nothing due.")]))
    assert result.text.endswith("You have nothing due.")


# ── The voice path: sentences are spoken as they arrive ──

class Chunks(Stream):
    """A stream that yields its text in pieces and lets the speaker run between them."""

    def __init__(self, pieces, marks):
        super().__init__([], "")
        self.pieces, self.marks = pieces, marks

    async def __aiter__(self):
        for piece in self.pieces:
            yield SimpleNamespace(type="content_block_delta",
                                  delta=SimpleNamespace(type="text_delta", text=piece))
            for _ in range(5):
                await asyncio.sleep(0)
            self.marks()


@pytest.fixture
def streaming_speech(turn, monkeypatch):
    monkeypatch.setattr(brain.registry, "api_schemas", lambda: [])
    return turn


def test_the_voice_path_speaks_an_ordinary_first_sentence_as_soon_as_it_is_whole(streaming_speech, db):
    turn = streaming_speech
    spoken_after = []
    turn.rounds.clear()
    stream = Chunks(["Sure, I can ", "help with that. ", "Anything else?"],
                    lambda: spoken_after.append((list(turn.audio), "".join(e[1] for e in turn.events if e[0] == "delta"))))

    async def run():
        async def request(client, text, pending_title=None, followup=None):
            raise AssertionError("an ordinary turn costs no intent call")
        brain.task_intent.request_intent = request
        turn.rounds.append(stream)
        return await brain.ask_nova_async("hello there", tier="hokage", channel="voice", device="pi",
                                          on_text=lambda *e: turn.events.append(e))

    result = asyncio.run(run())
    assert result.text == "Sure, I can help with that. Anything else?"
    # The first word was complete in the first piece, so the screen had it at once.
    assert spoken_after[0][1] == "Sure, I can "
    # The first sentence reached the speaker with the second piece, not at the end.
    assert spoken_after[1][0][:1] == ["Sure, I can help with that."]


def test_the_voice_path_speaks_nothing_of_a_claim_and_says_nothing_was_changed(streaming_speech, db):
    turn = streaming_speech
    heard_during = []
    stream = Chunks(["Stop", "ped the daily task: Stretching. ", "It is gone."], lambda: heard_during.append(list(turn.audio)))

    async def run():
        turn.rounds.append(stream)
        return await brain.ask_nova_async("hello there", tier="hokage", channel="voice", device="pi",
                                          on_text=lambda *e: turn.events.append(e))

    result = asyncio.run(run())
    assert result.text == "Nothing was changed."
    assert all(audio == [] for audio in heard_during)
    assert turn.audio == ["Nothing was changed."]
    assert not any("Stopped" in e[1] for e in turn.events if e[0] == "delta")


# ── The three routes ──

@pytest.mark.parametrize("route", ["/chat", "/chat/stream"])
def test_the_app_routes_never_show_or_keep_an_unsupported_claim(turn, db, monkeypatch, route):
    from fastapi.testclient import TestClient
    import server

    monkeypatch.setattr(brain, "_claude", lambda: SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: turn.rounds.pop(0))))
    monkeypatch.setattr(brain, "effective_tier", lambda: "hokage")
    monkeypatch.setitem(server.app.dependency_overrides, server.get_current_user, lambda: "fake-actor")
    turn.rounds.append(Stream([], "Stopped the daily task: Stretching."))
    body = {"message": "hello there", "channel": "text",
            "conversation_id": str(uuid4()), "request_id": str(uuid4())}
    client = TestClient(server.app)
    if route == "/chat":
        assert client.post(route, json=body).json()["response"] == "Nothing was changed."
        return
    with client.stream("POST", route, json=body) as streamed:
        frames = [json.loads(line[6:]) for line in streamed.iter_lines() if line.startswith("data: ")]
    shown = [frame["text"] for frame in frames if "text" in frame]
    assert shown and all("Stopped" not in text for text in shown) and shown[-1] == "Nothing was changed."
