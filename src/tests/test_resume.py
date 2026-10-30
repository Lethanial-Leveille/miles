"""What she had left to say when he cut her off is held for a go on.

Nothing here reaches Claude, ElevenLabs, the microphone or the speaker."""

import asyncio
from types import SimpleNamespace

import pytest

import brain
import resume
from tools import registry


@pytest.fixture(autouse=True)
def nothing_held():
    resume.clear()
    resume._lead_next = 0
    yield
    resume.clear()


# ── the store ──

def test_what_was_kept_is_taken_once():
    resume.keep("and then the rest.", now=100.0)
    assert resume.take(now=101.0) == "and then the rest."
    assert resume.take(now=102.0) is None, "one go on must never say it twice"


def test_nothing_kept_means_nothing_to_take():
    assert resume.take(now=0.0) is None


def test_a_blank_remainder_is_no_remainder():
    resume.keep("   ", now=0.0)
    assert resume.take(now=1.0) is None
    resume.keep("", now=0.0)
    assert resume.take(now=1.0) is None


def test_it_goes_stale_after_the_window_and_stays_gone():
    resume.keep("the rest.", now=0.0)
    assert resume.take(now=resume.RESUME_WINDOW_S + 1) is None
    assert resume.take(now=resume.RESUME_WINDOW_S + 2) is None


def test_the_window_includes_its_last_second():
    resume.keep("the rest.", now=0.0)
    assert resume.take(now=resume.RESUME_WINDOW_S) == "the rest."


def test_a_newer_cut_replaces_the_older_one():
    resume.keep("old.", now=0.0)
    resume.keep("new.", now=5.0)
    assert resume.take(now=6.0) == "new."


def test_keeping_nothing_drops_what_was_there():
    resume.keep("old.", now=0.0)
    resume.keep("", now=5.0)
    assert resume.take(now=6.0) is None


def test_clear_lets_go_of_it():
    resume.keep("old.", now=0.0)
    resume.clear()
    assert resume.take(now=1.0) is None


# ── where a resume starts ──

SAID = "The high today is seventy, with a light breeze and"


def _resume(said, left="the rest.", gap=0.0):
    resume.keep(left, said=said, now=0.0)
    return resume.speech(now=gap)


def test_it_backs_up_to_the_nearest_clause_word_and_leads_in():
    # Four words back covers "a light breeze and"; "and" is the nearest boundary.
    assert _resume(SAID, "seventy and sunny.") == "Anyway, and seventy and sunny."


def test_it_backs_up_to_the_word_after_a_comma():
    assert _resume("The high today is seventy, with a light") == \
        "Anyway, with a light the rest."


def test_a_cut_right_after_a_comma_needs_no_backing_up():
    assert _resume("The high today is seventy,") == "Anyway, the rest."


def test_a_cut_after_a_full_stop_starts_the_next_sentence_clean():
    assert _resume("It is sunny. Bring sunglasses.") == "Anyway, the rest."


def test_with_no_boundary_in_range_it_backs_up_the_whole_window():
    assert _resume("one two three four five six", gap=0.0) == \
        "Anyway, three four five six the rest."


def test_a_longer_gap_backs_up_further():
    said = "Step count alone matters little for gaining muscle on a lift"
    near = _resume(said, gap=5.0)
    far = _resume(said, gap=45.0)
    assert len(far.split()) - len(near.split()) == 4


def test_it_never_backs_up_past_what_it_kept():
    said = "a b c d e f g h i j k l m n o p"
    out = _resume(said, left="END", gap=100.0)
    assert out == "Anyway, " + " ".join(said.split()[-resume.SAID_KEPT_WORDS:]) + " END"


def test_the_lead_in_rotates_and_never_doubles_the_first_word():
    leads = [_resume("one two three four five six").split()[0] for _ in range(3)]
    assert len(set(leads)) == 3, "three resumes in a row should not open the same way"
    resume._lead_next = 1       # the next lead is "So,"
    assert _resume("The high is seventy, so") == "As I was saying, so the rest."


def test_nothing_said_before_the_cut_means_no_lead_in():
    assert _resume("", left="Step count matters.") == "Step count matters."


def test_it_is_said_once_and_goes_stale_like_take():
    resume.keep("the rest.", said=SAID, now=0.0)
    assert resume.speech(now=1.0) is not None
    assert resume.speech(now=2.0) is None
    resume.keep("the rest.", said=SAID, now=0.0)
    assert resume.speech(now=resume.RESUME_WINDOW_S + 1) is None


# ── a whole turn ──

class _FakeStream:
    """Stands in for claude.messages.stream: the text it was given, then a final
    message holding whatever tool calls it was given."""

    def __init__(self, texts, tool_uses=()):
        self._texts = texts
        self._final = SimpleNamespace(
            content=list(tool_uses),
            usage=SimpleNamespace(cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for text in self._texts:
            yield SimpleNamespace(type="content_block_delta",
                                  delta=SimpleNamespace(type="text_delta", text=text))
        for block in self._final.content:
            yield SimpleNamespace(type="content_block_start", content_block=block)

    async def get_final_message(self):
        return self._final


@pytest.fixture
def turn(monkeypatch):
    """Runs one whole turn on fakes. cut_first says whether the room speaker is
    interrupted during the first thing she says, a quarter of the way in."""
    monkeypatch.setattr(brain, "speak", lambda text, *a, **k: None)
    monkeypatch.setattr(brain.phrasebank, "play", lambda key, **k: key)
    monkeypatch.setattr(brain, "save_message", lambda *a, **k: 7)
    monkeypatch.setattr(brain, "last_reply_at", lambda: None)
    monkeypatch.setattr(brain, "delete_message", lambda message_id: None)
    monkeypatch.setattr(brain, "get_recent_messages",
                        lambda limit=20: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr(brain, "get_seed_memories", lambda: [])
    monkeypatch.setattr(brain, "get_episodic_memories", lambda: [])
    monkeypatch.setattr(brain, "memory_manifest", lambda: [])
    monkeypatch.setattr(brain, "build_enhanced_prompt", lambda *a, **k: "prompt")
    monkeypatch.setattr(brain, "search_memories", lambda *a, **k: [])
    monkeypatch.setattr(brain, "log_tool_call", lambda *a, **k: None)
    monkeypatch.setattr(brain.alerts, "take_for_fold", lambda: [])
    monkeypatch.setattr(registry, "call", lambda name, args: "ran")
    monkeypatch.setattr(brain.memory_pass, "notice_later", lambda *a, **k: None)

    def run(*streams, channel="voice", device="pi", cut_first=False):
        spoken = []

        def play(synthesis):
            spoken.append(synthesis.text)
            return cut_first and len(spoken) == 1, 0.0

        monkeypatch.setattr(brain, "start_synthesis",
                            lambda text, *a, **k: SimpleNamespace(
                                text=text, heard_s=1.0, audio_s=4.0))
        monkeypatch.setattr(brain, "_play_with_barge_in", play)
        remaining = list(streams)
        monkeypatch.setattr(brain, "claude", SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: remaining.pop(0))))
        return asyncio.run(brain.ask_nova_async("hi", device=device, channel=channel))
    return run


THREE = ["Step count alone matters little. You lift four days a week. ",
         "Recovery matters more."]


def test_a_cut_reply_keeps_the_rest_of_the_sentence_and_everything_after(turn):
    turn(_FakeStream(THREE), cut_first=True)
    # Five words, a quarter of the audio: "Step" was heard.
    assert resume.take() == ("count alone matters little. You lift four days a week. "
                             "Recovery matters more.")


def test_a_reply_that_played_through_keeps_nothing(turn):
    turn(_FakeStream(THREE))
    assert resume.take() is None


def test_a_reply_that_played_through_ends_the_chance_to_go_on(turn):
    resume.keep("the old reply's ending.")
    turn(_FakeStream(["Fine."]))
    assert resume.take() is None


def test_a_typed_turn_leaves_it_alone(turn):
    resume.keep("the old reply's ending.")
    turn(_FakeStream(["Fine."]), channel="text")
    assert resume.take() == "the old reply's ending."


def test_a_turn_from_the_phone_leaves_it_alone(turn):
    resume.keep("the old reply's ending.")
    turn(_FakeStream(["Fine."]), device="phone")
    assert resume.take() == "the old reply's ending."


def test_room_talk_she_stays_out_of_leaves_it_alone(turn):
    """It may be what cut her, and it is not an answer to what she was saying."""
    resume.keep("the old reply's ending.")
    ignore = SimpleNamespace(type="tool_use", name="ignore", id="toolu_1", input={})
    turn(_FakeStream([], [ignore]))
    assert resume.take() == "the old reply's ending."


def test_a_cut_during_the_lead_in_keeps_the_answer_he_never_heard(turn):
    """On a tool turn the follow up is collected, never played, once he has cut in."""
    weather = SimpleNamespace(type="tool_use", name="get_weather", id="toolu_1",
                              input={"location": "Gainesville"})
    turn(_FakeStream(["Let me check."], [weather]),
         _FakeStream(["It is sunny. Bring sunglasses."]), cut_first=True)
    assert resume.take() == "me check. It is sunny. Bring sunglasses."


def test_a_cut_reply_keeps_the_word_she_did_say(turn):
    turn(_FakeStream(THREE), cut_first=True)
    out = resume.speech()
    # "Step" was heard and is the start of the reply, so it is replayed.
    assert out == ("Anyway, Step count alone matters little. You lift four days a week. "
                   "Recovery matters more.")


# ── speaking it after a stop that came to nothing ──

@pytest.fixture
def speaker(monkeypatch):
    """Speaks what she is given on fakes. cut_at is the 1 based number of the
    request to interrupt, a quarter of the way in, or None to let all play."""
    def run(cut_at=None):
        sent = []

        def play(synthesis):
            sent.append(synthesis.text)
            return cut_at == len(sent), 0.0

        monkeypatch.setattr(brain, "start_synthesis",
                            lambda text, *a, **k: SimpleNamespace(
                                text=text, heard_s=1.0, audio_s=4.0))
        monkeypatch.setattr(brain, "_play_with_barge_in", play)
        return sent, brain.speak_what_was_left()
    return run


def test_nothing_held_means_she_says_nothing(speaker):
    sent, spoke = speaker()
    assert spoke == "" and sent == []


def test_she_finishes_the_reply_from_a_clause_and_says_it_once(speaker):
    resume.keep("seventy and sunny. Bring sunglasses.", said=SAID)
    sent, spoke = speaker()
    assert spoke == "Anyway, and seventy and sunny. Bring sunglasses.", "what she said"
    assert sent[0] == "Anyway, and seventy and sunny."
    assert "Bring sunglasses." in " ".join(sent)
    assert resume.take() is None, "a second stop to nothing must not repeat it"
    assert speaker()[1] == ""


def test_the_first_sentence_goes_out_on_its_own(speaker):
    resume.keep("It is sunny. Bring sunglasses. Enjoy.", said="Today,")
    sent, _ = speaker()
    assert len(sent) == 2, "the first sentence at once, the rest as one piece"


def test_if_he_cuts_in_again_what_is_still_unsaid_is_held_again(speaker):
    resume.keep("It is sunny today. Bring sunglasses.", said=SAID)
    speaker(cut_at=1)
    # "Anyway, and It is sunny today." is six words; a quarter of its audio played,
    # which is "Anyway, and". She backs up to the comma, rotating to the next lead in.
    assert resume.speech() == "So, and It is sunny today. Bring sunglasses."


def test_a_resume_that_plays_through_leaves_nothing_held(speaker):
    resume.keep("It is sunny.", said=SAID)
    speaker()
    assert resume.take() is None


def test_both_empty_stops_try_to_resume_before_listening_again():
    """The recording after a stop was noise, or too short to be speech."""
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "voice_main.py").read_text()
    for branch in ("if is_noise_transcript(followup_text):",
                   "if followup_result == audio.NO_AUDIO:"):
        body = source[source.index(branch):].split("break", 1)[0]
        interrupted = body[body.index("if interrupted_turn:"):]
        assert interrupted.index("resume_after_empty_stop()") < interrupted.index("continue"), branch


def test_a_stop_to_nothing_never_calls_claude():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "voice_main.py").read_text()
    helper = source[source.index("def resume_after_empty_stop"):source.index("def run_turn")]
    assert "ask_nova" not in helper and "run_turn" not in helper


# ── "go on", said by him ──

import local_intent as li


@pytest.fixture
def sure(monkeypatch):
    """The semantic half stubbed, as test_local_intent does: what is under test
    is the gate, which is exact."""
    monkeypatch.setattr(li, "_scores", lambda text: {n: 1.0 for n in li.EXAMPLES})


@pytest.mark.parametrize("text", ["go on", "Go on.", "keep going", "continue",
                                  "please go on", "Go on, please"])
def test_go_on_alone_resumes_when_something_is_held(text, sure):
    resume.keep("the rest.", said=SAID)
    match = li.classify(text)
    assert match is not None and match.name == "go_on", text


def test_classifying_does_not_use_up_what_is_held(sure):
    resume.keep("the rest.", said=SAID)
    li.classify("go on")
    li.classify("go on")
    assert resume.speech() is not None, "only speaking it may consume it"


@pytest.mark.parametrize("text", ["go on", "keep going", "continue"])
def test_go_on_with_nothing_held_goes_to_claude(text, sure):
    assert li.classify(text) is None


def test_go_on_after_the_window_goes_to_claude(sure, monkeypatch):
    resume.keep("the rest.", said=SAID, now=0.0)
    monkeypatch.setattr(resume.time, "monotonic", lambda: resume.RESUME_WINDOW_S + 1)
    assert li.classify("go on") is None
    monkeypatch.setattr(resume.time, "monotonic", lambda: resume.RESUME_WINDOW_S)
    assert li.classify("go on") is not None, "the last second of the window still counts"


@pytest.mark.parametrize("text", [
    "go on and set a timer for ten minutes",
    "go on what about friday",
    "wait go on",
    "no go on to the next one",
    "continue with the second point",
    "keep going straight",
])
def test_anything_more_than_go_on_is_a_new_turn(text, sure):
    """He said more than the continuation, so she answers him instead."""
    resume.keep("the rest.", said=SAID)
    match = li.classify(text)
    assert match is None or match.name != "go_on", text


def test_the_semantic_half_must_agree_too(monkeypatch):
    monkeypatch.setattr(li, "_scores", lambda text: {n: 0.0 for n in li.EXAMPLES})
    resume.keep("the rest.", said=SAID)
    assert li.classify("go on") is None


def _voice_main_function(name):
    import ast
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "voice_main.py"
    tree = ast.parse(src.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_a_guest_cannot_make_her_finish_what_may_be_his():
    import ast
    fn = _voice_main_function("_local_allowed")
    scope = {"registry": SimpleNamespace(get=lambda name: 1 / 0),
             "permits": lambda spec, tier: True}
    exec(compile(ast.Module([fn], []), "voice_main.py", "exec"), scope)
    allowed = scope["_local_allowed"]
    go_on = SimpleNamespace(name="go_on")
    assert allowed(go_on, None) is True
    assert allowed(go_on, "genin") is False


def test_go_on_saves_both_sides_and_falls_back_when_nothing_played():
    import ast
    fn = _voice_main_function("_run_go_on")
    saved = []
    said = ["Anyway, the rest."]
    scope = {"speak_what_was_left": lambda: said[0],
             "save_message": lambda role, text: saved.append((role, text)),
             "TurnResult": lambda text, dismissed: (text, dismissed),
             "print": lambda *a, **k: None}
    exec(compile(ast.Module([fn], []), "voice_main.py", "exec"), scope)
    run = scope["_run_go_on"]
    assert run("go on") == ("Anyway, the rest.", False)
    assert saved == [("user", "go on"), ("assistant", "Anyway, the rest.")]
    saved.clear()
    said[0] = ""
    assert run("go on") is None and saved == [], "Claude answers, and nothing was invented"


def test_run_turn_hands_go_on_to_the_resume_not_the_fixed_sentence_path():
    import ast
    body = ast.unparse(_voice_main_function("run_turn"))
    assert "_run_go_on(text)" in body
    assert body.index("match.name != 'go_on'") < body.index("_run_local(match, text)")
