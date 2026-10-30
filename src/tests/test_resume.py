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


# ── where a resume starts: his approved examples, Oct 30 2026 ──
#
# Each test is one example he approved before the code was written, numbered as
# in that list. They assert the exact sentence, because the tests these replaced
# asserted the mechanism and passed on "Anyway, but most granola is heavy".

SAID = "The high today is seventy, with a light breeze and"

GRANOLA_SAID = ("Granola's a mixed bag. It's got oats and nuts, which are solid, "
                "but most granola is heavy on")
GRANOLA_LEFT = "sugar and oil. It works as an occasional thing, not a staple."
GRANOLA_CUT_SENTENCE = ("It's got oats and nuts, which are solid, but most granola "
                        "is heavy on sugar and oil.")


def _resume(said, left="the rest.", gap=0.0):
    resume.keep(left, said=said, now=0.0)
    return resume.speech(now=gap)


def test_1_a_quick_go_on_restarts_the_cut_sentence_with_no_lead_in():
    assert _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=3.0) == \
        GRANOLA_CUT_SENTENCE + " It works as an occasional thing, not a staple."


def test_2_after_a_real_gap_she_leads_in_first():
    assert _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=40.0) == \
        "Anyway, it's got oats and nuts, which are solid, but most granola " \
        "is heavy on sugar and oil. It works as an occasional thing, not a staple."


def test_3_a_stop_to_nothing_resumes_the_same_way_as_a_quick_go_on():
    assert _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=2.0) == \
        GRANOLA_CUT_SENTENCE + " It works as an occasional thing, not a staple."


def test_4_a_cut_between_sentences_replays_nothing():
    said = GRANOLA_CUT_SENTENCE.join(["Granola's a mixed bag. ", ""])
    assert _resume(said, "It works as an occasional thing, not a staple.", gap=3.0) == \
        "It works as an occasional thing, not a staple."


def test_5_a_cut_in_the_first_sentence_restarts_the_reply():
    assert _resume("Granola's a mixed", "bag. It's got oats and nuts.", gap=3.0) == \
        "Granola's a mixed bag. It's got oats and nuts."


def test_6_a_cut_before_her_first_word_says_the_whole_reply():
    assert _resume("", left="Granola's a mixed bag.", gap=40.0) == "Granola's a mixed bag."


def test_7_a_second_cut_restarts_the_newer_sentence():
    said = GRANOLA_CUT_SENTENCE + " It works as an"
    assert _resume(said, "occasional thing, not a staple.", gap=3.0) == \
        "It works as an occasional thing, not a staple."


def test_8_two_resumes_after_gaps_never_open_the_same_way():
    first = _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=40.0)
    second = _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=40.0)
    assert first.startswith("Anyway, ") and second.startswith("So, ")


def test_11_a_very_long_sentence_backs_up_to_its_last_comma():
    said = ("Your readiness is seventy one, which is lower than your weekly average "
            "of seventy eight, mostly because your resting heart rate was up and "
            "you slept about")
    assert _resume(said, "six hours.", gap=3.0) == \
        "mostly because your resting heart rate was up and you slept about six hours."


# Examples 9 (too late) and 10 (a guest) are unchanged behavior, covered by
# test_it_goes_stale_after_the_window_and_stays_gone and
# test_a_guest_cannot_make_her_finish_what_may_be_his.

# ── the rules behind the examples, at their edges ──

def test_the_lead_in_starts_at_ten_seconds_exactly():
    assert not _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=9.9).startswith("Anyway")
    assert _resume(GRANOLA_SAID, GRANOLA_LEFT, gap=10.0).startswith("Anyway, ")


def test_twenty_words_of_a_sentence_still_restarts_it_whole():
    said = "one two three four five six seven eight nine ten, " \
           "eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    assert _resume(said, "end.", gap=3.0) == said + " end."


def test_a_long_sentence_never_backs_up_to_a_clause_that_opens_on_but():
    said = ("Oats are fine for most people, especially before a lift, but the sugar "
            "in most store granola adds up quickly over a")
    assert _resume(said, "week.", gap=3.0) == \
        "especially before a lift, but the sugar in most store granola adds up " \
        "quickly over a week."


def test_a_long_sentence_with_no_usable_comma_restarts_whole():
    said = ("Oats are fine for most people but the sugar in most store granola adds "
            "up quickly over a long")
    assert _resume(said, "week.", gap=3.0) == said + " week."


def test_a_lead_in_never_doubles_the_word_it_introduces():
    assert _resume("So the high is", "seventy.", gap=40.0) == "Anyway, so the high is seventy."


def test_a_lead_in_keeps_i_capitalized():
    assert _resume("I think the", "gym opens at six.", gap=40.0) == \
        "Anyway, I think the gym opens at six."


def test_a_resume_of_a_resume_does_not_stack_lead_ins():
    # She led in, got cut inside that same first sentence, and was asked again.
    said = "Anyway, it's got oats and nuts, which are"
    assert _resume(said, "solid.", gap=40.0) == \
        "So, it's got oats and nuts, which are solid."


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
    # "Step" was heard and is the start of the reply, so the sentence restarts;
    # resumed at once, so no lead in.
    assert out == ("Step count alone matters little. You lift four days a week. "
                   "Recovery matters more.")


# ── speaking it after a stop that came to nothing ──

@pytest.fixture
def speaker(monkeypatch):
    """Speaks what she is given on fakes. cut_at is the 1 based number of the
    request to interrupt, a quarter of the way in, or None to let all play."""
    def run(cut_at=None, amend_history=False):
        sent = []

        def play(synthesis):
            sent.append(synthesis.text)
            return cut_at == len(sent), 0.0

        monkeypatch.setattr(brain, "start_synthesis",
                            lambda text, *a, **k: SimpleNamespace(
                                text=text, heard_s=1.0, audio_s=4.0))
        monkeypatch.setattr(brain, "_play_with_barge_in", play)
        return sent, brain.speak_what_was_left(amend_history=amend_history)
    return run


def test_nothing_held_means_she_says_nothing(speaker):
    sent, spoke = speaker()
    assert spoke == "" and sent == []


def test_she_finishes_the_reply_from_a_clause_and_says_it_once(speaker):
    resume.keep("seventy and sunny. Bring sunglasses.", said=SAID)
    sent, spoke = speaker()
    assert spoke == "The high today is seventy, with a light breeze and seventy and sunny. Bring sunglasses.", "what she said"
    assert sent[0] == "The high today is seventy, with a light breeze and seventy and sunny."
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
    # The resumed first sentence is sixteen words; a quarter of its audio played,
    # "The high today is". The newer cut restarts that same sentence again.
    assert resume.speech() == ("The high today is seventy, with a light breeze and "
                               "It is sunny today. Bring sunglasses.")


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
             "resume": resume,
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


# ── her history after a resume ──

def test_extend_message_adds_to_the_row_and_drops_the_mark(db):
    mark = brain.CUT_OFF_MARK
    row = db.save_message("assistant", f"The high is seventy, {mark}")
    db.extend_message(row, "Anyway, and sunny.", drop_suffix=mark)
    assert db.get_recent_messages()[-1]["content"] == "The high is seventy, Anyway, and sunny."


def test_extend_message_leaves_a_row_without_the_mark_whole(db):
    row = db.save_message("assistant", "Fine.")
    db.extend_message(row, "More.", drop_suffix=brain.CUT_OFF_MARK)
    assert db.get_recent_messages()[-1]["content"] == "Fine. More."


def test_extend_message_on_a_missing_row_does_nothing(db):
    db.save_message("assistant", "Fine.")
    db.extend_message(9999, "More.")
    assert [m["content"] for m in db.get_recent_messages()] == ["Fine."]


def test_the_cut_reply_remembers_which_row_it_was(turn):
    turn(_FakeStream(THREE), cut_first=True)
    assert resume.message_id() == 7, "the row save_message returned for the cut reply"


def test_peeking_at_the_row_does_not_use_it_up():
    resume.keep("the rest.", said=SAID, message_id=3, now=0.0)
    assert resume.message_id(now=1.0) == 3
    assert resume.message_id(now=1.0) == 3
    assert resume.speech(now=1.0) is not None


def test_a_stale_or_empty_hold_has_no_row():
    assert resume.message_id() is None
    resume.keep("the rest.", said=SAID, message_id=3, now=0.0)
    assert resume.message_id(now=resume.RESUME_WINDOW_S + 1) is None


@pytest.fixture
def amended(monkeypatch):
    calls = []
    monkeypatch.setattr(brain, "extend_message",
                        lambda message_id, text, drop_suffix="", drop_words=0: calls.append(
                            (message_id, text, drop_suffix, drop_words)))
    return calls


def test_finishing_after_a_stop_to_nothing_extends_the_cut_row(speaker, amended):
    resume.keep("seventy and sunny.", said=SAID, message_id=5)
    _, said = speaker(amend_history=True)
    # Ten words of SAID were restarted, so ten come off the row before the append.
    assert amended == [(5, "The high today is seventy, with a light breeze and seventy and sunny.",
                        brain.CUT_OFF_MARK, 10)]
    assert said == "The high today is seventy, with a light breeze and seventy and sunny."


def test_cut_again_the_row_ends_in_the_mark_and_stays_amendable(speaker, amended):
    resume.keep("It is sunny today. Bring sunglasses.", said=SAID, message_id=5)
    _, said = speaker(cut_at=1, amend_history=True)
    assert said.endswith(brain.CUT_OFF_MARK)
    assert amended[0][:2] == (5, said)
    assert resume.message_id() == 5, "a third try amends the same row"


def test_no_saved_row_means_nothing_to_amend(speaker, amended):
    resume.keep("seventy and sunny.", said=SAID)
    speaker(amend_history=True)
    assert amended == []


def test_a_failed_amend_does_not_end_the_voice_loop(speaker, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(brain, "extend_message", broken)
    resume.keep("seventy and sunny.", said=SAID, message_id=5)
    _, said = speaker(amend_history=True)
    assert said, "she still spoke, and told the caller so"


def _next_turn_sees(db):
    return brain._trim_history(db.get_recent_messages())[-1]["content"]


def test_after_a_full_resume_the_next_turn_sees_a_reply_he_heard(speaker, db):
    row = db.save_message("assistant", f"The high today is {brain.CUT_OFF_MARK}")
    resume.keep("seventy and sunny.", said="The high today is", message_id=row)
    assert _next_turn_sees(db).endswith(brain.CUT_OFF_MARK)
    speaker(amend_history=True)
    after = _next_turn_sees(db)
    assert brain.CUT_OFF_MARK not in after, "he heard the end, so it no longer says he did not"
    assert "seventy and sunny" in after


def test_after_a_second_cut_the_next_turn_still_sees_where(speaker, db):
    row = db.save_message("assistant", f"The high today is {brain.CUT_OFF_MARK}")
    resume.keep("seventy and sunny. Bring sunglasses.", said="The high today is",
                message_id=row)
    speaker(cut_at=1, amend_history=True)
    assert _next_turn_sees(db).endswith(brain.CUT_OFF_MARK)


def test_a_spoken_go_on_does_not_amend_because_it_saves_its_own_rows(speaker, amended):
    resume.keep("seventy and sunny.", said=SAID, message_id=5)
    speaker()
    assert amended == []


# ── what history says afterwards: his approved examples 12 and 13 ──
#
# These read the final row, not the call that wrote it. The tests above that
# checked the call passed while the row read "The high today is The high today
# is seventy and sunny", and while a go on's cut reply was finished in the row
# of the reply before it.

MARK = brain.CUT_OFF_MARK
GRANOLA_HEAD = "Granola's a mixed bag."
GRANOLA_DONE = (f"{GRANOLA_HEAD} {GRANOLA_CUT_SENTENCE} "
                "It works as an occasional thing, not a staple.")


def _row(db, row_id):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    content = conn.execute("SELECT content FROM conversation_history WHERE id = ?",
                           (row_id,)).fetchone()[0]
    conn.close()
    return content


def _room(monkeypatch, cut_at=None):
    """Fake synthesis and playback; cut_at interrupts that numbered request a
    quarter of the way in. Returns what was sent to the voice."""
    sent = []

    def play(synthesis):
        sent.append(synthesis.text)
        return cut_at == len(sent), 0.0

    monkeypatch.setattr(brain, "start_synthesis",
                        lambda text, *a, **k: SimpleNamespace(text=text, heard_s=1.0,
                                                              audio_s=4.0))
    monkeypatch.setattr(brain, "_play_with_barge_in", play)
    return sent


def _go_on(db):
    import ast
    scope = {"speak_what_was_left": brain.speak_what_was_left,
             "save_message": db.save_message, "resume": resume,
             "TurnResult": lambda text, dismissed: text,
             "print": lambda *a, **k: None}
    exec(compile(ast.Module([_voice_main_function("_run_go_on")], []),
                 "voice_main.py", "exec"), scope)
    return scope["_run_go_on"]("Go on.")


def _granola_cut(db, said=GRANOLA_SAID, kept_at=None):
    row = db.save_message("assistant", f"{said} {MARK}")
    resume.keep(GRANOLA_LEFT, said=said, message_id=row,
                **({} if kept_at is None else {"now": kept_at}))
    return row


def test_13_finishing_replaces_the_partial_sentence_instead_of_repeating_it(
        monkeypatch, db):
    row = _granola_cut(db)
    _room(monkeypatch)
    brain.speak_what_was_left(amend_history=True)
    assert _row(db, row) == GRANOLA_DONE


def test_13_the_lead_in_is_spoken_but_never_written(monkeypatch, db):
    import time
    row = _granola_cut(db, kept_at=time.monotonic() - 40)
    sent = _room(monkeypatch)
    said = brain.speak_what_was_left(amend_history=True)
    assert sent[0].startswith("Anyway, it's got oats"), "he hears the lead in"
    assert _row(db, row) == GRANOLA_DONE, "history keeps only what she told him"
    assert not said.startswith("Anyway")


def test_13_cut_again_and_finished_the_row_still_reads_once(monkeypatch, db):
    row = _granola_cut(db)
    _room(monkeypatch, cut_at=1)
    brain.speak_what_was_left(amend_history=True)
    # A quarter of a sixteen word sentence: "It's got oats and".
    assert _row(db, row) == f"{GRANOLA_HEAD} It's got oats and {MARK}"
    _room(monkeypatch)
    brain.speak_what_was_left(amend_history=True)
    assert _row(db, row) == GRANOLA_DONE


def test_12_a_go_on_reply_cut_again_is_finished_in_its_own_row(monkeypatch, db):
    first = _granola_cut(db, said="Granola's a mixed bag. It's got oats and nuts, "
                                  "which are solid,")
    resume.keep("but most granola is heavy on sugar and oil. It works as an "
                "occasional thing, not a staple.",
                said="Granola's a mixed bag. It's got oats and nuts, which are solid,",
                message_id=first)
    _room(monkeypatch, cut_at=1)
    _go_on(db)
    go_on_row = first + 2       # his "Go on." and then her reply to it
    assert _row(db, go_on_row) == f"It's got oats and {MARK}"
    _room(monkeypatch)
    brain.speak_what_was_left(amend_history=True)   # a cough, and she finishes
    assert _row(db, first) == ("Granola's a mixed bag. It's got oats and nuts, "
                               f"which are solid, {MARK}"), "the earlier reply is left alone"
    assert _row(db, go_on_row) == (f"{GRANOLA_CUT_SENTENCE} "
                                   "It works as an occasional thing, not a staple.")


# ── the test paragraph: his approved examples P1 to P5 ──
#
# A fixed reply for the room test, so every run starts from the same words.
# Rung 4 of docs/TESTING.md needs a known input; Claude's replies vary.

def _test_paragraph(db, user_text="read the test paragraph"):
    import ast
    scope = {"speak_what_was_left": brain.speak_what_was_left,
             "save_message": db.save_message, "resume": resume,
             "TEST_PARAGRAPH": li.TEST_PARAGRAPH,
             "TurnResult": lambda text, dismissed: text,
             "print": lambda *a, **k: None}
    exec(compile(ast.Module([_voice_main_function("_run_test_paragraph")], []),
                 "voice_main.py", "exec"), scope)
    return scope["_run_test_paragraph"](user_text)


def test_p1_the_paragraph_is_the_approved_text():
    assert li.TEST_PARAGRAPH == (
        "Granola's a mixed bag. It's got oats and nuts, which are solid, but most "
        "granola is heavy on sugar and oil. It works as an occasional thing, not a "
        "staple. If you want carbs before a lift, a banana or plain oats do better, "
        "and they cost less, they keep you full longer, and they don't spike your "
        "sugar the way most store granola does. For snacking, the calories add up fast.")


def test_p1_she_reads_it_through_the_reply_path_first_sentence_alone(monkeypatch, db):
    sent = _room(monkeypatch)
    assert _test_paragraph(db) == li.TEST_PARAGRAPH
    assert sent[0] == "Granola's a mixed bag.", "the first sentence at once, as any reply"
    assert len(sent) == 2, "and the rest as one piece"


@pytest.mark.parametrize("text", ["read the test paragraph", "Read the test paragraph.",
                                  "read me the test paragraph", "test paragraph"])
def test_p2_only_the_whole_phrase_matches(text, sure):
    match = li.classify(text)
    assert match is not None and match.name == "test_paragraph"


@pytest.mark.parametrize("text", ["read the test paragraph about oats",
                                  "what's the test paragraph", "read a paragraph"])
def test_p2_anything_more_goes_to_claude(text, sure):
    match = li.classify(text)
    assert match is None or match.name != "test_paragraph"


def test_p3_a_guest_cannot_trigger_it():
    import ast
    scope = {"registry": SimpleNamespace(get=lambda name: 1 / 0),
             "permits": lambda spec, tier: True}
    exec(compile(ast.Module([_voice_main_function("_local_allowed")], []),
                 "voice_main.py", "exec"), scope)
    paragraph = SimpleNamespace(name="test_paragraph")
    assert scope["_local_allowed"](paragraph, None) is True
    assert scope["_local_allowed"](paragraph, "genin") is False


def test_p4_both_sides_are_saved_like_any_reply(monkeypatch, db):
    _room(monkeypatch)
    _test_paragraph(db)
    rows = db.get_recent_messages()[-2:]
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "read the test paragraph"), ("assistant", li.TEST_PARAGRAPH)]


def test_p4_cut_off_it_is_held_and_a_go_on_restarts_the_sentence(monkeypatch, db):
    _room(monkeypatch, cut_at=2)
    _test_paragraph(db)
    reply_id = db.latest_message_id()
    assert _row(db, reply_id).endswith(MARK)
    assert resume.message_id() == reply_id, "a stop to nothing finishes this row"
    # The rest is 65 words and a quarter were heard: 16, exactly "It's got oats
    # and nuts, which are solid, but most granola is heavy on sugar and oil." A
    # cut between sentences replays nothing (example 4): the next one starts.
    assert resume.speech().startswith("It works as an occasional thing, not a staple.")


def test_p4_run_turn_hands_it_to_its_own_runner():
    import ast
    body = ast.unparse(_voice_main_function("run_turn"))
    assert "_run_test_paragraph(text)" in body
