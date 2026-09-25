"""The word error rate itself, because a bug here corrupts silently.

label_speakers.py already makes this argument for speaker labels: a wrong label
does not crash, it just quietly reports that the encoder is better or worse than
it is. The same is true one level up. If `wer` or `normalize` is wrong, every
STT comparison built on it returns a confident number that means nothing, and
nothing about the output would look off.

So the scoring is pinned against cases worked out by hand.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts", "label_transcripts.py")

_spec = importlib.util.spec_from_file_location("label_transcripts", _PATH)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)


# ── normalization ──

@pytest.mark.parametrize("raw,expected", [
    ("What's the weather?", "what's the weather"),
    ("SET A TIMER", "set a timer"),
    ("  spaced   out  ", "spaced out"),
    # Punctuation is not an error worth counting and would swamp the ones that
    # are, so it is dropped before scoring rather than penalized.
    ("Hello, world.", "hello world"),
    ("Hello world", "hello world"),
])
def test_normalize(raw, expected):
    assert lt.normalize(raw) == expected


@pytest.mark.parametrize("raw", ["[BLANK_AUDIO]", "[silence]", "(door closes)",
                                 "(beeping)", "", None])
def test_non_speech_normalizes_to_nothing(raw):
    """A clip of a closing door must not be scored as words. Whisper spells
    these differently run to run, which is why the annotation is stripped
    rather than matched."""
    assert lt.normalize(raw) == ""


# ── word error rate ──

def test_identical_is_zero():
    assert lt.wer("set a timer for five minutes",
                  "Set a timer for five minutes.") == 0

def test_one_substitution_in_five_words():
    assert lt.wer("set a timer for five", "set a timer for four") == pytest.approx(1 / 5)

def test_one_deletion():
    assert lt.wer("set a timer", "set timer") == pytest.approx(1 / 3)

def test_one_insertion():
    assert lt.wer("set a timer", "set a big timer") == pytest.approx(1 / 3)

def test_completely_wrong_is_one():
    assert lt.wer("hello there", "wildly different") == 1.0


def test_silence_heard_as_silence_is_correct():
    assert lt.wer("", "[BLANK_AUDIO]") == 0.0


def test_silence_heard_as_words_is_wrong():
    """The Aug 10 2026 runaway loop was an empty room transcribing as "over."
    three times. A scorer that called that correct would rank the model that
    does it as the best one."""
    assert lt.wer("", "over") == 1.0


def test_wer_can_exceed_nothing_but_stays_sane_when_hypothesis_is_long():
    """More words heard than were said is still counted, so a model that
    hallucinates a paragraph over two words cannot score well."""
    assert lt.wer("thanks that's all", "thanks that's all and here is more") > 0


def test_the_real_disagreement_that_motivated_this():
    """base.en heard "7 times for 5 minutes." for what was plainly a timer
    request. Scored against base.en as truth, the prompt looked like a
    regression on this clip. Scored against what was actually said, it is a
    near perfect result and base.en is the one that failed."""
    truth = "set a timer for five minutes"
    base = "7 times for 5 minutes."
    prompted = "Set a timer for five minutes."

    assert lt.wer(truth, prompted) == 0
    assert lt.wer(truth, base) > 0.5


# ── per clip prompts ──
#
# A prompt that differs per clip is the only way to test "give the decoder the
# context it is missing", because the context is different for every follow up.
# The risk is that it silently applies where it should not, so both builders are
# pinned on the turn type they refuse.


def test_a_prior_turn_prompt_is_only_built_for_follow_ups():
    """An initial command gets nothing. Applying context everywhere is what was
    measured on Aug 13 2026 and came out a wash."""
    assert lt.prior_turn_prompt(
        {"turn_type": "initial", "file": "20260811T163519_500_initial.wav"}) is None


def test_the_domain_prompt_variant_refuses_initial_turns():
    assert lt.domain_prompt_followups_only({"turn_type": "followup"}) == lt.PROMPT
    assert lt.domain_prompt_followups_only({"turn_type": "initial"}) is None


def test_an_unparseable_clip_name_yields_no_prior_turns():
    """The manifest is hand editable and these names carry a timestamp by
    convention, not by guarantee. A scoring run must not die on one bad row."""
    assert lt._prior_turns("not-a-stamp.wav") == []


def test_the_prompt_keeps_the_tail_when_history_is_long(monkeypatch):
    """whisper caps the prompt, and the words closest to the clip are the ones
    worth keeping, so this trims from the front rather than the back."""
    monkeypatch.setattr(lt, "_prior_turns", lambda name, limit=2: ["x" * 500])
    prompt = lt.prior_turn_prompt(
        {"turn_type": "followup", "file": "20260811T163519_500_followup.wav"})
    assert len(prompt) == lt.PRIOR_TURN_CHARS


def test_no_history_means_no_prompt_rather_than_an_empty_one(monkeypatch):
    """An empty --prompt is not the same as no --prompt, and passing one would
    measure something nobody asked about."""
    monkeypatch.setattr(lt, "_prior_turns", lambda name, limit=2: [])
    assert lt.prior_turn_prompt(
        {"turn_type": "followup", "file": "20260811T163519_500_followup.wav"}) is None
