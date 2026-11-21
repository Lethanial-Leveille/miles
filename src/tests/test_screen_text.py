import pytest

from screen_text import for_display, numerals, undash


@pytest.mark.parametrize("spoken, written", [
    ("You have ninety six active applications.", "You have 96 active applications."),
    ("Ninety six is a lot.", "96 is a lot."),
    ("That's twenty days.", "That's 20 days."),
    ("twelve point seven five", "12.75"),
    ("Up fifteen percent today.", "Up 15% today."),
    ("five percent", "5%"),
    ("seven point five hours", "7.5 hours"),
    ("one hundred and twenty pounds", "120 pounds"),
    ("two thousand four hundred steps", "2400 steps"),
    ("twenty-five reps", "25 reps"),
    ("ten minutes", "10 minutes"),
])
def test_spelled_numbers_become_digits(spoken, written):
    assert numerals(spoken) == written


@pytest.mark.parametrize("text", [
    "Four reps means you're past that.",
    "one of those days",
    "It's eleven fifty eight.",
    "Born in nineteen eighty four.",
    "two and three",
    "Someone said hi.",
    "No numbers here.",
])
def test_small_numbers_and_ambiguous_runs_stay_as_written(text):
    assert numerals(text) == text


def test_a_comma_separates_two_numbers():
    assert numerals("twenty, thirty, forty") == "20, 30, 40"


@pytest.mark.parametrize("dashed, plain", [
    ("Five have gone quiet—Stoke, Adobe.", "Five have gone quiet, Stoke, Adobe."),
    ("Nova is me — the voice of Miles.", "Nova is me, the voice of Miles."),
    ("Cats sniff you—they read your scent.", "Cats sniff you, they read your scent."),
    ("Do 5–8 reps.", "Do 5 to 8 reps."),
    ("It ends here—.", "It ends here."),
    ("a - b", "a, b"),
])
def test_dashes_become_commas(dashed, plain):
    assert undash(dashed) == plain


@pytest.mark.parametrize("text", [
    "push-ups", "2026-11-21", "https://example.com/a-b", "Wi-Fi",
])
def test_hyphens_inside_words_are_left_alone(text):
    assert undash(text) == text


def test_voice_keeps_number_words_but_loses_dashes():
    assert for_display("ninety six—that's it", "voice") == "ninety six, that's it"


def test_text_gets_both():
    assert for_display("ninety six—that's it", "text") == "96, that's it"


def test_empty_reply_is_untouched():
    assert for_display("", "text") == ""
