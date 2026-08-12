import pytest

from parsing import strip_leading_bracket_cue, is_noise_transcript


def test_strips_unrecognized_leading_bracket():
    result = strip_leading_bracket_cue("[clearly] Connection confirmed.")
    assert result == "Connection confirmed."


def test_strips_multi_word_leading_bracket():
    result = strip_leading_bracket_cue("[with clear confirmation] Hello Lethanial.")
    assert result == "Hello Lethanial."


def test_leaves_text_without_leading_bracket_untouched():
    text = "No bracket here at all."
    assert strip_leading_bracket_cue(text) == text


def test_leaves_action_tag_untouched():
    text = "[ACTION: weather] Let me check on that."
    assert strip_leading_bracket_cue(text) == text


def test_leaves_memory_tag_untouched():
    text = "[MEMORY: likes pizza] Noted."
    assert strip_leading_bracket_cue(text) == text


def test_leaves_memory_explicit_tag_untouched():
    text = "[MEMORY-EXPLICIT: exam is Friday] I'll remember that."
    assert strip_leading_bracket_cue(text) == text


def test_empty_string_is_noop():
    assert strip_leading_bracket_cue("") == ""


def test_bracket_not_at_start_is_untouched():
    text = "Connection confirmed [clearly] and stable."
    assert strip_leading_bracket_cue(text) == text


def test_seen_set_suppresses_duplicate_log_but_still_strips(capsys):
    seen = set()
    first = strip_leading_bracket_cue("[clearly] First sentence.", seen)
    capsys.readouterr()  # discard first log
    second = strip_leading_bracket_cue("[clearly] Second sentence.", seen)
    captured = capsys.readouterr()

    # both calls still strip the bracket even though only the first logs
    assert first == "First sentence."
    assert second == "Second sentence."
    assert captured.out == ""


def test_seen_set_logs_each_distinct_leak_once():
    seen = set()
    strip_leading_bracket_cue("[clearly] One.", seen)
    strip_leading_bracket_cue("[dryly] Two.", seen)
    assert seen == {"clearly", "dryly"}


# ── is_noise_transcript ──

@pytest.mark.parametrize("transcript", [
    "",
    None,
    "   ",
    "[BLANK_AUDIO]",
    "(beep)",
    "[ Silence ]",
    "over.",
    "Over",
    "  over.  ",
    "Enola.",
    "you",
    "Thank you.",
    "Thanks for watching!",
    "(wind blowing)",
])
def test_noise_transcripts_are_rejected(transcript):
    assert is_noise_transcript(transcript) is True


@pytest.mark.parametrize("transcript", [
    "What's the weather today?",
    "Set a timer for fifteen minutes",
    "yes",
    "no",
    "why",
    "What about in game 3?",
    "8-pack abs.",
    "Do you have a favorite Pokemon?",
])
def test_real_speech_is_kept(transcript):
    assert is_noise_transcript(transcript) is False


def test_noise_word_inside_a_real_sentence_is_kept():
    """'over' alone is noise, but the same word inside a sentence is speech.
    Guards against a regression to substring matching."""
    assert is_noise_transcript("Read that over for me") is False
    assert is_noise_transcript("Is the timer over yet?") is False


def test_annotation_alongside_speech_is_kept():
    assert is_noise_transcript("(beep) what's the weather") is False


# ── wake phrase during a follow up ──

@pytest.mark.parametrize("said,expected_rest", [
    ("Hey Nova, what is the weather?", "what is the weather?"),
    ("hey, nova. remind me",           "remind me"),
    ("HEY NOVA set a timer",           "set a timer"),
    ("hey nova",                       ""),
])
def test_wake_phrase_is_split_off(said, expected_rest):
    from parsing import split_wake_phrase
    heard, rest = split_wake_phrase(said)
    assert heard is True
    assert rest == expected_rest


@pytest.mark.parametrize("said", [
    "What about tomorrow?",
    "Nova what time is it",          # bare name is not enough, deliberately
    "Nova told me it was raining",   # him talking about her, not to her
    "Novak Djokovic is playing",     # the word boundary earns its keep here
    "",
])
def test_ordinary_follow_ups_are_left_alone(said):
    """A false positive here hijacks a follow up into a new turn and plays a
    chime over him, so the phrase has to be required in full. A bare "Nova"
    would also match him saying her name to somebody else in the room, which is
    the addressing problem rather than a fix for it."""
    from parsing import split_wake_phrase
    heard, rest = split_wake_phrase(said)
    assert heard is False
    assert rest == said
