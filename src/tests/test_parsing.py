from parsing import strip_leading_bracket_cue


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
