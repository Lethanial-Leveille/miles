import re
from prompts import build_enhanced_prompt


SEED_ROWS = [
    (1, "Full name is Lethanial LeeAndon Leveille.", "identity"),
    (2, "Born in Hollywood, Florida.", "identity"),
    (3, "Has a younger sister, Rye.", "family"),
]
EPISODIC_ROWS = [(10, "Exam is Friday.")]


def test_prompt_without_memories_omits_memory_blocks():
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "WHAT YOU KNOW ABOUT LETHANIAL" not in prompt
    assert "THINGS LETHANIAL HAS TOLD YOU TO REMEMBER" not in prompt


def test_prompt_with_seed_memories_groups_by_category():
    prompt = build_enhanced_prompt(SEED_ROWS, [], channel="voice")
    assert "WHAT YOU KNOW ABOUT LETHANIAL" in prompt
    assert "IDENTITY:" in prompt
    assert "FAMILY:" in prompt
    assert "- Full name is Lethanial LeeAndon Leveille." in prompt
    assert "- Has a younger sister, Rye." in prompt
    # identity heading should appear before family heading (query orders by category)
    assert prompt.index("IDENTITY:") < prompt.index("FAMILY:")


def test_prompt_with_episodic_memories_in_separate_block():
    prompt = build_enhanced_prompt([], EPISODIC_ROWS, channel="voice")
    assert "THINGS LETHANIAL HAS TOLD YOU TO REMEMBER" in prompt
    assert "- Exam is Friday." in prompt


def test_prompt_memory_context_section_removed():
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "MEMORY CONTEXT" not in prompt


def test_prompt_general_knowledge_present_for_both_channels():
    for channel in ("voice", "text"):
        prompt = build_enhanced_prompt([], [], channel=channel)
        assert "GENERAL KNOWLEDGE:" in prompt
        assert "broad general knowledge" in prompt


def test_prompt_voice_channel_uses_short_response_length_and_spelled_numbers():
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "Answer only what was asked" in prompt
    assert "Always spell out numbers as words" in prompt
    assert "Give a brief answer first" not in prompt
    assert "Use normal numerals" not in prompt


def test_voice_prompt_has_no_hard_sentence_ceiling():
    """The old wording was a ceiling with no way out ("3 sentences maximum"),
    which conflicted with answering the question; the model resolved that by
    declining technical questions instead of exceeding the cap. The number
    survives only as a target, and always beside an explicit release."""
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "3 sentences maximum" not in prompt
    assert "maximum" not in prompt.split("RESPONSE LENGTH:")[1].split("\n\n\n")[0]
    assert "That target is a habit, not a limit" in prompt


def test_voice_prompt_requires_second_person_address():
    """The length block previously used "he" eight times, which primed Nova to
    talk about Lethanial in the third person instead of to him."""
    from prompts import RESPONSE_LENGTH_VOICE

    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "Never refer to him in the third person" in prompt
    # Checked against the constant rather than sliced out of the assembled
    # prompt: slicing on a separator silently swallowed later sections and
    # counted their pronouns instead of this block's.
    #
    # The rule forbidding third person has to say "him" to state itself, so a
    # small count is expected. Eight is what caused the problem.
    assert len(re.findall(r"\b(he|him|his)\b", RESPONSE_LENGTH_VOICE, re.I)) <= 5


def test_voice_prompt_forbids_refusing_for_brevity():
    """Guards the specific regression: Nova claiming a technical question was
    outside her knowledge base in order to stay short."""
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "Never refuse a question" in prompt
    assert "never say something is outside what you know in order to stay short" in prompt


def test_prompt_text_channel_uses_longer_response_length_and_normal_numerals():
    prompt = build_enhanced_prompt([], [], channel="text")
    assert "Give a brief answer first" in prompt
    assert "Use normal numerals" in prompt
    assert "Keep responses to 1 to 2 sentences" not in prompt
    assert "Always spell out numbers as words" not in prompt


def test_prompt_cannot_do_scoped_to_actions_not_questions():
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "applies only to actions" in prompt
