import re

import pytest

import prompts
from prompts import build_enhanced_prompt


# Synthetic on purpose. These assert prompt assembly, not the contents of the
# real corpus, and real personal facts do not belong in a tracked test.
SEED_ROWS = [
    (1, "Drinks tea, never coffee.", "identity"),
    (2, "Left handed.", "identity"),
    (3, "Has an older brother.", "family"),
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
    assert "Drinks tea, never coffee." in prompt
    assert "Has an older brother." in prompt
    # identity heading should appear before family heading (query orders by category)
    assert prompt.index("IDENTITY:") < prompt.index("FAMILY:")


def test_prompt_with_episodic_memories_in_separate_block():
    prompt = build_enhanced_prompt([], EPISODIC_ROWS, channel="voice")
    assert "THINGS LETHANIAL HAS TOLD YOU TO REMEMBER" in prompt
    assert "Exam is Friday." in prompt


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


# ── the transcript is not his words ──

def test_every_tier_is_told_the_input_is_a_transcript():
    """Nova had no notion that what reaches her is speech recognition output.

    On Aug 13 2026 "Where do you think that I live right now?" arrived as the
    fragment "live right now." Asked afterwards what he had just said, she did
    not report the fragment. She reconstructed an intent for it, landed on
    wording from the lower_access tool description, and told him he had asked
    to drop his own clearance. No tool was ever called and his tier never
    changed, but he had every reason to believe otherwise.

    Present at every tier, because transcription is no more reliable when the
    person speaking is not him."""
    for tier in ("hokage", "jonin", "genin"):
        prompt = build_enhanced_prompt([], [], channel="voice", tier=tier)
        assert "WHAT REACHES YOU" in prompt, tier
        assert "speech recognition output" in prompt, tier


def test_prompt_forbids_reconstructing_intent_from_a_fragment():
    prompt = build_enhanced_prompt([], [], channel="voice")
    assert "Never reconstruct an intent from a fragment" in prompt
    # The specific trap: a fragment matching tool wording, presented back as
    # his meaning, reads as a report that the tool ran.
    assert "belongs to a tool" in prompt


def test_transcript_warning_precedes_the_instruction_to_answer_confidently():
    """GENERAL KNOWLEDGE tells her to answer directly and confidently from what
    she has. This is the counterweight and has to be read alongside it, not
    hundreds of lines later among the tier specific blocks."""
    prompt = build_enhanced_prompt([], [], channel="voice")
    knowledge = prompt.index("GENERAL KNOWLEDGE:")
    reaches = prompt.index("WHAT REACHES YOU:")
    assert 0 < reaches - knowledge < 1200, "the two blocks have drifted apart"



@pytest.mark.parametrize("tier", ["hokage", "jonin", "genin"])
def test_results_are_interpreted_not_recited_at_every_tier(tier):
    """Sep 13 2026: asked how he slept, Nova read every field of the tool result
    in order. The instruction that stops it has to reach every speaker, because
    every tier can call a READ tool."""
    prompt = prompts.build_enhanced_prompt([], [], "voice", [], tier=tier)
    assert "TALKING ABOUT WHAT A TOOL GIVES YOU" in prompt
    assert "Only say what the numbers actually show" in prompt


def test_personality_no_longer_asks_for_distance():
    """"Warm but never overly familiar" and "clean, well structured sentences"
    were the performed composure he heard as reading, not speaking."""
    assert "never overly familiar" not in prompts.SYSTEM_PROMPT_HEADER
    assert "well structured sentences" not in prompts.SYSTEM_PROMPT_HEADER
    assert "answer the feeling first" in prompts.SYSTEM_PROMPT_HEADER
