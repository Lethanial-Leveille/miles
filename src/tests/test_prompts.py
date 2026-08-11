from prompts import build_enhanced_prompt


SEED_ROWS = [
    (1, "Full name is Lethanial LeeAndon Leveille.", "identity"),
    (2, "Born in Hollywood, Florida.", "identity"),
    (3, "Has a younger sister, Rye.", "family"),
]
EPISODIC_ROWS = [(10, "Exam is Friday.")]


def test_prompt_without_memories_omits_memory_blocks():
    prompt = build_enhanced_prompt([], [], device="pi")
    assert "WHAT YOU KNOW ABOUT LETHANIAL" not in prompt
    assert "THINGS LETHANIAL HAS TOLD YOU TO REMEMBER" not in prompt


def test_prompt_with_seed_memories_groups_by_category():
    prompt = build_enhanced_prompt(SEED_ROWS, [], device="pi")
    assert "WHAT YOU KNOW ABOUT LETHANIAL" in prompt
    assert "IDENTITY:" in prompt
    assert "FAMILY:" in prompt
    assert "- Full name is Lethanial LeeAndon Leveille." in prompt
    assert "- Has a younger sister, Rye." in prompt
    # identity heading should appear before family heading (query orders by category)
    assert prompt.index("IDENTITY:") < prompt.index("FAMILY:")


def test_prompt_with_episodic_memories_in_separate_block():
    prompt = build_enhanced_prompt([], EPISODIC_ROWS, device="pi")
    assert "THINGS LETHANIAL HAS TOLD YOU TO REMEMBER" in prompt
    assert "- Exam is Friday." in prompt


def test_prompt_memory_context_section_removed():
    prompt = build_enhanced_prompt([], [], device="pi")
    assert "MEMORY CONTEXT" not in prompt


def test_prompt_general_knowledge_present_for_both_devices():
    for device in ("pi", "app"):
        prompt = build_enhanced_prompt([], [], device=device)
        assert "GENERAL KNOWLEDGE:" in prompt
        assert "broad general knowledge" in prompt


def test_prompt_voice_device_uses_short_response_length_and_spelled_numbers():
    prompt = build_enhanced_prompt([], [], device="pi")
    assert "Answer only what he asked" in prompt
    assert "Always spell out numbers as words" in prompt
    assert "Give a brief answer first" not in prompt
    assert "Use normal numerals" not in prompt


def test_voice_prompt_has_no_hard_sentence_ceiling():
    """The old wording was a ceiling with no way out ("3 sentences maximum"),
    which conflicted with answering the question; the model resolved that by
    declining technical questions instead of exceeding the cap. The number
    survives only as a target, and always beside an explicit release."""
    prompt = build_enhanced_prompt([], [], device="pi")
    assert "3 sentences maximum" not in prompt
    assert "maximum" not in prompt.split("RESPONSE LENGTH:")[1].split("\n\n\n")[0]
    assert "That target is a habit, not a limit" in prompt


def test_voice_prompt_forbids_refusing_for_brevity():
    """Guards the specific regression: Nova claiming a technical question was
    outside her knowledge base in order to stay short."""
    prompt = build_enhanced_prompt([], [], device="pi")
    assert "Never refuse a question" in prompt
    assert "never say something is outside what you know in order to stay short" in prompt


def test_prompt_app_device_uses_longer_response_length_and_normal_numerals():
    prompt = build_enhanced_prompt([], [], device="app")
    assert "Give a brief answer first" in prompt
    assert "Use normal numerals" in prompt
    assert "Keep responses to 1 to 2 sentences" not in prompt
    assert "Always spell out numbers as words" not in prompt


def test_prompt_cannot_do_scoped_to_actions_not_questions():
    prompt = build_enhanced_prompt([], [], device="pi")
    assert "applies only to actions" in prompt
