"""Tests for the capability slot in build_enhanced_prompt.

Separate from test_prompts.py, which covers the pre migration prompt assembly.
Keeping these apart means Phase 2 can delete the legacy assertions without
disturbing the ones that outlive the migration.
"""

import pytest

import prompts
from tools import Permission, ToolRegistry


@pytest.fixture
def reg():
    """A throwaway registry swapped in for the production singleton, so these
    tests do not depend on which tools happen to be registered yet."""
    return ToolRegistry()


@pytest.fixture
def native(monkeypatch, reg):
    """An isolated registry in place of the production singleton, so these
    tests do not depend on which tools happen to be registered."""
    monkeypatch.setattr(prompts, "registry", reg)
    return reg


def _register(reg, name, description):
    @reg.register(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        permission=Permission.READ,
        returns_to_model=True,
    )
    def handler():
        return None


def _build():
    return prompts.build_enhanced_prompt(seed_rows=[], episodic_rows=[], channel="voice")


# ── flag routing ──

def test_native_path_uses_the_generated_block(native):
    _register(native, "get_weather", "Current outdoor conditions. More detail.")
    prompt = _build()
    assert "YOUR TOOLS:" in prompt
    assert "Current outdoor conditions" in prompt
    # The whole point: the hand written tag instructions are gone.
    assert "ACTION INSTRUCTION:" not in prompt
    assert "[ACTION: weather" not in prompt


def test_native_path_with_empty_registry_claims_nothing(native):
    """An empty registry must produce no capability block at all, not a header
    with nothing under it. Claiming zero tools is correct; claiming a heading
    over an empty list invites the model to invent entries."""
    prompt = _build()
    assert "YOUR TOOLS:" not in prompt
    assert "ACTION INSTRUCTION:" not in prompt


def test_prompt_can_only_claim_registered_tools(native):
    """The reason the block is generated rather than written by hand."""
    _register(native, "get_weather", "Current outdoor conditions.")
    prompt = _build()
    assert "Current outdoor conditions" in prompt
    assert "set_timer" not in prompt

    # Hevy used to be banned from the prompt outright, on the reasoning that
    # naming an unregistered integration invites the model to claim it. That
    # still holds for any positive mention, but Lethanial asked to be told the
    # log is missing when lifting numbers come up, which is the opposite of a
    # capability claim and is the only thing that will remind him to add it.
    #
    # So the ban narrows rather than lifts: the name may appear, and only
    # inside a sentence that says it is not connected.
    lowered = prompt.lower()
    for sentence in lowered.split("."):
        if "hevy" in sentence:
            assert "not connected" in sentence, (
                f"hevy named without saying it is unavailable: {sentence.strip()!r}")


# ── blocks that survive both paths ──

def test_memory_instructions_survive_the_migration(native):
    """The memory section outlived the action tag migration, then became tool
    instructions of its own when remember replaced the bracket tags."""
    prompt = _build()
    assert "MEMORY:" in prompt
    assert "supersede" in prompt.lower()
    assert "[MEMORY-EXPLICIT:" not in prompt, "the tag path is gone"


def test_clock_instructions_survive_the_migration(native):
    """Regression guard for 5ad97de. The clock paragraph used to sit inside the
    action tag block; deleting that block in Phase 2 would have taken it along
    and Nova would go back to copying the date out of her own prompt, dating
    every reminder months in the past so none could ever fire."""
    prompt = _build()
    assert "CLOCK:" in prompt
    assert "supplied at the end of every message" in prompt
    assert "A reminder dated in the past will never fire" in prompt


def test_clock_block_asks_for_only_what_was_requested(native):
    prompt = _build()
    assert "Answer only the part that was asked" in prompt


# ── ordering, which the cache depends on ──

def test_volatile_episodic_block_stays_last(native):
    """Everything before the episodic block is comparatively stable. If the
    capability block ever landed after it, a memory save would invalidate the
    tool definitions too."""
    _register(native, "get_weather", "Current outdoor conditions.")
    prompt = prompts.build_enhanced_prompt(
        seed_rows=[], episodic_rows=[(1, "zzmarkerzz distinctive memory")], channel="voice"
    )
    # A marker string, because instruction prose can legitimately contain
    # memory shaped sentences and index() finds the first occurrence.
    assert prompt.index("YOUR TOOLS:") < prompt.index("zzmarkerzz")
    assert prompt.index("CLOCK:") < prompt.index("zzmarkerzz")


def test_capability_block_order_is_stable_across_builds(native):
    """Registration order must not reach the prompt. Tools render ahead of the
    system prompt in the cached prefix, so a reordering would move every byte
    after it and silently drop every cache hit."""
    _register(native, "set_timer", "Start a countdown.")
    _register(native, "get_weather", "Current outdoor conditions.")
    assert _build() == _build()
    assert _build().index("Current outdoor conditions") < _build().index("Start a countdown")


def test_channel_still_selects_voice_or_text_blocks(native):
    _register(native, "get_weather", "Current outdoor conditions.")
    voice = prompts.build_enhanced_prompt([], [], channel="voice")
    text = prompts.build_enhanced_prompt([], [], channel="text")
    assert "The voice synthesizer reads digits incorrectly" in voice
    assert "You are writing, not speaking through a voice synthesizer" in text
