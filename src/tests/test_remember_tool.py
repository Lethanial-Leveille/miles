"""The remember tool, and the duplication it exists to prevent.

The bracket tags it replaces could only do one thing, add a row, and the
duplicate guard was exact string match. So "Traveled to Singapore before summer
2026" landed in the review queue next to two seed rows that already covered the
trip, because the sentences differed even though the fact did not.

Storing, superseding and skipping are three different moves, and only the model
can tell which applies. It can, because every memory is in its prompt with an
id, which is why this needed ids rather than retrieval.
"""

import pytest

import memory_tool
import prompts
from tools import Permission, registry


@pytest.fixture(autouse=True)
def isolated(db, monkeypatch):
    """Point the tool at a throwaway database."""
    monkeypatch.setattr(memory_tool, "save_memory", db.save_memory)
    monkeypatch.setattr(memory_tool, "supersede_memory", db.supersede_memory)
    return db


def _call(**kwargs):
    return registry.call("remember", kwargs)


def _active(db):
    return [c for _, c in db.get_episodic_memories(limit=50)]


def _pending(db):
    return [row[1] for row in db.get_pending_memories()]


def _id_of(db, content):
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute("SELECT id FROM memories WHERE content = ?",
                       (content,)).fetchone()
    conn.close()
    return row[0]


# ── registration ──

def test_registered_as_a_write_tool_with_no_round_trip():
    """A second Claude call to announce a save would make every remembered fact
    cost a full extra turn of latency, for a sentence nobody asked for."""
    spec = registry.get("remember")
    assert spec.permission is Permission.WRITE
    assert spec.returns_to_model is False


def test_supersedes_is_in_the_schema_and_optional():
    schema = registry.get("remember").input_schema
    assert schema["required"] == ["content"]
    assert "supersedes" in schema["properties"]
    assert schema["properties"]["certainty"]["enum"] == ["asked", "inferred"]


def test_the_description_tells_the_model_to_check_first():
    """The whole reason this tool exists rather than a plain save."""
    d = registry.get("remember").description.lower()
    assert "already know" in d
    assert "supersedes" in d


# ── storing ──

def test_asked_is_stored_immediately(db):
    _call(content="exam is Friday", certainty="asked")
    assert _active(db) == ["exam is Friday"]
    assert _pending(db) == []


def test_inferred_goes_to_the_review_queue(db):
    """Nova's guesses do not become things she believes about him until he
    says so. This is the split the explicit and implicit tags encoded."""
    _call(content="he seems tired lately", certainty="inferred")
    assert _active(db) == []
    assert _pending(db) == ["he seems tired lately"]


def test_certainty_defaults_to_inferred(db):
    """Defaulting to asked would put every guess straight into his permanent
    record, which is the expensive direction to be wrong in."""
    _call(content="he seems tired lately")
    assert _pending(db) == ["he seems tired lately"]
    assert _active(db) == []


# ── superseding, the point of the tool ──

def test_superseding_replaces_rather_than_duplicating(db):
    _call(content="exam is Friday", certainty="asked")
    old = _id_of(db, "exam is Friday")

    _call(content="exam is Thursday", certainty="asked", supersedes=old)

    assert _active(db) == ["exam is Thursday"]


def test_the_superseded_version_is_still_readable(db):
    _call(content="exam is Friday", certainty="asked")
    old = _id_of(db, "exam is Friday")
    _call(content="exam is Thursday", certainty="asked", supersedes=old)

    chain = db.get_memory_chain(_id_of(db, "exam is Thursday"))
    assert [row[1] for row in chain] == ["exam is Thursday", "exam is Friday"]


def test_a_bad_id_stores_rather_than_losing_the_fact(db):
    """A wrong reference is a worse reason to drop information than a duplicate
    is to keep it. The model is told, so it can correct itself."""
    result = _call(content="exam is Thursday", certainty="asked", supersedes=9999)
    assert "stored as new" in result
    assert _active(db) == ["exam is Thursday"]


# ── skipping ──

def test_an_exact_duplicate_is_not_stored_twice(db):
    _call(content="exam is Friday", certainty="asked")
    result = _call(content="exam is Friday", certainty="asked")
    assert "already stored" in result
    assert _active(db) == ["exam is Friday"]


def test_a_reworded_duplicate_is_not_caught_by_the_guard(db):
    """Documenting the limit rather than pretending it away. The guard is exact
    string match, so this is exactly the case that produced the Singapore
    duplicate, and it is the model reading its own memory list that prevents it,
    not the database."""
    _call(content="exam is Friday", certainty="asked")
    _call(content="his exam takes place on Friday", certainty="asked")
    assert len(_active(db)) == 2


# ── shelf life ──

def test_a_temporary_fact_expires_after_its_date(db):
    from datetime import datetime, timedelta
    past = (datetime.now() - timedelta(days=1)).isoformat()
    _call(content="training for the March meet", certainty="asked",
          temporary=True, until=past)
    assert _active(db) == []


def test_a_temporary_fact_before_its_date_still_applies(db):
    from datetime import datetime, timedelta
    future = (datetime.now() + timedelta(days=30)).isoformat()
    _call(content="training for the December meet", certainty="asked",
          temporary=True, until=future)
    assert _active(db) == ["training for the December meet"]


def test_the_description_warns_that_temporary_needs_a_date(db):
    """volatile without a date never expires, so the date is the whole point."""
    d = registry.get("remember").input_schema["properties"]["temporary"]["description"]
    assert "until" in d and "never expires" in d


# ── the ids that make supersedes possible ──

def test_memory_ids_appear_in_the_prompt(db):
    """Without a name for a memory, the model cannot point at one, which is why
    ids mattered more than retrieval did."""
    block = prompts._episodic_block([(61, "exam is Friday")])
    assert "(#61)" in block


def test_seed_ids_appear_too(db):
    block = prompts._seed_block([(61, "sleep was poor in Singapore", "sleep")])
    assert "(#61)" in block


def test_the_prompt_forbids_reading_ids_aloud():
    """They are addressing handles for tool calls. Spoken, they are noise."""
    assert "never spoken" in prompts.MEMORY_INSTRUCTIONS.lower()


def test_the_prompt_teaches_all_three_moves():
    text = prompts.MEMORY_INSTRUCTIONS.lower()
    assert "supersede" in text
    assert "do nothing" in text
    assert "already know" in text


def test_the_bracket_tag_instructions_are_gone():
    """Leaving both paths live is how the duplicate happened. The prompt teaches
    the tool only."""
    assert "[MEMORY-EXPLICIT:" not in prompts.MEMORY_INSTRUCTIONS
    assert "[MEMORY:" not in prompts.MEMORY_INSTRUCTIONS


def test_brain_no_longer_saves_extracted_tags():
    """A stray tag is a leak now, not an instruction. Saving it as well would
    double write whatever the tool already stored."""
    import inspect
    import brain
    source = inspect.getsource(brain.ask_nova_async)
    assert "extract_memories" in source, "still stripped so it is not spoken"
    assert "save_memory" not in source, "but no longer saved"


# ── ids must actually identify the memory they are printed next to ──

def test_every_rendered_id_matches_its_row(db):
    """The id is an addressing handle. If it drifts from its content, Nova
    supersedes the wrong memory, which silently destroys a correct fact and
    replaces it with an unrelated one. This is the worst failure this feature
    can have and it would be invisible in her replies."""
    import re
    _call(content="exam is Friday", certainty="asked")
    _call(content="bench is 225", certainty="asked")

    rows = db.get_episodic_memories(limit=50)
    block = prompts._episodic_block(rows)
    rendered = dict((int(i), c) for i, c in re.findall(r'\(#(\d+)\) (.+)', block))

    assert rendered == {mid: content for mid, content in rows}


def test_seed_ids_survive_category_grouping(db):
    """The seed block regroups rows by category before rendering. Grouping is
    exactly where an id could get separated from its text."""
    import re
    seeds = [
        (10, "studies computer engineering", "academics"),
        (11, "bench is 225", "training"),
        (12, "graduates in 2029", "academics"),
        (13, "runs a marathon", "training"),
    ]
    block = prompts._seed_block(seeds)
    rendered = dict((int(i), c) for i, c in re.findall(r'\(#(\d+)\) (.+)', block))

    assert rendered == {mid: content for mid, content, _ in seeds}


def test_every_id_in_the_prompt_is_a_real_memory(db):
    """No invented ids anywhere in the prompt, including in instructions.

    The memory instructions originally used (#61) and (#42) as examples. Nova
    cannot tell an example from a real row, so she could have tried to supersede
    #42 purely because it appeared beside the word supersede. That is the same
    failure as 5ad97de, where she copied the reminder date out of the example in
    her own prompt and dated every reminder months in the past."""
    import re
    _call(content="exam is Friday", certainty="asked")

    seeds = [(900, "seed fact", "general"), (901, "another seed", "general")]
    episodic = db.get_episodic_memories(limit=50)
    prompt = prompts.build_enhanced_prompt(seeds, episodic, "voice")

    real = {mid for mid, _, _ in seeds} | {mid for mid, _ in episodic}
    found = {int(i) for i in re.findall(r'\(#(\d+)\)', prompt)}
    assert found == real, f"prompt names memories that do not exist: {found - real}"


def test_the_instructions_contain_no_example_ids():
    """Guards the fix directly, so a future edit cannot reintroduce a number
    that looks addressable but is not."""
    import re
    assert re.search(r'\(#\d+\)', prompts.MEMORY_INSTRUCTIONS) is None


def test_ids_are_unique_in_the_prompt(db):
    """A repeated id would make supersedes ambiguous."""
    import re
    _call(content="exam is Friday", certainty="asked")
    prompt = prompts.build_enhanced_prompt(
        [(900, "seed fact", "general")], db.get_episodic_memories(), "voice")
    ids = re.findall(r'\(#(\d+)\)', prompt)
    assert len(ids) == len(set(ids))


def test_a_superseded_memory_stops_being_addressable(db):
    """Its id must leave the prompt with it. Otherwise Nova could supersede a
    row that is already retired, building a chain off a dead branch."""
    import re
    _call(content="exam is Friday", certainty="asked")
    old = _id_of(db, "exam is Friday")
    _call(content="exam is Thursday", certainty="asked", supersedes=old)

    block = prompts._episodic_block(db.get_episodic_memories(limit=50))
    assert f"(#{old})" not in block
    assert f"(#{_id_of(db, 'exam is Thursday')})" in block
