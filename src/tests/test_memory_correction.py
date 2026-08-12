"""Supersede and expiry.

Both existed as columns and neither did anything: superseded_by was declared and
never written, volatile and references_date were written and never read. The
store was append only in practice, so a memory recorded wrong stayed wrong and
the only remedy was deleting the row, which also destroyed the fact that it had
ever changed.

That is why the remember tool was blocked on this. Automating writes into a
store that cannot be corrected makes errors permanent and accumulate, and the
failure gets worse the better the tool works.
"""

from datetime import datetime, timedelta

import pytest


def _save(db, content, **kw):
    db.save_memory(content, source=kw.pop("source", "explicit"), **kw)
    rows = db.get_episodic_memories(limit=50)
    return rows[0][0] if rows else None


def _episodic(db):
    return [content for _, content in db.get_episodic_memories(limit=50)]


def _id_of(db, content):
    """Look the id up directly rather than through get_episodic_memories, which
    filters expired rows out and so cannot find a memory that is already past
    its date."""
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute("SELECT id FROM memories WHERE content = ?",
                       (content,)).fetchone()
    conn.close()
    return row[0]


def _past():
    return (datetime.now() - timedelta(days=1)).isoformat()


def _future():
    return (datetime.now() + timedelta(days=30)).isoformat()


# ── supersede ──

def test_the_replacement_is_what_nova_sees(db):
    old = _save(db, "exam is Friday")
    db.supersede_memory(old, "exam is Thursday")
    assert _episodic(db) == ["exam is Thursday"]


def test_the_old_row_survives_rather_than_being_deleted(db):
    """The whole point. An exam that moved is different information from an exam
    that was always Thursday, and a delete cannot tell them apart."""
    old = _save(db, "exam is Friday")
    new = db.supersede_memory(old, "exam is Thursday")

    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute(
        "SELECT content, status, superseded_by, superseded_at FROM memories WHERE id = ?",
        (old,)
    ).fetchone()
    conn.close()
    assert row[0] == "exam is Friday"
    assert row[1] == "superseded"
    assert row[2] == new
    assert row[3] is not None


def test_superseding_an_unknown_id_returns_none(db):
    assert db.supersede_memory(9999, "whatever") is None


def test_classification_is_inherited_unless_overridden(db):
    """A correction is usually the same kind of fact. Re-specifying every field
    to fix a typo is how fields drift apart."""
    db.save_memory("meet is in March", source="explicit", category="training",
                   volatile=True, references_date=_future())
    old = _id_of(db, "meet is in March")
    new = db.supersede_memory(old, "meet is in April")

    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    row = conn.execute(
        "SELECT category, source, volatile FROM memories WHERE id = ?", (new,)
    ).fetchone()
    conn.close()
    assert row == ("training", "explicit", 1)


def test_an_override_wins_over_inheritance(db):
    db.save_memory("meet is in March", volatile=True,
                   references_date=_future())
    old = _id_of(db, "meet is in March")
    new = db.supersede_memory(old, "meet is permanent", volatile=False)

    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    volatile = conn.execute("SELECT volatile FROM memories WHERE id = ?",
                            (new,)).fetchone()[0]
    conn.close()
    assert volatile == 0


def test_a_correction_can_itself_be_corrected(db):
    first = _save(db, "exam is Friday")
    second = db.supersede_memory(first, "exam is Thursday")
    third = db.supersede_memory(second, "exam is Wednesday")
    assert _episodic(db) == ["exam is Wednesday"]
    assert third is not None


# ── the chain ──

def test_the_chain_walks_back_through_every_version(db):
    first = _save(db, "exam is Friday")
    second = db.supersede_memory(first, "exam is Thursday")
    third = db.supersede_memory(second, "exam is Wednesday")

    chain = db.get_memory_chain(third)
    assert [row[1] for row in chain] == [
        "exam is Wednesday", "exam is Thursday", "exam is Friday"]


def test_the_chain_of_an_uncorrected_memory_is_just_itself(db):
    only = _save(db, "bench is 225")
    assert len(db.get_memory_chain(only)) == 1


def test_the_chain_of_an_unknown_id_is_empty(db):
    assert db.get_memory_chain(9999) == []


# ── expiry ──

def test_a_volatile_memory_past_its_date_stops_reaching_the_prompt(db):
    db.save_memory("training for the March meet", volatile=True,
                   references_date=_past())
    assert _episodic(db) == []


def test_a_volatile_memory_before_its_date_still_reaches_the_prompt(db):
    db.save_memory("training for the December meet", volatile=True,
                   references_date=_future())
    assert _episodic(db) == ["training for the December meet"]


def test_a_volatile_memory_with_no_date_never_expires(db):
    """volatile alone says the fact is temporary, not when it stops. Expiring
    without a date would be guessing."""
    db.save_memory("currently reading Naruto", volatile=True)
    assert _episodic(db) == ["currently reading Naruto"]


def test_a_permanent_memory_with_a_past_date_is_unaffected(db):
    """references_date on a non volatile memory is just context about when the
    fact refers to, not a shelf life."""
    db.save_memory("graduated high school in 2025", volatile=False,
                   references_date=_past())
    assert _episodic(db) == ["graduated high school in 2025"]


def test_expiry_is_enforced_at_read_not_by_a_sweep(db):
    """No job to schedule and nothing to fall out of sync. The memory is hidden
    the moment its date passes, whether or not expire_memories has ever run."""
    db.save_memory("stale fact", volatile=True, references_date=_past())
    assert _episodic(db) == []

    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    status = conn.execute(
        "SELECT status FROM memories WHERE content = 'stale fact'").fetchone()[0]
    conn.close()
    assert status == "active", "still marked active, yet already invisible"


def test_the_sweep_marks_what_the_read_already_hides(db):
    db.save_memory("stale fact", volatile=True, references_date=_past())
    db.save_memory("fresh fact", volatile=True, references_date=_future())

    assert db.expire_memories() == 1

    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    rows = dict(conn.execute("SELECT content, status FROM memories "
                             "WHERE source = 'explicit'").fetchall())
    conn.close()
    assert rows["stale fact"] == "expired"
    assert rows["fresh fact"] == "active"


def test_the_sweep_is_idempotent(db):
    db.save_memory("stale fact", volatile=True, references_date=_past())
    assert db.expire_memories() == 1
    assert db.expire_memories() == 0


# ── retired memories stay out of every retrieval path ──

@pytest.mark.parametrize("retire", ["supersede", "expire"])
def test_retired_memories_leave_the_active_listing(db, retire):
    db.save_memory("exam is Friday")
    mid = db.get_episodic_memories()[0][0]

    if retire == "supersede":
        db.supersede_memory(mid, "exam is Thursday")
        expected_remaining = 1          # the replacement
    else:
        import sqlite3
        conn = sqlite3.connect(db.DB_PATH)
        conn.execute("UPDATE memories SET volatile = 1, references_date = ? "
                     "WHERE id = ?", (_past(), mid))
        conn.commit()
        conn.close()
        db.expire_memories()
        expected_remaining = 0

    rows, _ = db.get_active_memories()
    contents = [r[1] for r in rows]
    assert "exam is Friday" not in contents
    assert len(_episodic(db)) == expected_remaining
