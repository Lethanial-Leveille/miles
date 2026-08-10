def test_save_memory_dedup_skips_duplicate_active(db):
    db.save_memory("Likes pull ups", source="seed", status="active", category="fitness")
    db.save_memory("Likes pull ups", source="seed", status="active", category="fitness")

    rows = db.get_seed_memories()
    assert len(rows) == 1


def test_save_memory_dedup_only_checks_active_rows(db):
    # A pending row with the same content should not block a new active insert.
    db.save_memory("Exam is Friday", source="implicit", status="pending")
    db.save_memory("Exam is Friday", source="explicit", status="active")

    pending = db.get_pending_memories()
    episodic = db.get_episodic_memories()
    assert len(pending) == 1
    assert len(episodic) == 1


def test_get_seed_memories_filters_source_and_status(db):
    db.save_memory("Seed active", source="seed", status="active", category="identity")
    db.save_memory("Seed pending", source="seed", status="pending", category="identity")
    db.save_memory("Explicit active", source="explicit", status="active")

    rows = db.get_seed_memories()
    contents = [r[1] for r in rows]
    assert contents == ["Seed active"]


def test_get_seed_memories_ordered_by_category_then_id(db):
    db.save_memory("Family fact", source="seed", status="active", category="family")
    db.save_memory("Identity fact one", source="seed", status="active", category="identity")
    db.save_memory("Identity fact two", source="seed", status="active", category="identity")

    rows = db.get_seed_memories()
    categories = [r[2] for r in rows]
    assert categories == ["family", "identity", "identity"]
    # within identity, ordered by id (insertion order)
    identity_contents = [r[1] for r in rows if r[2] == "identity"]
    assert identity_contents == ["Identity fact one", "Identity fact two"]


def test_get_episodic_memories_filters_source_and_status(db):
    db.save_memory("Explicit active", source="explicit", status="active")
    db.save_memory("Explicit pending", source="explicit", status="pending")
    db.save_memory("Implicit pending", source="implicit", status="pending")
    db.save_memory("Seed active", source="seed", status="active", category="identity")

    rows = db.get_episodic_memories()
    contents = [r[1] for r in rows]
    assert contents == ["Explicit active"]


def test_approved_implicit_memory_appears_in_episodic_but_pending_does_not(db):
    db.save_memory("Approved implicit fact", source="implicit", status="pending")
    db.save_memory("Still pending implicit fact", source="implicit", status="pending")

    pending_ids = {r[0]: r[1] for r in db.get_pending_memories()}
    approved_id = next(mid for mid, content in pending_ids.items() if content == "Approved implicit fact")
    assert db.approve_memory(approved_id) is True

    episodic_contents = [r[1] for r in db.get_episodic_memories()]
    assert "Approved implicit fact" in episodic_contents
    assert "Still pending implicit fact" not in episodic_contents

    # source stays 'implicit' after approval, only status changes
    active_rows, _ = db.get_active_memories()
    active_sources = {r[1]: r[3] for r in active_rows}
    assert active_sources["Approved implicit fact"] == "implicit"


def test_get_episodic_memories_ordered_by_id_desc_and_respects_limit(db):
    for i in range(5):
        db.save_memory(f"Fact {i}", source="explicit", status="active")

    rows = db.get_episodic_memories(limit=3)
    contents = [r[1] for r in rows]
    assert contents == ["Fact 4", "Fact 3", "Fact 2"]


def test_pending_vs_active_filtering(db):
    db.save_memory("Pending one", source="implicit", status="pending")
    db.save_memory("Active one", source="explicit", status="active")

    pending = db.get_pending_memories()
    active_rows, active_total = db.get_active_memories()

    assert [r[1] for r in pending] == ["Pending one"]
    assert [r[1] for r in active_rows] == ["Active one"]
    assert active_total == 1


def test_approve_memory_flips_pending_to_active(db):
    db.save_memory("Needs review", source="implicit", status="pending")
    pending_id = db.get_pending_memories()[0][0]

    result = db.approve_memory(pending_id)

    assert result is True
    assert db.get_pending_memories() == []
    active_rows, _ = db.get_active_memories()
    active_contents = [r[1] for r in active_rows]
    assert "Needs review" in active_contents


def test_approve_memory_returns_false_when_not_pending(db):
    assert db.approve_memory(9999) is False

    db.save_memory("Already active", source="explicit", status="active")
    active_rows, _ = db.get_active_memories()
    active_id = active_rows[0][0]
    # already active, not pending, so approving again should report failure
    assert db.approve_memory(active_id) is False


def test_get_active_memories_returns_full_fields(db):
    db.save_memory("Full field check", source="seed", status="active",
                    category="identity", confidence="low")

    rows, total = db.get_active_memories()
    memory_id, content, category, source, confidence, status = rows[0]
    assert content == "Full field check"
    assert category == "identity"
    assert source == "seed"
    assert confidence == "low"
    assert status == "active"
    assert total == 1


def test_get_active_memories_pagination(db):
    for i in range(5):
        db.save_memory(f"Fact {i}", source="explicit", status="active")

    page1, total1 = db.get_active_memories(limit=2, offset=0)
    page2, total2 = db.get_active_memories(limit=2, offset=2)

    assert total1 == 5
    assert total2 == 5
    assert [r[1] for r in page1] == ["Fact 4", "Fact 3"]
    assert [r[1] for r in page2] == ["Fact 2", "Fact 1"]
