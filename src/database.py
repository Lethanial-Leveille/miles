import sqlite3
from datetime import datetime
from config import DB_PATH


# ── Schema migrations ──
# Each entry is (version, function). init_db() runs any migration whose
# version is greater than what is recorded in schema_version, in order,
# once. Add new schema changes by appending a new (version, function) pair
# here rather than hand running ALTER TABLE on the Pi.

def _migration_001_memories_v2(conn):
    """Replace the v1 memories table (content/source/created_at/last_referenced/
    relevance_score only) with the richer schema. The table is empty in
    production as of this migration, so drop and recreate instead of
    migrating data."""
    conn.execute("DROP TABLE IF EXISTS memories")
    conn.execute("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT,
            source TEXT NOT NULL DEFAULT 'explicit',
            created_at TEXT NOT NULL,
            references_date TEXT,
            volatile INTEGER NOT NULL DEFAULT 0,
            confidence TEXT NOT NULL DEFAULT 'high',
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by INTEGER,
            last_referenced TEXT,
            relevance_score REAL NOT NULL DEFAULT 1.0
        )
    """)


def _migration_002_verification_log_v2(conn):
    """Add embedded_duration_seconds, wake_confidence, and turn_type to
    verification_log. The table has zero rows in production as of this
    migration (Phase 1 logging just went live and hasn't logged an
    attempt yet), so drop and recreate instead of migrating data."""
    conn.execute("DROP TABLE IF EXISTS verification_log")
    conn.execute("""
        CREATE TABLE verification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            similarity REAL NOT NULL,
            accepted INTEGER NOT NULL,
            threshold_used REAL NOT NULL,
            transcript TEXT,
            duration_seconds REAL NOT NULL,
            embedded_duration_seconds REAL NOT NULL,
            wake_confidence REAL,
            turn_type TEXT NOT NULL
        )
    """)


def _migration_003_verification_outcome(conn):
    """Add an outcome column so attempts that never produced a real embedding
    are separable from genuine accept/reject decisions.

    A clip that trims to no voiced audio still yields a similarity score, and
    that score is meaningless: one such row scored 0.426 and was the only
    rejection in the first ten attempts, which made the sample look like a
    false rejection problem when it was not.

    ALTER rather than recreate, since this table now holds real collected
    data. Existing rows predate the guard and were all genuinely scored."""
    conn.execute(
        "ALTER TABLE verification_log ADD COLUMN outcome TEXT NOT NULL DEFAULT 'scored'"
    )


def _migration_004_acoustic_measures(conn):
    """Log the acoustic correlates of each attempt so distance, loudness, and
    vocal effort can be separated by regression instead of by block labels,
    which depend on reproducing "projecting" consistently by feel.

    snr_db is the one that matters most: simulation puts embedding cosine at
    0.98 for 25dB SNR and 0.68 for 10dB, which is the whole observed score
    range. spectral_tilt is the production side marker, since a close normal
    voice and a far projected voice can arrive at the same level but not with
    the same tilt."""
    for column in ("rms_dbfs REAL", "snr_db REAL", "spectral_tilt REAL"):
        conn.execute(f"ALTER TABLE verification_log ADD COLUMN {column}")


MIGRATIONS = [
    (1, _migration_001_memories_v2),
    (2, _migration_002_verification_log_v2),
    (3, _migration_003_verification_outcome),
    (4, _migration_004_acoustic_measures),
]


def _run_migrations(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        current = 0
    else:
        current = row[0]

    for version, migration_fn in MIGRATIONS:
        if version > current:
            migration_fn(conn)
            conn.execute("UPDATE schema_version SET version = ?", (version,))
            conn.commit()
            print(f"Applied migration {version}: {migration_fn.__name__}", flush=True)
            current = version


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_device TEXT NOT NULL DEFAULT 'pi'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            due_at TEXT,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()

    _run_migrations(conn)

    conn.close()


# ── Memories ──

def save_memory(content, source="explicit", status="active", category=None,
                 confidence="high", volatile=False, references_date=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    existing = c.execute(
        "SELECT id FROM memories WHERE content = ? AND status = 'active'",
        (content,)
    ).fetchone()
    if existing:
        print(f"Duplicate memory skipped (matches active id {existing[0]}): {content}", flush=True)
        conn.close()
        return False

    c.execute(
        """INSERT INTO memories
           (content, category, source, created_at, references_date, volatile, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, category, source, datetime.now().isoformat(), references_date,
         int(volatile), confidence, status)
    )
    conn.commit()
    conn.close()
    print(f"Memory saved ({source}, {status}): {content}", flush=True)
    return True


def get_seed_memories():
    """All active seed memories, grouped for prompt assembly by category then id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content, category FROM memories "
        "WHERE source = 'seed' AND status = 'active' "
        "ORDER BY category, id"
    ).fetchall()
    conn.close()
    return rows


def get_episodic_memories(limit=20):
    """Most recent active explicit or approved-implicit memories. Ordered by
    id, not created_at: seed rows share a timestamp and ties on created_at
    have no defined order in SQLite.

    Includes source='implicit' as well as 'explicit': an implicit row only
    reaches status='active' by being approved through the pending queue, so
    including it here is already correct without rewriting its source,
    which stays as provenance for tracking approved vs rejected model
    generated memories."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content FROM memories "
        "WHERE source IN ('explicit', 'implicit') AND status = 'active' "
        "ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_pending_memories(limit=100):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content, category, source, confidence FROM memories "
        "WHERE status = 'pending' ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_active_memories(limit=50, offset=0):
    """All active memories (seed and explicit/approved-implicit) with full
    metadata, paginated, for the GET /memories listing endpoint. The seed
    corpus alone can run over a hundred rows, so this always returns the
    total active count alongside the page so a caller can tell when there
    is more."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]
    rows = c.execute(
        "SELECT id, content, category, source, confidence, status FROM memories "
        "WHERE status = 'active' ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return rows, total


def approve_memory(memory_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE memories SET status = 'active' WHERE id = ? AND status = 'pending'",
        (memory_id,)
    )
    updated = c.rowcount
    conn.commit()
    conn.close()
    return updated > 0


def delete_memory(memory_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


# ── Conversation history ──

def save_message(role, content, device="pi"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversation_history (role, content, created_at, source_device) VALUES (?, ?, ?, ?)",
        (role, content, datetime.now().isoformat(), device)
    )
    conn.commit()
    conn.close()


def get_recent_messages(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversation_history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]


def get_history(limit: int = 50, offset: int = 0) -> list:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, created_at, source_device FROM conversation_history"
        " ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset)
    )
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    return [
        {"role": r, "content": c, "created_at": t, "source_device": d}
        for r, c, t, d in rows
    ]


# ── Verification log ──

def log_verification(similarity, accepted, threshold_used, transcript, duration_seconds,
                      embedded_duration_seconds, turn_type, wake_confidence=None,
                      outcome='scored', rms_dbfs=None, snr_db=None, spectral_tilt=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO verification_log
           (created_at, similarity, accepted, threshold_used, transcript, duration_seconds,
            embedded_duration_seconds, wake_confidence, turn_type, outcome,
            rms_dbfs, snr_db, spectral_tilt)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), similarity, int(accepted), threshold_used,
         transcript, duration_seconds, embedded_duration_seconds, wake_confidence,
         turn_type, outcome, rms_dbfs, snr_db, spectral_tilt)
    )
    conn.commit()
    conn.close()
