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


MIGRATIONS = [
    (1, _migration_001_memories_v2),
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
        return

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
    """Most recent active explicit memories. Ordered by id, not created_at:
    seed rows share a timestamp and ties on created_at have no defined
    order in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content FROM memories "
        "WHERE source = 'explicit' AND status = 'active' "
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


def get_active_memories(limit=100):
    """All active memories (seed and explicit) with full metadata, for the
    GET /memories listing endpoint."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT id, content, category, source, confidence, status FROM memories "
        "WHERE status = 'active' ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


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
