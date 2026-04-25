import sqlite3
from datetime import datetime
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'explicit',
            created_at TEXT NOT NULL,
            last_referenced TEXT,
            relevance_score REAL NOT NULL DEFAULT 1.0
        )
    """)

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
    conn.close()


def save_memory(content, source="explicit"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memories (content, source, created_at) VALUES (?, ?, ?)",
        (content, source, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    print(f"Memory saved ({source}): {content}")


def get_memories(limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, content FROM memories ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


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


def delete_memory(memory_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted > 0


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
