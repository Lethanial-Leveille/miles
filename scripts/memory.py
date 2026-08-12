#!/usr/bin/env python3
"""Inspect and correct Nova's memories, without SQL.

    python3 scripts/memory.py list                  # active episodic memories
    python3 scripts/memory.py list --all            # include seed
    python3 scripts/memory.py pending               # awaiting your approval
    python3 scripts/memory.py approve 42
    python3 scripts/memory.py forget 42             # delete outright
    python3 scripts/memory.py fix 42 "corrected text"
    python3 scripts/memory.py chain 42              # what this replaced, and when
    python3 scripts/memory.py temporary 42 2026-12-20   # expires after that date
    python3 scripts/memory.py expire                # mark anything past its date

`fix` supersedes rather than edits: the old row is retired and pointed at the
new one, so "the exam moved to Thursday" stays distinguishable from "the exam
was always Thursday". `chain` is what that buys.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import sqlite3                                       # noqa: E402
from config import DB_PATH                           # noqa: E402
from database import (init_db, approve_memory, delete_memory,   # noqa: E402
                      supersede_memory, get_memory_chain,
                      expire_memories, get_pending_memories)


def _rows(where, params=()):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, content, source, status, volatile, references_date "
        f"FROM memories WHERE {where} ORDER BY id DESC", params
    ).fetchall()
    conn.close()
    return rows


def _show(rows):
    if not rows:
        print("  (none)")
        return
    for mid, content, source, status, volatile, ref in rows:
        flags = []
        if volatile:
            flags.append(f"expires {ref[:10]}" if ref else "volatile, no date")
        if status != "active":
            flags.append(status)
        tail = f"   [{', '.join(flags)}]" if flags else ""
        print(f"  {mid:4}  {source:9} {content[:74]}{tail}")


def cmd_list(include_seed=False):
    if include_seed:
        print("all active memories:")
        _show(_rows("status = 'active'"))
    else:
        print("episodic memories, what Nova learned from talking to you:")
        _show(_rows("status = 'active' AND source IN ('explicit', 'implicit')"))
        print("\n(seed memories hidden; --all to include them)")


def cmd_pending():
    rows = get_pending_memories()
    print("waiting for your approval. Nova inferred these, you did not ask:")
    if not rows:
        print("  (none)")
        return
    for mid, content, category, source, confidence in rows:
        print(f"  {mid:4}  [{confidence}] {content[:80]}")
    print("\napprove <id> to keep, forget <id> to drop")


def cmd_fix(memory_id, new_content):
    new_id = supersede_memory(memory_id, new_content)
    if new_id is None:
        print(f"no memory with id {memory_id}")
        return
    print(f"{memory_id} retired, {new_id} is now current")
    print("the old text is still readable with: chain", new_id)


def cmd_chain(memory_id):
    chain = get_memory_chain(memory_id)
    if not chain:
        print(f"no memory with id {memory_id}")
        return
    print(f"history of memory {memory_id}, newest first:\n")
    for mid, content, status, created, sup_by, sup_at in chain:
        marker = "current" if status == "active" else status
        print(f"  {mid:4}  {marker:10} {created[:16]}  {content[:66]}")
        if sup_at:
            print(f"        replaced {sup_at[:16]}")


def cmd_temporary(memory_id, date):
    """Mark a memory as having a shelf life."""
    conn = sqlite3.connect(DB_PATH)
    updated = conn.execute(
        "UPDATE memories SET volatile = 1, references_date = ? WHERE id = ?",
        (date, memory_id)
    ).rowcount
    conn.commit()
    conn.close()
    if not updated:
        print(f"no memory with id {memory_id}")
        return
    print(f"{memory_id} now expires after {date}")
    print("it stops reaching Nova's prompt the moment that date passes")


def main():
    init_db()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    command, rest = args[0], args[1:]
    if command == "list":
        cmd_list("--all" in rest)
    elif command == "pending":
        cmd_pending()
    elif command == "approve" and rest:
        print("approved" if approve_memory(int(rest[0])) else "not pending, or no such id")
    elif command == "forget" and rest:
        print("deleted" if delete_memory(int(rest[0])) else "no such id")
    elif command == "fix" and len(rest) >= 2:
        cmd_fix(int(rest[0]), " ".join(rest[1:]))
    elif command == "chain" and rest:
        cmd_chain(int(rest[0]))
    elif command == "temporary" and len(rest) == 2:
        cmd_temporary(int(rest[0]), rest[1])
    elif command == "expire":
        print(f"marked {expire_memories()} expired")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
