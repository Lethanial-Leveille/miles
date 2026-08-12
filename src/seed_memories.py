"""Loader for Nova's seed memory corpus.

The corpus itself lives at `config.SEED_PATH` (`data/seed_memories.json`) and is
deliberately **not** in this file and **not** in git.

It used to be. The memories were a list of dict literals right here in `src/`,
which is tracked, on a public remote. That published Lethanial's birthday,
finances, family, and faith, along with the full names of roughly thirty
relatives and friends who were never asked. Splitting the loader from the
payload keeps this file reviewable without publishing the people in it.

`data/` is gitignored, so a fresh clone has the code and no corpus, and
`main()` says so rather than failing on a missing file.

Run manually, from src/:
    python3 seed_memories.py --dry-run   # preview only, writes nothing
    python3 seed_memories.py             # actually inserts

Idempotent: relies on save_memory's own dedup check (skips inserting a
memory whose content already matches an existing active row), so
rerunning this script inserts nothing new.

Note that idempotence is on *exact content*. Editing a memory here and
rerunning does not update the stored row, it inserts a second one and leaves
the first active, so the prompt ends up holding both halves of a
contradiction. Corrections go through `supersede_memory` instead, which is
what `scripts/memory.py fix` is for.
"""

import argparse
import json
import sqlite3

from config import DB_PATH, SEED_PATH
from database import save_memory


REQUIRED_FIELDS = ("content", "category", "volatile", "confidence",
                   "references_date")


def load_memories(path=SEED_PATH):
    """Read the corpus, failing loudly on a malformed entry.

    A missing key here would otherwise surface much later as a KeyError
    partway through a run that has already written rows, leaving the store
    half seeded with no clean way to tell where it stopped."""
    with open(path) as f:
        memories = json.load(f)

    for index, memory in enumerate(memories):
        missing = [field for field in REQUIRED_FIELDS if field not in memory]
        if missing:
            raise ValueError(
                f"entry {index} is missing {', '.join(missing)}: "
                f"{memory.get('content', '<no content>')!r}"
            )
    return memories


def _would_be_duplicate(content):
    """Read only check for --dry-run: does an active row with this exact
    content already exist. Mirrors save_memory's own dedup query but never
    writes anything."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT 1 FROM memories WHERE content = ? AND status = 'active'",
        (content,)
    ).fetchone()
    conn.close()
    return row is not None


def main(dry_run=False):
    try:
        memories = load_memories()
    except FileNotFoundError:
        print(f"No corpus at {SEED_PATH}.")
        print("This file is gitignored on purpose, so a fresh clone will not "
              "have it. Restore it from a backup before seeding.")
        return

    inserted = 0
    skipped = 0

    for mem in memories:
        if dry_run:
            if _would_be_duplicate(mem["content"]):
                skipped += 1
                print(f"[would skip, duplicate] {mem['content']}")
            else:
                inserted += 1
                print(f"[would insert] ({mem['category']}) {mem['content']}")
            continue

        did_insert = save_memory(
            mem["content"],
            source="seed",
            status="active",
            category=mem["category"],
            confidence=mem["confidence"],
            volatile=mem["volatile"],
            references_date=mem["references_date"],
        )
        if did_insert:
            inserted += 1
        else:
            skipped += 1

    verb = "Would insert" if dry_run else "Inserted"
    print(f"\n{verb}: {inserted}, skipped as duplicates: {skipped}, total entries: {len(memories)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Nova's memory store from the corpus at config.SEED_PATH.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be inserted without writing to the database.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
