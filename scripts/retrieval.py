#!/usr/bin/env python3
"""Review what memory retrieval actually returned, and label it.

A miss cannot be detected automatically. An empty result means either the
question was not about Lethanial or the words did not line up, and nothing in
the query separates those two. So every retrieval is logged and judged here by
hand, and the labels accumulate into an eval set built from real questions.

That eval set is the point. Deciding whether to add semantic search on the
strength of two anecdotes is guessing; replaying a few hundred labelled queries
through both methods is a measurement.

    python3 scripts/retrieval.py stats
    python3 scripts/retrieval.py review          # label the unjudged rows
    python3 scripts/retrieval.py review --zero   # only the empty ones, fastest
    python3 scripts/retrieval.py list miss       # everything labelled a miss
"""

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import DB_PATH

VERDICTS = {"g": "good", "m": "miss", "n": "noise", "s": None}


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _memories(conn, ids):
    """Fetch memories in the order they were ranked, not the order SQLite likes.

    IN (...) returns rows by rowid, so zipping the result against the score list
    silently paired the top score with whichever memory happened to have the
    lower id. The review screen showed the right memories with the wrong numbers
    against them, which is a worse failure than showing nothing, because the
    labels drawn from it would have looked fine."""
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    found = {r["id"]: r for r in conn.execute(
        f"SELECT id, content FROM memories WHERE id IN ({q})", ids)}
    return [found[i] for i in ids if i in found]


def cmd_stats():
    conn = _conn()
    total = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
    if not total:
        print("No retrievals logged yet. Talk to Nova and come back.")
        return
    zero = conn.execute(
        "SELECT COUNT(*) FROM retrieval_log WHERE hit_count = 0").fetchone()[0]
    unjudged = conn.execute(
        "SELECT COUNT(*) FROM retrieval_log WHERE verdict IS NULL").fetchone()[0]

    print(f"logged      {total}")
    print(f"zero hit    {zero}  ({zero / total:.0%})")
    print(f"unreviewed  {unjudged}")
    print()
    rows = conn.execute(
        "SELECT verdict, COUNT(*) FROM retrieval_log WHERE verdict IS NOT NULL "
        "GROUP BY verdict ORDER BY COUNT(*) DESC").fetchall()
    if not rows:
        print("Nothing labelled yet. Run: retrieval.py review")
    else:
        judged = sum(r[1] for r in rows)
        for verdict, n in rows:
            print(f"  {verdict:6} {n:4}  ({n / judged:.0%})")
        good = dict(rows).get("good", 0)
        print(f"\nrecall so far: {good / judged:.0%} of {judged} judged")
        print("Treat that as a baseline to beat, not a score.")
    conn.close()


def cmd_review(zero_only):
    conn = _conn()
    sql = "SELECT * FROM retrieval_log WHERE verdict IS NULL"
    if zero_only:
        sql += " AND hit_count = 0"
    sql += " ORDER BY id"
    rows = conn.execute(sql).fetchall()
    if not rows:
        print("Nothing to review.")
        return

    print(f"{len(rows)} to review.  [g]ood  [m]iss  [n]oise  [s]kip  [q]uit\n")
    for row in rows:
        terms = json.loads(row["terms"])
        ids = json.loads(row["returned_ids"])
        scores = json.loads(row["scores"] or "[]")

        print("=" * 72)
        print(f"  {row['query']}")
        print(f"  terms: {terms or '(none survived the stopword list)'}")
        if not ids:
            print("  returned: nothing")
        else:
            for m, s in zip(_memories(conn, ids), scores + [None] * len(ids)):
                print(f"    {s if s is None else f'{s:.2f}'}  {m['content'][:78]}")

        try:
            answer = input("\n  verdict> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nstopped")
            break
        if answer == "q":
            break
        if answer not in VERDICTS:
            print("  skipped")
            continue
        verdict = VERDICTS[answer]
        if verdict is None:
            continue
        note = input("  note (optional)> ").strip() or None
        conn.execute("UPDATE retrieval_log SET verdict = ?, note = ? WHERE id = ?",
                     (verdict, note, row["id"]))
        conn.commit()
        print(f"  -> {verdict}\n")
    conn.close()


def cmd_list(verdict):
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM retrieval_log WHERE verdict = ? ORDER BY id", (verdict,)
    ).fetchall()
    if not rows:
        print(f"Nothing labelled {verdict}.")
        return
    for row in rows:
        print(f"[{row['id']}] {row['query']}")
        print(f"      terms={json.loads(row['terms'])} hits={row['hit_count']}")
        if row["note"]:
            print(f"      note: {row['note']}")
    conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    r = sub.add_parser("review")
    r.add_argument("--zero", action="store_true",
                   help="only rows that returned nothing")
    l = sub.add_parser("list")
    l.add_argument("verdict", choices=["good", "miss", "noise"])

    args = p.parse_args()
    if args.cmd == "stats":
        cmd_stats()
    elif args.cmd == "review":
        cmd_review(args.zero)
    elif args.cmd == "list":
        cmd_list(args.verdict)


if __name__ == "__main__":
    main()
